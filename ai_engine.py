import os, io, re, json, base64, time, requests
import pandas as pd, numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

GEMINI_URL = "https://googleapis.com"
SYNONYMS = {"???????":"??????","??????":"??????","?????":"????","?????":"????","?????":"????","??????":"????","????":"?????","?????":"?????","????":"???","?????":"???","??????":"?????? ???????","???":"???????","????":"?????????","???? ?????":"?????????","??? ????":"??? ?????","???":"????","????":"????","???":"????","??????":"??????"}

def clean_ar(text):
    s = str(text)
    for a, b in [("?","?"),("?","?"),("?","?"),("?","?"),("?","?")]: s = s.replace(a, b)
    for w in ["?????? ??????","?????? ??????","?????? ??????","??????????","????? ????","????? ???","??? ?????????"]: s = s.replace(w, " ")
    for k, v in SYNONYMS.items():
        if k in s: s += f" {v}"
    return re.sub(r"\s+", " ", s).strip()

def norm_unit(u):
    s = str(u).strip().replace(" ", "")
    if any(k in s for k in ["2","?","????","????"]): return "?2"
    if any(k in s for k in ["3","?","????"]): return "?3"
    if any(k in s for k in ["?","ml","????"]): return "?.?"
    if any(k in s for k in ["?","ls","?????"]): return "?.?"
    return s if s else "???"

def _parse_image_gemini(args):
    idx, img_bytes, api_key = args
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    prompt = '?????? ???? ???? ???? ??????? ???????? ?? ??? ?????? ???? ???? ?? ????? ????? ??? ??????? ??? ?? ??? ??????? ???????? ??? ????. ???? ??? ???????? ??????? ??? ??????? ???????. ???? JSON ???: {"rows": [{"no": "1", "desc": "???? ???????", "unit": "?2", "qty": 100, "price": 50, "total": 5000}]}'
    payload = {"contents": [{"parts": [{"text": prompt}, {"inline_data": {"mime_type": "image/jpeg", "data": b64}}]}], "generationConfig": {"response_mime_type": "application/json", "temperature": 0.0}}
    for _ in range(4):
        try:
            r = requests.post(GEMINI_URL, headers={"x-goog-api-key": api_key}, json=payload, timeout=30)
            if r.status_code == 200:
                cands = r.json().get("candidates", [])
                if cands:
                    txt = next(iter(next(iter(cands)).get("content", {}).get("parts", [{}]))).get("text", "{}")
                    rows = json.loads(txt).get("rows", [])
                    if rows: return rows
            elif r.status_code == 429: time.sleep(3.0)
        except Exception: time.sleep(1.0)
    return []

def _read_word_tables(file_bytes):
    from docx import Document
    doc = Document(io.BytesIO(file_bytes))
    all_rows = [[c.text.strip().replace("\n", " ") for c in row.cells] for t in doc.tables for row in t.rows if len(row.cells) >= 4]
    if not all_rows: return pd.DataFrame()
    header, data = next(iter(all_rows)), all_rows[1:]
    mc = max(len(r) for r in data)
    header = (header + [f"col_{i}" for i in range(mc)])[:mc]
    return pd.DataFrame([(r + [""]*mc)[:mc] for r in data], columns=header)

def extract_boq(file_bytes, filename, api_key):
    ext = filename.lower().split(".")[-1]
    if ext in ["xlsx", "xls", "docx"]:
        df = pd.read_excel(io.BytesIO(file_bytes)) if ext in ["xlsx", "xls"] else _read_word_tables(file_bytes)
        df.columns = [str(c).strip() for c in df.columns]
        cm = {}
        for c in df.columns:
            if any(k in c for k in ["???","?","no"]): cm[c] = "item_no"
            elif any(k in c for k in ["????","???","???","desc"]): cm[c] = "description"
            elif any(k in c for k in ["????","unit"]): cm[c] = "unit"
            elif any(k in c for k in ["????","qty"]): cm[c] = "qty"
            elif any(k in c for k in ["???","???","price"]) and "??????" not in c and "??????" not in c: cm[c] = "unit_price"
            elif any(k in c for k in ["??????","??????","????","total"]): cm[c] = "total_price"
        df = df.rename(columns=cm)
    else:
        images = []
        if ext == "pdf":
            import fitz
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            for i in range(len(doc)):
                im = Image.open(io.BytesIO(doc.load_page(i).get_pixmap(dpi=120).tobytes("png"))).convert("RGB")
                if max(im.size) > 1400: im.thumbnail((1400, 1400))
                buf = io.BytesIO(); im.save(buf, format="JPEG", quality=78)
                images.append((i, buf.getvalue(), api_key))
        else:
            im = Image.open(io.BytesIO(file_bytes)).convert("RGB")
            if max(im.size) > 1400: im.thumbnail((1400, 1400))
            buf = io.BytesIO(); im.save(buf, format="JPEG", quality=78)
            images.append((0, buf.getvalue(), api_key))
        rows = []
        with ThreadPoolExecutor(max_workers=3) as ex:
            for res in ex.map(_parse_image_gemini, images):
                if res: rows.extend(res)
        df = pd.DataFrame(rows).rename(columns={"no":"item_no","desc":"description","price":"unit_price","total":"total_price"})

    for col in ["item_no","description","unit"]:
        if col not in df.columns: df[col] = ""
        df[col] = df[col].fillna("").astype(str).str.strip()
    for col in ["qty","unit_price","total_price"]:
        if col not in df.columns: df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    df["item_no"] = df["item_no"].apply(lambda x: re.sub(r"[^0-9]", "", next(iter(x.split(".")))) or x)
    df = df[df["item_no"] != ""].drop_duplicates(subset=["item_no"], keep="first").reset_index(drop=True)
    df["unit"] = df["unit"].apply(norm_unit)
    df["calc_total"] = (df["qty"] * df["unit_price"]).round(2)
    df["math_diff"] = (df["total_price"] - df["calc_total"]).round(2)
    return df

def compare_with_guide(offer_df, guide_df, allowance_pct=0.0):
    g_descs = guide_df["description"].fillna("").astype(str).tolist()
    g_clean = [clean_ar(d) for d in g_descs]
    p_col = "price" if "price" in guide_df.columns else guide_df.columns[-1]
    g_prices = pd.to_numeric(guide_df[p_col], errors="coerce").fillna(0.0).tolist()
    g_units = [norm_unit(u) for u in guide_df["unit"].fillna("").astype(str).tolist()] if "unit" in guide_df.columns else [""]*len(g_descs)
    rc_m3 = next((p for d,p,u in zip(g_clean,g_prices,g_units) if u=="?3" and "?????? ?????" in d and p>150), 280.0)
    exc_m3 = next((p for d,p,u in zip(g_clean,g_prices,g_units) if u=="?3" and "???" in d and 5<=p<=50), 15.0)
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    g_mat = vec.fit_transform(g_clean)
    sims = cosine_similarity(vec.transform([clean_ar(d) for d in offer_df["description"].tolist()]), g_mat)
    mult = 1.0 + (allowance_pct / 100.0)
    gb, ga, gt, dt, dp, mt = [], [], [], [], [], []
    for i, r in offer_df.iterrows():
        desc, u, qty, up, tp = r["description"], r["unit"], r["qty"], r["unit_price"], r["calc_total"]
        gp, gm = 0.0, ""
        if (u == "?.?" or "?????" in desc) and any(k in desc for k in ["????","???","?????"]):
            nums = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)", desc) if 0.5 <= float(x) <= 25.0]
            if len(nums) >= 3:
                L, W, H = nums, nums, nums; t = 0.25
                cv = round((L+2*t)*(W+2*t)*(2*t) + 2*(L+W+2*t)*H*t, 2)
                ev = round((L+1.0)*(W+1.0)*(H+0.5), 2)
                gp = round((cv * rc_m3) + (ev * exc_m3), 2)
                gm = f"???? ?????? ?????? ({cv} ?3 ? {rc_m3} + ???)"
        if gp == 0.0:
            rs = sims[i].copy()
            for j in range(len(rs)):
                if g_units[j] and u and g_units[j] != u: rs[j] *= 0.40
            bj = int(np.argmax(rs))
            if rs[bj] >= 0.16 and g_prices[bj] > 0:
                gp, gm = round(g_prices[bj], 2), f"{g_descs[bj][:45]} ({int(rs[bj]*100)}%)"
        if gp == 0.0 and up > 0: gp, gm = round(up * 0.92, 2), "????? ??? ????"
        gpa = round(gp * mult, 2); g_tot = round(gpa * qty, 2); d_tot = round(tp - g_tot, 2)
        gb.append(gp); ga.append(gpa); gt.append(g_tot); dt.append(d_tot)
        dp.append(round((d_tot / g_tot * 100), 1) if g_tot > 0 else 0.0); mt.append(gm)
    out = offer_df.copy()
    out["??? ???????"], out["??????? + ???????"], out["?????? ???????"] = gb, ga, gt
    out["????? (?.?)"], out["???? ????? %"], out["????? ??????? ????????"] = dt, dp, mt
    return out
