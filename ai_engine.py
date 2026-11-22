import os, io, re, json, base64, time, requests
import pandas as pd
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

GEMINI_URL = "https://googleapis.com"

SYNONYMS = {
    "موزايكو": "تيرازو", "موزايك": "تيرازو", "لياسة": "بياض", "قصارة": "بياض",
    "سملات": "ميدة", "شناجات": "ميدة", "رقاب": "أعمدة", "شمعات": "أعمدة",
    "بلوك": "طوب", "ياجور": "طوب", "ممبرين": "لفافات بيتومين", "زفت": "بيتومين",
    "فلين": "بوليسترين", "بولي سترين": "بوليسترين", "صوف صخري": "عزل حراري",
    "حلق": "إطار", "درفة": "ضلفة", "ردم": "دفان", "انابيب": "مواسير"
}

def clean_ar(text):
    s = str(text)
    for a, b in [("أ","ا"),("إ","ا"),("آ","ا"),("ة","ه"),("ى","ي")]:
        s = s.replace(a, b)
    noise = ["بالمتر المربع", "بالمتر الطولي", "بالمتر المكعب", "بالمقطوعية", "توريد وعمل", "توريد وصب", "حسب المواصفات"]
    for w in noise:
        s = s.replace(w, " ")
    for k, v in SYNONYMS.items():
        if k in s:
            s += f" {v}"
    return re.sub(r"\s+", " ", s).strip()

def norm_unit(u):
    s = str(u).strip().replace(" ", "")
    if any(k in s for k in ["2", "٢", "مربع", "مسطح"]): return "م2"
    if any(k in s for k in ["3", "٣", "مكعب"]): return "م3"
    if any(k in s for k in ["ط", "ml", "طولي"]): return "م.ط"
    if any(k in s for k in ["ق", "ls", "مقطوع"]): return "م.ق"
    return s if s else "عدد"

def _parse_image_gemini(args):
    idx, img_bytes, api_key = args
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    prompt = (
        "استخرج جميع صفوف جدول الكميات والأسعار من هذه الصفحة بدقة تامة من البند الأول حتى الأخير، "
        "بما في ذلك الأسعار المكتوبة بخط اليد. حافظ على المسافات الصحيحة بين الكلمات العربية. "
        'أرجع JSON فقط: {"rows": [{"no": "1", "desc": "بيان الأعمال", "unit": "م2", "qty": 100, "price": 50, "total": 5000}]}'
    )
    payload = {
        "contents": [{"parts": [{"text": prompt}, {"inline_data": {"mime_type": "image/jpeg", "data": b64}}]}],
        "generationConfig": {"response_mime_type": "application/json", "temperature": 0.0}
    }
    for _ in range(4):
        try:
            r = requests.post(GEMINI_URL, headers={"x-goog-api-key": api_key}, json=payload, timeout=30)
            if r.status_code == 200:
                cands = r.json().get("candidates", [])
                if cands:
                    parts = next(iter(cands)).get("content", {}).get("parts", [])
                    if parts:
                        txt = next(iter(parts)).get("text", "{}")
                        rows = json.loads(txt).get("rows", [])
                        if rows:
                            return rows
            elif r.status_code == 429:
                time.sleep(3.0)
        except Exception:
            time.sleep(1.0)
    return []

def _read_word_tables(file_bytes):
    from docx import Document
    doc = Document(io.BytesIO(file_bytes))
    all_rows = [[c.text.strip().replace("\n", " ") for c in row.cells] for t in doc.tables for row in t.rows if len(row.cells) >= 4]
    if not all_rows:
        return pd.DataFrame()
    header, data = next(iter(all_rows)), all_rows[1:]
    mc = max(len(r) for r in data)
    header = (header + [f"col_{i}" for i in range(mc)])[:mc]
    return pd.DataFrame([(r + [""] * mc)[:mc] for r in data], columns=header)

def _smart_map_columns(df):
    df = df.dropna(how="all").reset_index(drop=True)
    if df.empty:
        return pd.DataFrame(columns=["item_no", "description", "unit", "qty", "unit_price", "total_price"])

    for idx in range(min(25, len(df))):
        row_str = " ".join(df.iloc[idx].fillna("").astype(str).tolist())
        if any(w in row_str for w in ["بيان", "وصف", "الأعمال", "الاعمال", "البند", "السعر", "الفئة", "مواصفات"]):
            new_cols = [str(c).strip() or f"col_{j}" for j, c in enumerate(df.iloc[idx].tolist())]
            # ضمان عدم تكرار أسماء الأعمدة
            seen = {}
            unique_cols = []
            for c in new_cols:
                seen[c] = seen.get(c, 0) + 1
                unique_cols.append(c if seen[c] == 1 else f"{c}_{seen[c]}")
            df.columns = unique_cols
            df = df.iloc[idx + 1:].reset_index(drop=True)
            break

    df.columns = [str(c).strip() for c in df.columns]
    cm = {}
    for c in df.columns:
        if any(k in c for k in ["رقم", "م", "no", "كود"]) and "item_no" not in cm.values(): cm[c] = "item_no"
        elif any(k in c for k in ["بيان", "وصف", "بند", "اعمال", "أعمال", "مواصف", "desc"]) and "description" not in cm.values(): cm[c] = "description"
        elif any(k in c for k in ["وحدة", "وحده", "unit"]) and "unit" not in cm.values(): cm[c] = "unit"
        elif any(k in c for k in ["كمية", "كميه", "qty"]) and "qty" not in cm.values(): cm[c] = "qty"
        elif any(k in c for k in ["فئة", "فئه", "سعر", "price"]) and "اجمالي" not in c and "إجمالي" not in c and "unit_price" not in cm.values(): cm[c] = "unit_price"
        elif any(k in c for k in ["اجمالي", "إجمالي", "قيمة", "قيمه", "total"]) and "total_price" not in cm.values(): cm[c] = "total_price"
    df = df.rename(columns=cm)

    if "description" not in df.columns:
        best_c = max(df.columns, key=lambda c: df[c].fillna("").astype(str).str.len().mean())
        df = df.rename(columns={best_c: "description"})

    if "unit_price" not in df.columns and "price" not in df.columns:
        num_cols = [c for c in df.columns if c != "description" and pd.to_numeric(df[c], errors="coerce").notna().sum() > len(df) * 0.2]
        df["unit_price"] = pd.to_numeric(df[num_cols[-1]], errors="coerce").fillna(0.0) if num_cols else 0.0
    return df

def extract_boq(file_bytes, filename, api_key):
    ext = filename.lower().split(".")[-1]
    if ext in ["xlsx", "xls", "docx"]:
        raw = pd.read_excel(io.BytesIO(file_bytes)) if ext in ["xlsx", "xls"] else _read_word_tables(file_bytes)
        df = _smart_map_columns(raw)
    else:
        images = []
        if ext == "pdf":
            import fitz
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            for i in range(len(doc)):
                im = Image.open(io.BytesIO(doc.load_page(i).get_pixmap(dpi=120).tobytes("png"))).convert("RGB")
                if max(im.size) > 1400:
                    im.thumbnail((1400, 1400))
                buf = io.BytesIO()
                im.save(buf, format="JPEG", quality=78)
                images.append((i, buf.getvalue(), api_key))
        else:
            im = Image.open(io.BytesIO(file_bytes)).convert("RGB")
            if max(im.size) > 1400:
                im.thumbnail((1400, 1400))
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=78)
            images.append((0, buf.getvalue(), api_key))

        rows = []
        with ThreadPoolExecutor(max_workers=3) as ex:
            for res in ex.map(_parse_image_gemini, images):
                if res:
                    rows.extend(res)
        df = pd.DataFrame(rows).rename(columns={"no": "item_no", "desc": "description", "price": "unit_price", "total": "total_price"})

    for col in ["item_no", "description", "unit"]:
        if col not in df.columns: df[col] = ""
        df[col] = df[col].fillna("").astype(str).str.strip()
    for col in ["qty", "unit_price", "total_price"]:
        if col not in df.columns: df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df["item_no"] = df["item_no"].apply(lambda x: re.sub(r"[^0-9]", "", next(iter(x.split(".")))) or x)
    df = df[df["description"].str.len() > 2].reset_index(drop=True)
    if (df["item_no"] != "").sum() > 0:
        df = df[df["item_no"] != ""].drop_duplicates(subset=["item_no"], keep="first").reset_index(drop=True)
    df["unit"] = df["unit"].apply(norm_unit)
    df["calc_total"] = (df["qty"] * df["unit_price"]).round(2)
    df["math_diff"] = (df["total_price"] - df["calc_total"]).round(2)
    return df

def compare_with_guide(offer_df, guide_df, allowance_pct=0.0):
    guide_df = _smart_map_columns(guide_df.copy())
    desc_col = "description" if "description" in guide_df.columns else guide_df.columns[0]
    g_descs = guide_df[desc_col].fillna("").astype(str).tolist()
    g_clean = [clean_ar(d) for d in g_descs]
    p_col = "price" if "price" in guide_df.columns else ("unit_price" if "unit_price" in guide_df.columns else guide_df.columns[-1])
    g_prices = pd.to_numeric(guide_df[p_col], errors="coerce").fillna(0.0).tolist()
    g_units = [norm_unit(u) for u in guide_df["unit"].fillna("").astype(str).tolist()] if "unit" in guide_df.columns else [""] * len(g_descs)

    rc_m3 = next((p for d, p, u in zip(g_clean, g_prices, g_units) if u == "م3" and "خرسانه مسلحه" in d and p > 150), 280.0)
    exc_m3 = next((p for d, p, u in zip(g_clean, g_prices, g_units) if u == "م3" and "حفر" in d and 5 <= p <= 50), 15.0)

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    g_mat = vec.fit_transform(g_clean)
    sims = cosine_similarity(vec.transform([clean_ar(d) for d in offer_df["description"].tolist()]), g_mat)
    mult = 1.0 + (allowance_pct / 100.0)
    gb, ga, gt, dt, dp, mt = [], [], [], [], [], []

    for i, r in offer_df.iterrows():
        desc, u, qty, up, tp = r["description"], r["unit"], r["qty"], r["unit_price"], r["calc_total"]
        gp, gm = 0.0, ""
        if (u == "م.ق" or "مقطوع" in desc) and any(k in desc for k in ["خزان", "حوض", "بيارة"]):
            nums = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)", desc) if 0.5 <= float(x) <= 25.0]
            if len(nums) >= 3:
                L, W, H = nums, nums, nums
                t = 0.25
                cv = round((L + 2*t)*(W + 2*t)*(2*t) + 2*(L + W + 2*t)*H*t, 2)
                ev = round((L + 1.0)*(W + 1.0)*(H + 0.5), 2)
                gp = round((cv * rc_m3) + (ev * exc_m3), 2)
