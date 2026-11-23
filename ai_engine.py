import os, io, re, json, base64, time, requests
import pandas as pd
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

HOST = "generativelanguage." + "googleapis.com"

SYNONYMS = {
    "موزايكو": "تيرازو", "موزايك": "تيرازو", "لياسة": "بياض", "قصارة": "بياض",
    "سملات": "ميدة", "شناجات": "ميدة", "رقاب": "أعمدة", "شمعات": "أعمدة",
    "بلوك": "طوب", "ياجور": "طوب", "ممبرين": "لفافات بيتومين", "زفت": "بيتومين",
    "فلين": "بوليسترين", "بولي سترين": "بوليسترين", "صوف صخري": "عزل حراري",
    "حلق": "إطار", "درفة": "ضلفة", "ردم": "دفان", "انابيب": "مواسير",
    "بلاط": "ارضيات", "سيراميك": "ارضيات", "بورسلين": "ارضيات"
}

def clean_ar(text):
    s = str(text)
    for a, b in [("أ", "ا"), ("إ", "ا"), ("آ", "ا"), ("ة", "ه"), ("ى", "ي")]:
        s = s.replace(a, b)
    for w in ["بالمتر المربع", "بالمتر الطولي", "بالمتر المكعب", "بالمقطوعية", "توريد وعمل", "توريد وصب", "حسب المواصفات"]:
        s = s.replace(w, " ")
    for k, v in SYNONYMS.items():
        if k in s:
            s += f" {v}"
    return re.sub(r"\s+", " ", s).strip()

def norm_unit(u):
    s = str(u).strip().replace(" ", "")
    if any(k in s for k in ["2", "٢", "مربع", "مسطح"]):
        return "م2"
    if any(k in s for k in ["3", "٣", "مكعب"]):
        return "م3"
    if any(k in s for k in ["ط", "ml", "طولي"]):
        return "م.ط"
    if any(k in s for k in ["ق", "ls", "مقطوع"]):
        return "م.ق"
    return s if s else "عدد"

def classify_work_group(desc):
    d = clean_ar(desc)
    if any(k in d for k in ["ازاله", "حفر", "ردم", "دفان", "احلال", "تسويه", "تمهيد"]):
        return "1. الأعمال التمهيدية والحفر والردم"
    if any(k in d for k in ["خرسانه", "اساسات", "قواعد", "ميده", "اعمده", "اسقف", "كمرات", "حديد", "شدة"]):
        return "2. أعمال الخرسانات العادية والمسلحة"
    if any(k in d for k in ["طوب", "مباني", "بلوك", "حوائط", "قواطيع"]):
        return "3. أعمال المباني"
    if any(k in d for k in ["عزل", "بيتومين", "لفافات", "بوليسترين", "رطوبه", "حراري"]):
        return "4. أعمال العزل المائي والحراري"
    if any(k in d for k in ["بياض", "لياسه", "دهان", "طلاء", "ارضيات", "بلاط", "سيراميك", "تيرازو", "رخام", "جبس"]):
        return "5. أعمال البياض والأرضيات والدهانات"
    if any(k in d for k in ["ابواب", "نوافذ", "الومنيوم", "خشب", "حديد مشغول", "حمايه", "درابزين"]):
        return "6. أعمال النجارة والألومنيوم والحديد"
    if any(k in d for k in ["مواسير", "صحي", "سباكه", "خزان", "صرف", "تغذيه", "مرحاض", "مغسله", "مضخه"]):
        return "7. الأعمال الصحية والسباكة"
    if any(k in d for k in ["كهرباء", "كابل", "اسلاك", "اناره", "لوحه", "مفتاح", "بريزه", "تكييف"]):
        return "8. الأعمال الكهربائية والميكانيكية"
    return "9. أعمال متنوعة وتكميلية"

def _get_models(api_key):
    try:
        r = requests.get(f"https://{HOST}/v1beta/models?key={api_key.strip()}", timeout=15)
        if r.status_code == 200:
            ms = [m["name"].replace("models/", "") for m in r.json().get("models", []) if "generateContent" in m.get("supportedGenerationMethods", []) and "gemini" in m.get("name", "") and not any(x in m.get("name", "") for x in ["tts", "audio", "image", "robotics", "embedding", "aqa"])]
            if ms:
                return ms
    except Exception:
        pass
    return ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-3-flash-preview"]

def _parse_image_gemini(args):
    idx, img_bytes, api_key = args
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    prompt = 'Extract all BOQ table rows from this page accurately including handwritten prices. Ignore sub-total or grand-total summary rows. Return JSON only: {"rows": [{"no": "1", "desc": "item description in Arabic", "unit": "m2", "qty": 100, "price": 50, "total": 5000}]}'
    payload = {
        "contents": [{"parts": [{"text": prompt}, {"inline_data": {"mime_type": "image/jpeg", "data": b64}}]}],
        "generationConfig": {"response_mime_type": "application/json", "temperature": 0.0}
    }
    last_err = ""
    for mdl in _get_models(api_key):
        endpoint = f"https://{HOST}/v1beta/models/{mdl}:generateContent"
        for attempt in range(2):
            try:
                r = requests.post(endpoint, headers={"x-goog-api-key": api_key.strip()}, json=payload, timeout=50)
                if r.status_code == 200:
                    cands = r.json().get("candidates", [])
                    if cands:
                        parts = cands[0].get("content", {}).get("parts", [])
                        if parts:
                            data = json.loads(parts[0].get("text", "{}"))
                            rows = data if isinstance(data, list) else next((v for v in data.values() if isinstance(v, list)), [])
                            if rows:
                                return rows
                    break
                elif r.status_code == 503:
                    last_err = f"ضغط مؤقت على {mdl}، أعد المحاولة"
                    time.sleep(2.5)
                elif r.status_code in (400, 404):
                    break
                elif r.status_code == 429:
                    last_err = "تجاوز حد الطلبات المجاني (429)، انتظر دقيقة"
                    time.sleep(3.0)
                else:
                    last_err = f"خطأ ({r.status_code}): {r.text[:120]}"
                    break
            except Exception as e:
                last_err = str(e)
                break
    if last_err:
        raise ValueError(last_err)
    return []

def _read_word_tables(file_bytes):
    from docx import Document
    doc = Document(io.BytesIO(file_bytes))
    all_rows = [[c.text.strip().replace("\n", " ") for c in row.cells] for t in doc.tables for row in t.rows if len(row.cells) >= 3]
    if not all_rows:
        return pd.DataFrame()
    header, data = all_rows[0], all_rows[1:]
    mc = max(len(r) for r in data)
    header = (header + [f"col_{i}" for i in range(mc)])[:mc]
    return pd.DataFrame([(r + [""] * mc)[:mc] for r in data], columns=header)

def read_excel_all_sheets(file_bytes):
    xl = pd.ExcelFile(io.BytesIO(file_bytes))
    frames = []
    for sh in xl.sheet_names:
        sub = xl.parse(sh, header=None).dropna(how="all")
        if not sub.empty and len(sub.columns) >= 2:
            sub = _smart_map_columns(sub)
            if not sub.empty:
                frames.append(sub)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def _smart_map_columns(df):
    df = df.dropna(how="all").reset_index(drop=True)
    if df.empty:
        return pd.DataFrame(columns=["item_no", "description", "unit", "qty", "unit_price", "total_price"])
    for idx in range(min(25, len(df))):
        row_str = " ".join(df.iloc[idx].fillna("").astype(str).tolist())
        if any(w in row_str for w in ["بيان", "وصف", "الأعمال", "الاعمال", "البند", "السعر", "الفئة", "مواصفات"]):
            seen, ucols = {}, []
            for j, c in enumerate(df.iloc[idx].tolist()):
                nm = str(c).strip() or f"col_{j}"
                seen[nm] = seen.get(nm, 0) + 1
                ucols.append(nm if seen[nm] == 1 else f"{nm}_{seen[nm]}")
            df.columns = ucols
            df = df.iloc[idx + 1:].reset_index(drop=True)
            break
    df.columns = [str(c).strip() for c in df.columns]
    cm = {}
    for c in df.columns:
        if any(k in c for k in ["رقم", "م", "no", "كود"]) and "item_no" not in cm.values():
            cm[c] = "item_no"
        elif any(k in c for k in ["بيان", "وصف", "بند", "اعمال", "أعمال", "مواصف", "desc"]) and "description" not in cm.values():
            cm[c] = "description"
        elif any(k in c for k in ["وحدة", "وحده", "unit"]) and "unit" not in cm.values():
            cm[c] = "unit"
        elif any(k in c for k in ["كمية", "كميه", "qty"]) and "qty" not in cm.values():
            cm[c] = "qty"
        elif any(k in c for k in ["فئة", "فئه", "سعر", "price"]) and "اجمالي" not in c and "إجمالي" not in c and "unit_price" not in cm.values():
            cm[c] = "unit_price"
        elif any(k in c for k in ["اجمالي", "إجمالي", "قيمة", "قيمه", "total"]) and "total_price" not in cm.values():
            cm[c] = "total_price"
    df = df.rename(columns=cm)
    if "description" not in df.columns:
        best_c = max(df.columns, key=lambda c: df[c].fillna("").astype(str).str.len().mean())
        df = df.rename(columns={best_c: "description"})
    if "unit_price" not in df.columns and "price" not in df.columns:
        num_cols = [c for c in df.columns if c != "description" and pd.to_numeric(df[c], errors="coerce").notna().sum() > len(df) * 0.15]
        df["unit_price"] = pd.to_numeric(df[num_cols[-1]], errors="coerce").fillna(0.0) if num_cols else 0.0
    return df

def extract_boq(file_bytes, filename, api_key):
    ext = filename.lower().split(".")[-1]
    if ext in ["xlsx", "xls"]:
        df = read_excel_all_sheets(file_bytes)
    elif ext == "docx":
        df = _smart_map_columns(_read_word_tables(file_bytes))
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
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str).str.strip()
    for col in ["qty", "unit_price", "total_price"]:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    df["item_no"] = df["item_no"].apply(lambda x: re.sub(r"[^0-9]", "", str(x).split(".")[0]) or str(x))
    df = df[~df["description"].str.contains("إجمالي|اجمالي|المجموع|مجموع|ينقل|ما قبله|تابع", regex=True, na=False)]
    df = df[(df["description"].str.len() > 3) & (df["qty"] > 0) & (df["unit_price"] > 0)].reset_index(drop=True)
    if (df["item_no"] != "").sum() > 2:
        df = df[df["item_no"] != ""].drop_duplicates(subset=["item_no"], keep="first").reset_index(drop=True)
    df["calc_total"] = (df["qty"] * df["unit_price"]).round(2)
    df["math_diff"] = (df["total_price"] - df["calc_total"]).round(2)
    df["مجموعة الأعمال"] = df["description"].apply(classify_work_group)
    return df

def compare_with_guide(offer_df, guide_df, allowance_pct=0.0):
    cols = ["item_no", "مجموعة الأعمال", "description", "unit", "qty", "unit_price", "total_price", "calc_total", "math_diff", "سعر اللائحة", "اللائحة + العلاوة", "إجمالي اللائحة", "الفرق (د.ل)", "نسبة الفرق %", "البند المطابق باللائحة"]
    if offer_df is None or offer_df.empty:
        return pd.DataFrame(columns=cols)
    if "description" not in guide_df.columns:
        guide_df = _smart_map_columns(guide_df.copy())
    desc_col = "description" if "description" in guide_df.columns else guide_df.columns[0]
    g_descs = guide_df[desc_col].fillna("").astype(str).tolist()
    g_clean = [clean_ar(d) for d in g_descs]
    p_col = "price" if "price" in guide_df.columns else ("unit_price" if "unit_price" in guide_df.columns else guide_df.columns[-1])
    g_prices = pd.to_numeric(guide_df[p_col], errors="coerce").fillna(0.0).tolist()
    g_units = [norm_unit(u) for u in guide_df["unit"].fillna("").astype(str).tolist()] if "unit" in guide_df.columns else [""] * len(g_descs)

    valid = [(d, c, p, u) for d, c, p, u in zip(g_descs, g_clean, g_prices, g_units) if len(c) > 2 and p > 0]
    if valid:
        g_descs, g_clean, g_prices, g_units = map(list, zip(*valid))

    rc_m3 = next((p for d, p, u in zip(g_clean, g_prices, g_units) if u == "م3" and "خرسانه مسلحه" in d and p > 150), 280.0)
    exc_m3 = next((p for d, p, u in zip(g_clean, g_prices, g_units) if u == "م3" and "حفر" in d and 5 <= p <= 50), 15.0)
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    g_mat = vec.fit_transform(g_clean) if g_clean else None
    try:
        sims = cosine_similarity(vec.transform([clean_ar(d) or "بند" for d in offer_df["description"].tolist()]), g_mat) if g_mat is not None else np.zeros((len(offer_df), 1))
    except Exception:
        sims = np.zeros((len(offer_df), max(len(g_clean), 1)))
    mult = 1.0 + (allowance_pct / 100.0)
    gb, ga, gt, dt, dp, mt = [], [], [], [], [], []
    for i, r in offer_df.iterrows():
        desc, u, qty, up, tp = r["description"], r["unit"], r["qty"], r["unit_price"], r["calc_total"]
        gp, gm = 0.0, ""
        if (u == "م.ق" or "مقطوع" in desc) and any(k in desc for k in ["خزان", "حوض", "بيارة"]):
            nums = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)", desc) if 0.5 <= float(x) <= 25.0]
            if len(nums) >= 3:
                L, W, H = nums[0], nums[1], nums[2]
                t = 0.25
                cv = round((L + 2*t)*(W + 2*t)*(2*t) + 2*(L + W + 2*t)*H*t, 2)
                ev = round((L + 1.0)*(W + 1.0)*(H + 0.5), 2)
                gp = round((cv * rc_m3) + (ev * exc_m3), 2)
                gm = f"حساب خرسانة الخزان ({cv} م3 × {rc_m3} + حفر)"
        if gp == 0.0 and g_prices:
            rs = sims[i].copy()
            for j in range(len(rs)):
                if g_units[j] and u and g_units[j] != u:
                    rs[j] *= 0.25
            bj = int(np.argmax(rs))
            if rs[bj] >= 0.22 and g_prices[bj] > 0:
                gp = round(g_prices[bj], 2)
                gm = f"{g_descs[bj][:45]} ({int(rs[bj]*100)}%)"
        if gp == 0.0 and up > 0:
            gp = round(up * 0.92, 2)
            gm = "تقدير سوق محلي (غير مدرج)"
        gpa = round(gp * mult, 2)
        g_tot = round(gpa * qty, 2)
        d_tot = round(tp - g_tot, 2)
        gb.append(gp)
        ga.append(gpa)
        gt.append(g_tot)
        dt.append(d_tot)
        dp.append(round((d_tot / g_tot * 100), 1) if g_tot > 0 else 0.0)
        mt.append(gm)
    out = offer_df.copy()
    out["سعر اللائحة"] = gb
    out["اللائحة + العلاوة"] = ga
    out["إجمالي اللائحة"] = gt
    out["الفرق (د.ل)"] = dt
    out["نسبة الفرق %"] = dp
    out["البند المطابق باللائحة"] = mt
    return out
