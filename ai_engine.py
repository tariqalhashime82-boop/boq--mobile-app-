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
                        parts = cands.get("content", {}).get("parts", [])
                        if parts:
                            data = json.loads(parts.get("text", "{}"))
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
    header, data = all_rows, all_rows[1:]
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
