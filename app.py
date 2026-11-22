import io
import pandas as pd
import streamlit as st
from ai_engine import extract_boq, compare_with_guide

st.set_page_config(page_title="منظومة تسعير ومقارنة العروض 2026", page_icon="🏗️", layout="centered")
st.markdown("<style>.stApp { direction: rtl; text-align: right; }</style>", unsafe_allow_html=True)

st.title("🏗️ منظومة تسعير ومقارنة العروض 2026")
st.caption("إعداد وتصميم: المهندس طارق محمد الهاشمي | تعمل بالذكاء الاصطناعي السحابي فقط")

api_key = st.secrets.get("GEMINI_API_KEY", "") or st.text_input("🔑 مفتاح Gemini API:", type="password")
allowance = st.number_input("📍 علاوة المنطقة (%):", min_value=0.0, max_value=100.0, value=0.0, step=5.0)
guide_file = st.file_uploader("📘 ملف اللائحة الاسترشادية 2026 (Excel):", type=["xlsx", "xls"])
offer_files = st.file_uploader("📂 ارفع العروض أو صوّر المقايسة (PDF / Excel / Word / صور):", type=["pdf", "png", "jpg", "jpeg", "xlsx", "xls", "docx"], accept_multiple_files=True)

if st.button("🚀 بدء التحليل والمقارنة الآن", type="primary", use_container_width=True):
    if not api_key:
        st.error("يرجى إدخال مفتاح Gemini API أولاً.")
    elif not guide_file or not offer_files:
        st.warning("يرجى رفع ملف اللائحة وعرض سعر واحد على الأقل.")
    else:
        guide_df = pd.read_excel(guide_file)
        guide_df.columns = [str(c).strip() for c in guide_df.columns]
        for c in guide_df.columns:
            if any(k in c for k in ["بيان", "وصف", "بند"]): guide_df = guide_df.rename(columns={c: "description"})
            elif any(k in c for k in ["وحدة"]): guide_df = guide_df.rename(columns={c: "unit"})
            elif any(k in c for k in ["سعر", "فئة", "قيمة"]): guide_df = guide_df.rename(columns={c: "price"})

        summary_rows, all_details = [], {}
        prog = st.progress(0, text="جاري التحليل عبر الذكاء الاصطناعي...")
        for idx, f in enumerate(offer_files):
            try:
                raw_df = extract_boq(f.getvalue(), f.name, api_key)
            except Exception as err:
                st.error(f"⚠️ رسالة سيرفر الذكاء الاصطناعي: {err}")
                st.stop()
            comp_df = compare_with_guide(raw_df, guide_df, allowance)
            all_details[f.name] = comp_df
            math_errs = int((pd.to_numeric(comp_df["math_diff"], errors="coerce").abs() > 0.5).sum()) if not comp_df.empty else 0
            c_tot = float(pd.to_numeric(comp_df["calc_total"], errors="coerce").sum()) if not comp_df.empty else 0.0
            g_tot = float(pd.to_numeric(comp_df["إجمالي اللائحة"], errors="coerce").sum()) if not comp_df.empty else 0.0
            summary_rows.append({
                "اسم العرض": f.name,
                "البنود": int(len(comp_df)),
                "إجمالي العرض المصحح": round(c_tot, 2),
                "إجمالي اللائحة": round(g_tot, 2),
                "الفرق %": round(((c_tot - g_tot) / max(g_tot, 1.0)) * 100.0, 2),
                "أخطاء حسابية": math_errs
            })
            prog.progress((idx + 1) / len(offer_files), text=f"تمت معالجة: {f.name}")

        sum_df = pd.DataFrame(summary_rows).sort_values("إجمالي العرض المصحح")
        st.subheader("🏆 ترتيب العروض والتدقيق الحسابي")
        st.dataframe(sum_df, use_container_width=True, hide_index=True)

        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            sum_df.to_excel(writer, sheet_name="الترتيب والتدقيق", index=False)
            for name, df_det in all_details.items():
                df_det.to_excel(writer, sheet_name=name[:28].replace(":", "").replace("/", ""), index=False)

        st.download_button("📥 تحميل التقرير الشامل (Excel)", data=buf.getvalue(), file_name="تقرير_مقارنة_العروض_2026.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
        for name, df_det in all_details.items():
            with st.expander(f"📋 تفاصيل: {name}"):
                st.dataframe(df_det, use_container_width=True, hide_index=True)
