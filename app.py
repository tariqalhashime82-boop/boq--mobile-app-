import io
import pandas as pd
import streamlit as st
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from ai_engine import extract_boq, compare_with_guide, read_excel_all_sheets

st.set_page_config(page_title="منظومة تسعير ومقارنة العروض 2026", page_icon="🏗️", layout="centered")
st.markdown("<style>.stApp { direction: rtl; text-align: right; }</style>", unsafe_allow_html=True)

st.title("🏗️ منظومة تسعير ومقارنة العروض 2026")
st.caption("إعداد وتصميم: المهندس طارق محمد الهاشمي | حسب اللائحة الاسترشادية لسنة 2026")

api_key = st.secrets.get("GEMINI_API_KEY", "") or st.text_input("🔑 مفتاح Gemini API:", type="password")
allowance = st.number_input("📍 علاوة المنطقة (%):", min_value=0.0, max_value=100.0, value=0.0, step=5.0)
guide_file = st.file_uploader("📘 ملف اللائحة الاسترشادية 2026 (Excel):", type=["xlsx", "xls"])
offer_files = st.file_uploader("📂 ارفع العروض أو صوّر المقايسة (PDF / Excel / Word / صور):", type=["pdf", "png", "jpg", "jpeg", "xlsx", "xls", "docx"], accept_multiple_files=True)

def style_sheet(ws, add_totals=False, sum_cols=None):
    ws.views.sheetView.rightToLeft = True
    hdr_fill = PatternFill("solid", fgColor="1B4F72")
    hdr_font = Font(name="Arial", size=11, bold=True, color="FFFFFF")
    tot_fill = PatternFill("solid", fgColor="D6EAF8")
    tot_font = Font(name="Arial", size=11, bold=True, color="000000")
    body_font = Font(name="Arial", size=10)
    thin = Side(border_style="thin", color="BDC3C7")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)

    for cell in ws:
        cell.fill = hdr_fill
        cell.font = hdr_font
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = border
    ws.row_dimensions.height = 28

    max_r = ws.max_row
    for r in range(2, max_r + 1):
        for c in range(1, ws.max_column + 1):
            cell = ws.cell(row=r, column=c)
            cell.font = body_font
            cell.border = border
            cell.alignment = Alignment(horizontal="right" if c in (2, 3, ws.max_column) else "center", vertical="center", wrap_text=True)
            if isinstance(cell.value, (int, float)):
                cell.number_format = "#,##0.00"

    if add_totals and max_r >= 2 and sum_cols:
        tr = max_r + 1
        ws.cell(row=tr, column=1, value="المجموع الكلي").font = tot_font
        for c in range(1, ws.max_column + 1):
            cell = ws.cell(row=tr, column=c)
            cell.fill = tot_fill
            cell.font = tot_font
            cell.border = border
            if c in sum_cols:
                col_ltr = get_column_letter(c)
                cell.value = f"=SUM({col_ltr}2:{col_ltr}{max_r})"
                cell.number_format = "#,##0.00"

    for col in ws.columns:
        ltr = get_column_letter(col.column)
        max_len = max((len(str(cell.value or "")) for cell in col), default=10)
        ws.column_dimensions[ltr].width = min(max(max_len + 4, 12), 45)

if st.button("🚀 بدء التحليل والمقارنة الآن", type="primary", use_container_width=True):
    if not api_key:
        st.error("يرجى إدخال مفتاح Gemini API أولاً.")
    elif not guide_file or not offer_files:
        st.warning("يرجى رفع ملف اللائحة وعرض سعر واحد على الأقل.")
    else:
        guide_df = read_excel_all_sheets(guide_file.getvalue())
        summary_rows, all_details, group_tables = [], {}, []
        prog = st.progress(0, text="جاري التحليل عبر الذكاء الاصطناعي...")
        for idx, f in enumerate(offer_files):
            try:
                raw_df = extract_boq(f.getvalue(), f.name, api_key)
                comp_df = compare_with_guide(raw_df, guide_df, allowance)
            except Exception as err:
                st.error(f"⚠️ تنبيه أثناء قراءة الملف ({f.name}): {err}")
                st.stop()

            ar_df = comp_df.rename(columns={
                "item_no": "رقم البند",
                "description": "بيان الأعمال",
                "unit": "الوحدة",
                "qty": "الكمية",
                "unit_price": "سعر العرض",
                "total_price": "إجمالي العرض المسجل",
                "calc_total": "إجمالي العرض المدقق",
                "math_diff": "الفرق الحسابي"
            })
            cols_order = ["رقم البند", "مجموعة الأعمال", "بيان الأعمال", "الوحدة", "الكمية", "سعر العرض", "إجمالي العرض المسجل", "إجمالي العرض المدقق", "الفرق الحسابي", "سعر اللائحة", "اللائحة + العلاوة", "إجمالي اللائحة", "الفرق (د.ل)", "نسبة الفرق %", "البند المطابق باللائحة"]
            ar_df = ar_df[[c for c in cols_order if c in ar_df.columns]]
            all_details[f.name] = ar_df

            if not ar_df.empty:
                grp = ar_df.groupby("مجموعة الأعمال", as_index=False)[["إجمالي العرض المدقق", "إجمالي اللائحة", "الفرق (د.ل)"]].sum()
                grp["نسبة الفرق %"] = ((grp["الفرق (د.ل)"] / grp["إجمالي اللائحة"].replace(0, 1)) * 100).round(2)
                grp.insert(0, "اسم العرض", f.name)
                group_tables.append(grp)

            math_errs = int((pd.to_numeric(comp_df["math_diff"], errors="coerce").abs() > 0.5).sum()) if not comp_df.empty else 0
            c_tot = float(pd.to_numeric(comp_df["calc_total"], errors="coerce").sum()) if not comp_df.empty else 0.0
            g_tot = float(pd.to_numeric(comp_df["إجمالي اللائحة"], errors="coerce").sum()) if not comp_df.empty else 0.0
            summary_rows.append({
                "اسم العرض": f.name,
                "عدد البنود": int(len(comp_df)),
                "إجمالي العرض المدقق (د.ل)": round(c_tot, 2),
                "إجمالي اللائحة (د.ل)": round(g_tot, 2),
                "الفرق الكلي (د.ل)": round(c_tot - g_tot, 2),
                "نسبة الفرق %": round(((c_tot - g_tot) / max(g_tot, 1.0)) * 100.0, 2),
                "أخطاء حسابية": math_errs
            })
            prog.progress((idx + 1) / len(offer_files), text=f"تمت معالجة: {f.name}")

        sum_df = pd.DataFrame(summary_rows).sort_values("إجمالي العرض المدقق (د.ل)")
        st.subheader("🏆 ترتيب العروض والتدقيق الحسابي")
        st.dataframe(sum_df, use_container_width=True, hide_index=True)

        all_groups_df = pd.concat(group_tables, ignore_index=True) if group_tables else pd.DataFrame()
        if not all_groups_df.empty:
            st.subheader("📊 مجموع بنود الأعمال (مقارنة الأقسام باللائحة)")
            st.dataframe(all_groups_df, use_container_width=True, hide_index=True)

        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            sum_df.to_excel(writer, sheet_name="الترتيب والتدقيق", index=False)
            style_sheet(writer.sheets["الترتيب والتدقيق"], add_totals=False)

            if not all_groups_df.empty:
                all_groups_df.to_excel(writer, sheet_name="مجموع بنود الأعمال", index=False)
                style_sheet(writer.sheets["مجموع بنود الأعمال"], add_totals=True, sum_cols=)

            for name, df_det in all_details.items():
                sh_name = name[:28].replace(":", "").replace("/", "")
                df_det.to_excel(writer, sheet_name=sh_name, index=False)
                style_sheet(writer.sheets[sh_name], add_totals=True, sum_cols=)

        st.download_button("📥 تحميل التقرير الشامل المنسق (Excel)", data=buf.getvalue(), file_name="تقرير_مقارنة_العروض_2026.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
        for name, df_det in all_details.items():
            with st.expander(f"📋 تفاصيل بنود: {name}"):
                st.dataframe(df_det, use_container_width=True, hide_index=True)
