# right_itech_app.py
# Full, fixed Right iTech Streamlit app (all tabs + exports + robust loading)

import os
import io
import tempfile
from datetime import date

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# reportlab for PDFs
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4

# -------------------------
# Config
# -------------------------
st.set_page_config(page_title="Right iTech Student Insights", layout="wide", initial_sidebar_state="expanded")

PALETTE = px.colors.qualitative.Set2
ATT_PRESENT_COLOR = "#2ca02c"
ATT_ABSENT_COLOR = "#d62728"

# -------------------------
# Helpers: safe CSV loading
# -------------------------
@st.cache_data
def safe_read_csv(uploaded_file, fallback_path):
    """
    If uploaded_file is provided, read it. Else, read fallback_path only if exists.
    Returns empty DataFrame if neither is available or reading fails.
    """
    try:
        if uploaded_file is not None:
            return pd.read_csv(uploaded_file)
        if fallback_path and os.path.exists(fallback_path):
            return pd.read_csv(fallback_path)
    except Exception:
        return pd.DataFrame()
    return pd.DataFrame()

# -------------------------
# Small utility functions
# -------------------------
def parse_attendance_dates(df, date_col="Date"):
    df = df.copy()
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")
    return df

def standardize_attendance_status(df, status_col="Status"):
    df = df.copy()
    if status_col in df.columns:
        # map common tokens
        df["_present_flag_"] = df[status_col].astype(str).str.upper().map({
            "P": 1, "PRESENT": 1, "1": 1,
            "A": 0, "ABSENT": 0, "0": 0
        })
        # fallback from first character
        df["_present_flag_"] = df["_present_flag_"].fillna(df[status_col].astype(str).str[0].map({"P":1,"A":0}))
    else:
        df["_present_flag_"] = np.nan
    return df

def assign_subject_colors(subjects):
    subjects = sorted(set([s for s in subjects if pd.notna(s)]))
    mapping = {}
    for i, s in enumerate(subjects):
        mapping[s] = PALETTE[i % len(PALETTE)]
    return mapping

# -------------------------
# File Upload UI (sidebar)
# -------------------------
st.sidebar.header("Upload datasets (optional)")
uploaded_marks = st.sidebar.file_uploader("Marks CSV (cleanest_marks.csv)", type=["csv"])
uploaded_att = st.sidebar.file_uploader("Attendance CSV (combined_attendance.csv)", type=["csv"])

# fallback paths - only used if exist
FALLBACK_MARKS = "/mnt/data/cleanest_marks.csv"
FALLBACK_ATT = "/mnt/data/combined_attendance.csv"

marks_df = safe_read_csv(uploaded_marks, FALLBACK_MARKS)
att_df = safe_read_csv(uploaded_att, FALLBACK_ATT)

if marks_df.empty and att_df.empty:
    st.sidebar.error("No data loaded. Upload files or ensure fallback files exist at the server path.")
# continue - UI will show messages where appropriate

# -------------------------
# Normalize columns & types
# -------------------------
# strip whitespace from column names
marks_df.columns = [c.strip() for c in marks_df.columns] if not marks_df.empty else []
att_df.columns = [c.strip() for c in att_df.columns] if not att_df.empty else []

# ensure commonly expected columns (graceful)
if not marks_df.empty:
    if "Marks" in marks_df.columns:
        marks_df["Marks"] = pd.to_numeric(marks_df["Marks"], errors="coerce")
    else:
        marks_df["Marks"] = np.nan
    if "FullMarks" in marks_df.columns:
        marks_df["FullMarks"] = pd.to_numeric(marks_df["FullMarks"], errors="coerce")
    else:
        marks_df["FullMarks"] = np.nan
    if "WasAbsent" in marks_df.columns:
        marks_df["WasAbsent"] = marks_df["WasAbsent"].astype(str)
    else:
        marks_df["WasAbsent"] = "False"

if not att_df.empty:
    att_df = parse_attendance_dates(att_df, date_col="Date")
    att_df = standardize_attendance_status(att_df, status_col="Status")

# attendance summary per student (if applicable)
if not att_df.empty and "ID" in att_df.columns:
    att_summary = att_df.groupby(["ID","Roll","Name"]).agg(
        total_days=("Date","nunique"),
        present_count=("_present_flag_","sum")
    ).reset_index()
    att_summary["attendance_rate"] = att_summary["present_count"] / att_summary["total_days"]
else:
    att_summary = pd.DataFrame()

# subject & student summaries
if not marks_df.empty and "Subject" in marks_df.columns:
    subj_summary = marks_df.groupby("Subject").agg(
        exams=("ExamNumber","nunique"),
        avg_score=("Marks","mean"),
        entries=("Marks","count")
    ).reset_index()
    SUBJECT_COLORS = assign_subject_colors(subj_summary["Subject"].unique())
else:
    subj_summary = pd.DataFrame()
    SUBJECT_COLORS = {}

def student_overview_marks(df):
    if df.empty:
        return pd.DataFrame()
    s = df.groupby(["ID","Roll","Name"]).agg(
        total_exams=("ExamNumber","nunique"),
        avg_score=("Marks","mean"),
        total_entries=("Marks","count"),
        absent_count=("WasAbsent", lambda x: x.astype(str).str.lower().isin(["true","1","yes"]).sum())
    ).reset_index()
    return s

student_mark_summary = student_overview_marks(marks_df)

# -------------------------
# Controls (sidebar)
# -------------------------
st.sidebar.header("Filters & UI options")
min_score = st.sidebar.number_input("Minimum score threshold (for flags)", min_value=0, max_value=100, value=40)
min_att_pct = st.sidebar.slider("Minimum attendance threshold (%)", min_value=0, max_value=100, value=75)
show_explanations_toggle = st.sidebar.checkbox("Auto-expand explanations by default", value=False)

# -------------------------
# Color guide
# -------------------------
st.markdown("<h2>Right iTech — Student Insights</h2>", unsafe_allow_html=True)
if SUBJECT_COLORS:
    cols_html = "<div style='display:flex; flex-wrap:wrap; gap:10px; margin-bottom:8px;'>"
    for sub, col in SUBJECT_COLORS.items():
        cols_html += f"<div style='display:flex;align-items:center;gap:6px;'><div style='width:18px;height:18px;background:{col};border-radius:4px;'></div><div>{sub}</div></div>"
    cols_html += "</div>"
    st.markdown(cols_html, unsafe_allow_html=True)

st.write("---")

# -------------------------
# Tabs
# -------------------------
tabs = st.tabs(["Class overview","Single student overview","Compare students","Attendance explorer","Insights & exports"])

# -------------------------
# Tab 1: Class overview
# -------------------------
with tabs[0]:
    st.header("Class overview")

    # overall distribution (simple)
    st.subheader("Score distribution — (simple)")
    if marks_df.empty:
        st.info("No marks data available.")
    else:
        dist_df = marks_df.copy()
        fig = px.histogram(dist_df, x="Marks", nbins=25, color="Subject" if "Subject" in dist_df.columns else None, color_discrete_map=SUBJECT_COLORS or None)
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("Explanation", expanded=show_explanations_toggle):
            st.write("Histogram shows how scores are distributed across the class and subjects. Hover to see counts.")

    # subject averages (simple)
    st.subheader("Subject averages (simple)")
    if subj_summary.empty:
        st.info("Not enough subject data.")
    else:
        fig = px.bar(subj_summary, x="Subject", y="avg_score", color="Subject", color_discrete_map=SUBJECT_COLORS)
        fig.update_layout(yaxis_title="Average score")
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("Explanation", expanded=show_explanations_toggle):
            st.write("Bars show average class score per subject. Useful to spot subjects where class is strong or weak.")

    # correlation heatmap (complex)
    st.subheader("Subject correlation (complex)")
    if not marks_df.empty and "Subject" in marks_df.columns:
        pivot = marks_df.groupby(["ID","Name","Subject"])["Marks"].mean().reset_index()
        wide = pivot.pivot_table(index=["ID","Name"], columns="Subject", values="Marks")
        if wide.shape[1] >= 2:
            corr = wide.corr()
            fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu")
            st.plotly_chart(fig, use_container_width=True)
            with st.expander("Explanation", expanded=show_explanations_toggle):
                st.write("Correlation matrix shows how scores in subjects relate. Blue=positive, Red=negative.")
        else:
            st.info("Need at least 2 distinct subjects to compute correlations.")
    else:
        st.info("No marks/subject data to compute correlation.")

# -------------------------
# Tab 2: Single Student overview
# -------------------------
with tabs[1]:
    st.header("Single student overview")

    # choose student
    name_candidates = []
    if not marks_df.empty and "Name" in marks_df.columns:
        name_candidates = sorted(marks_df["Name"].dropna().unique().tolist())
    elif not att_df.empty and "Name" in att_df.columns:
        name_candidates = sorted(att_df["Name"].dropna().unique().tolist())

    if not name_candidates:
        st.info("No student names found in uploaded datasets.")
    else:
        student = st.selectbox("Select student", name_candidates)
        # student data
        s_marks = marks_df[marks_df["Name"] == student] if not marks_df.empty and "Name" in marks_df.columns else pd.DataFrame()
        s_att = att_df[att_df["Name"] == student] if not att_df.empty and "Name" in att_df.columns else pd.DataFrame()

        # profile block - use triple-quoted markdown to avoid f-string issues
        if not s_marks.empty:
            sid = s_marks["ID"].iloc[0] if "ID" in s_marks.columns else "N/A"
            sroll = s_marks["Roll"].iloc[0] if "Roll" in s_marks.columns else "N/A"
        elif not s_att.empty:
            sid = s_att["ID"].iloc[0] if "ID" in s_att.columns else "N/A"
            sroll = s_att["Roll"].iloc[0] if "Roll" in s_att.columns else "N/A"
        else:
            sid = "N/A"
            sroll = "N/A"

        st.markdown(
            f"""
            **Name:** {student}  
            **ID:** {sid}  
            **Roll:** {sroll}
            """
        )

        # Simple: subject-wise average (bar) + attendance pie
        st.subheader("Simple visuals")
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("**Average marks by subject (simple)**")
            if not s_marks.empty and "Subject" in s_marks.columns:
                subj_avg = s_marks.groupby("Subject")["Marks"].mean().reset_index()
                fig = px.bar(subj_avg, x="Subject", y="Marks", color="Subject", color_discrete_map=SUBJECT_COLORS)
                st.plotly_chart(fig, use_container_width=True)
                with st.expander("Explanation", expanded=show_explanations_toggle):
                    st.write("This bar chart shows the selected student's average score per subject.")

            else:
                st.info("No subject marks for this student.")

        with c2:
            st.markdown("**Attendance breakdown (simple)**")
            if not s_att.empty and "_present_flag_" in s_att.columns:
                present = int(s_att["_present_flag_"].sum())
                absent = int((s_att["_present_flag_"] == 0).sum())
                pie_df = pd.DataFrame({"status": ["Present", "Absent"], "count": [present, absent]})
                fig = px.pie(pie_df, names="status", values="count", color="status",
                             color_discrete_map={"Present": ATT_PRESENT_COLOR, "Absent": ATT_ABSENT_COLOR})
                st.plotly_chart(fig, use_container_width=True)
                with st.expander("Explanation", expanded=show_explanations_toggle):
                    st.write("Proportion of days present vs absent for the student.")
            else:
                st.info("No attendance records for this student.")

        # Complex: trend + timeline
        st.subheader("Complex visuals")
        st.markdown("**Marks trend across exams**")
        if not s_marks.empty and "ExamNumber" in s_marks.columns:
            trend = s_marks.groupby(["ExamNumber", "ExamType"])["Marks"].mean().reset_index()
            fig = px.line(trend.sort_values("ExamNumber"), x="ExamNumber", y="Marks", markers=True, color_discrete_sequence=PALETTE)
            st.plotly_chart(fig, use_container_width=True)
            with st.expander("Explanation", expanded=show_explanations_toggle):
                st.write("This shows how the student's average across exams evolved over time.")
        else:
            st.info("No exam sequence info available to draw trend.")

        st.markdown("**Attendance timeline (dots)**")
        if not s_att.empty and "Date" in s_att.columns:
            att_sorted = s_att.sort_values("Date")
            fig = px.scatter(att_sorted, x="Date", y="_present_flag_", labels={"_present_flag_":"Present (1) / Absent (0)"})
            fig.update_yaxes(tickmode="array", tickvals=[0,1], ticktext=["Absent","Present"])
            st.plotly_chart(fig, use_container_width=True)
            with st.expander("Explanation", expanded=show_explanations_toggle):
                st.write("Daily attendance plotted over time. Look for streaks or long gaps.")
        else:
            st.info("No attendance dates available.")

        # Export: Excel and PDF
        st.markdown("---")
        st.subheader("Export student report")
        col_x, col_y = st.columns(2)

        def generate_student_excel(student_name):
            out = io.BytesIO()
            with pd.ExcelWriter(out, engine="openpyxl") as writer:
                s_m = marks_df[marks_df["Name"] == student_name] if not marks_df.empty else pd.DataFrame()
                s_a = att_df[att_df["Name"] == student_name] if not att_df.empty else pd.DataFrame()
                s_m.to_excel(writer, index=False, sheet_name="Marks")
                s_a.to_excel(writer, index=False, sheet_name="Attendance")
            out.seek(0)
            return out.getvalue()

        def generate_student_pdf(student_name):
            s_m = marks_df[marks_df["Name"] == student_name] if not marks_df.empty else pd.DataFrame()
            s_a = att_df[att_df["Name"] == student_name] if not att_df.empty else pd.DataFrame()

            buffer = io.BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=A4)
            styles = getSampleStyleSheet()
            story = []
            story.append(Paragraph(f"Student Report — {student_name}", styles["Title"]))
            story.append(Spacer(1, 12))
            if not s_m.empty:
                story.append(Paragraph(f"Average marks: {s_m['Marks'].mean():.2f}", styles["Normal"]))
            if not s_a.empty and "_present_flag_" in s_a.columns:
                p = int(s_a["_present_flag_"].sum())
                t = s_a.shape[0]
                story.append(Paragraph(f"Attendance: {p}/{t} ({(p/t*100) if t>0 else 0:.1f}%)", styles["Normal"]))
            story.append(Spacer(1,12))

            # Try to embed small charts (requires plotly image export availability)
            imgs_written = []
            try:
                if not s_m.empty and "Subject" in s_m.columns:
                    subj_avg = s_m.groupby("Subject")["Marks"].mean().reset_index()
                    f = px.bar(subj_avg, x="Subject", y="Marks", color="Subject", color_discrete_map=SUBJECT_COLORS)
                    img_bytes = f.to_image(format="png")
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                    tmp.write(img_bytes)
                    tmp.flush()
                    imgs_written.append(tmp.name)
                    story.append(RLImage(tmp.name, width=450, height=200))
                    story.append(Spacer(1,12))
                if not s_a.empty and "Date" in s_a.columns:
                    att_sorted = s_a.sort_values("Date")
                    f2 = px.scatter(att_sorted, x="Date", y="_present_flag_")
                    img2 = f2.to_image(format="png")
                    tmp2 = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                    tmp2.write(img2)
                    tmp2.flush()
                    imgs_written.append(tmp2.name)
                    story.append(RLImage(tmp2.name, width=450, height=150))
                    story.append(Spacer(1,12))
            except Exception:
                # If image export isn't available, skip images gracefully
                pass

            doc.build(story)
            buffer.seek(0)
            pdf_bytes = buffer.getvalue()
            # cleanup temp files if any
            try:
                for p in imgs_written:
                    os.unlink(p)
            except Exception:
                pass
            return pdf_bytes

        with col_x:
            excel_data = generate_student_excel(student)
            st.download_button("Download Excel report", data=excel_data, file_name=f"{student}_report.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        with col_y:
            pdf_data = generate_student_pdf(student)
            st.download_button("Download PDF report", data=pdf_data, file_name=f"{student}_report.pdf", mime="application/pdf")

# -------------------------
# Tab 3: Compare students
# -------------------------
with tabs[2]:
    st.header("Comparison between students")

    # selector
    available_names = sorted(set(marks_df["Name"].dropna().tolist()) if "Name" in marks_df.columns else [])
    selection = st.multiselect("Select up to 6 students", options=available_names, max_selections=6)
    if len(selection) < 2:
        st.info("Select two or more students to compare.")
    else:
        comp_df = marks_df[marks_df["Name"].isin(selection)]
        # simple: grouped average bar
        st.subheader("Average score (simple)")
        avg_by_student = comp_df.groupby("Name")["Marks"].mean().reset_index()
        fig = px.bar(avg_by_student, x="Name", y="Marks", color="Name", color_discrete_sequence=PALETTE)
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("Explanation", expanded=show_explanations_toggle):
            st.write("Bar chart compares average scores among selected students.")

        # complex: radar
        st.subheader("Radar (complex)")
        pivot = comp_df.groupby(["Name","Subject"])["Marks"].mean().reset_index()
        wide = pivot.pivot_table(index="Name", columns="Subject", values="Marks").fillna(0)
        if wide.shape[1] >= 3:
            categories = list(wide.columns)
            fig = go.Figure()
            maxv = np.nanmax(wide.values) if wide.values.size else 1
            for idx, row in wide.iterrows():
                values = row.values.tolist()
                norm = [val/maxv if maxv>0 else 0 for val in values]
                fig.add_trace(go.Scatterpolar(r=norm, theta=categories, fill="toself", name=str(idx)))
            fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1])))
            st.plotly_chart(fig, use_container_width=True)
            with st.expander("Explanation", expanded=show_explanations_toggle):
                st.write("Radar normalizes subject averages so you can compare relative strengths.")
        else:
            st.info("Not enough distinct subjects to draw radar. Need at least 3.")

        # scatter: attendance vs avg score if attendance available
        if not att_summary.empty and "ID" in marks_df.columns:
            merged = marks_df.groupby(["ID","Name"])["Marks"].mean().reset_index().merge(att_summary[["ID","attendance_rate"]], on="ID", how="left")
            merged_sel = merged[merged["Name"].isin(selection)]
            if not merged_sel.empty:
                st.subheader("Attendance vs Average score")
                fig = px.scatter(merged_sel, x="attendance_rate", y="Marks", text="Name", size="Marks")
                st.plotly_chart(fig, use_container_width=True)
                with st.expander("Explanation", expanded=show_explanations_toggle):
                    st.write("Each point is a student: x=attendance rate, y=average score. See if attendance relates to performance.")

# -------------------------
# Tab 4: Attendance explorer
# -------------------------
with tabs[3]:
    st.header("Attendance explorer")

    if att_df.empty:
        st.info("No attendance data available.")
    else:
        # date-range selector
        min_date = att_df["Date"].min().date() if "Date" in att_df.columns and not att_df["Date"].isna().all() else date.today()
        max_date = att_df["Date"].max().date() if "Date" in att_df.columns and not att_df["Date"].isna().all() else date.today()
        dr = st.date_input("Select date range", value=(min_date, max_date))
        # date_input may return tuple or two items
        try:
            if isinstance(dr, (list, tuple)) and len(dr) == 2:
                start_d, end_d = dr[0], dr[1]
            else:
                start_d, end_d = dr, dr
        except Exception:
            start_d, end_d = min_date, max_date

        mask = (att_df["Date"].dt.date >= start_d) & (att_df["Date"].dt.date <= end_d)
        att_filtered = att_df[mask]

        # class attendance over time
        st.subheader("Class attendance over time")
        att_over_time = att_filtered.groupby(att_filtered["Date"].dt.date)["_present_flag_"].mean().reset_index()
        att_over_time.columns = ["Date", "attendance_rate"]
        fig = px.line(att_over_time, x="Date", y="attendance_rate", markers=True)
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("Explanation", expanded=show_explanations_toggle):
            st.write("Daily average attendance for the class in the selected range.")

        # monthly summary simple
        st.subheader("Monthly attendance (simple)")
        try:
            att_filtered["month"] = att_filtered["Date"].dt.to_period("M").astype(str)
            monthly = att_filtered.groupby("month")["_present_flag_"].mean().reset_index()
            fig = px.bar(monthly, x="month", y="_present_flag_")
            fig.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)
            with st.expander("Explanation", expanded=show_explanations_toggle):
                st.write("Average attendance per month.")
        except Exception:
            st.info("Could not compute monthly attendance for the selected range.")

        # heatmap
        st.subheader("Attendance heatmap (complex)")
        try:
            heat = att_filtered.pivot_table(index="Name", columns=att_filtered["Date"].dt.date, values="_present_flag_", aggfunc="mean").fillna(0)
            max_show = min(200, heat.shape[0])
            n_show = st.slider("Number of students to show", min_value=10, max_value=max_show, value=min(50, max_show))
            top_students = heat.mean(axis=1).sort_values(ascending=False).head(n_show).index
            heat_small = heat.loc[top_students]
            fig = px.imshow(heat_small, color_continuous_scale="RdYlGn", aspect="auto")
            st.plotly_chart(fig, use_container_width=True)
            with st.expander("Explanation", expanded=show_explanations_toggle):
                st.write("Heatmap where rows are students and columns are dates. Green ~ present, Red ~ absent.")
        except Exception:
            st.info("Could not render heatmap. Try widening the date range or reducing the number of students.")

# -------------------------
# Tab 5: Insights & Exports
# -------------------------
with tabs[4]:
    st.header("Insights & exports")

    if student_mark_summary.empty:
        st.info("Not enough student-level marks data to compute insights.")
    else:
        # merge attendance if available
        student_level = student_mark_summary.copy()
        if not att_summary.empty and "ID" in att_summary.columns:
            student_level = student_level.merge(att_summary[["ID","attendance_rate"]], on="ID", how="left")
        student_level["flag_low_attendance"] = student_level["attendance_rate"].fillna(1) < (min_att_pct/100.0)
        student_level["flag_low_score"] = student_level["avg_score"] < min_score
        flagged = student_level[student_level["flag_low_attendance"] | student_level["flag_low_score"]]

        st.subheader("Flagged students (table)")
        st.dataframe(flagged[["ID","Roll","Name","avg_score","attendance_rate","flag_low_attendance","flag_low_score"]])
        with st.expander("Explanation", expanded=show_explanations_toggle):
            st.write("Students flagged for low attendance or low average score based on thresholds in the sidebar.")

        # exports
        st.markdown("---")
        st.markdown("**Download flagged list**")
        if not flagged.empty:
            csv_bytes = flagged.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", csv_bytes, file_name="flagged_students.csv", mime="text/csv")

            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                flagged.to_excel(writer, index=False, sheet_name="Flagged")
            st.download_button("Download Excel", buf.getvalue(), file_name="flagged_students.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        else:
            st.info("No flagged students under current thresholds.")

st.caption("Right iTech — interactive student insights. If anything still breaks, paste the traceback and I'll fix just that line.")
