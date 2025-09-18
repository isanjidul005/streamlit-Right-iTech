# Right_iTech_final_polished.py
# Copy-paste ready Streamlit app — polished UI, better colors, robust exports.

import os
import io
import tempfile
from datetime import date

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4

# -------------------------
# Page + theme config
# -------------------------
st.set_page_config(page_title="Right iTech", layout="wide", initial_sidebar_state="expanded")

# Gentle, distinctive palette (eye-friendly)
PALETTE = ["#0F4C81", "#1F77B4", "#2CA02C", "#FF7F0E", "#D62728", "#9467BD", "#8C564B", "#E377C2"]
ATT_PRESENT_COLOR = "#2CA02C"
ATT_ABSENT_COLOR = "#D62728"
MUTED = "#6C757D"
BG = "#F7F9FB"

# Minimal CSS to polish header & cards
st.markdown(
    f"""
    <style>
    .ri-header {{
        font-family: 'Segoe UI', Roboto, Arial, sans-serif;
        padding: 10px 6px;
        border-radius: 10px;
        background: linear-gradient(90deg, #ffffff, #f7fbff);
        box-shadow: 0 4px 20px rgba(15,76,129,0.06);
        margin-bottom: 10px;
    }}
    .ri-sub {{
        color: {MUTED};
        margin-bottom: 16px;
    }}
    .metric-card .stMetricValue {{
        font-weight: 700;
        color: #0F4C81;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div class='ri-header'><h1 style='margin:0'>Right iTech — Student Insights</h1></div>", unsafe_allow_html=True)
st.markdown("<div class='ri-sub'>Sleek dashboard for marks & attendance — interactive, exportable, and teacher-friendly.</div>", unsafe_allow_html=True)
st.write("---")

# -------------------------
# Sidebar: uploads + settings
# -------------------------
st.sidebar.header("Data upload & settings")

uploaded_att = st.sidebar.file_uploader("Upload Attendance CSV (optional)", type=["csv"])
uploaded_marks = st.sidebar.file_uploader("Upload Marks CSV (optional)", type=["csv"])

# fallback paths (used only if present)
FALLBACK_ATT = "/mnt/data/combined_attendance.csv"
FALLBACK_MARKS = "/mnt/data/cleanest_marks.csv"

@st.cache_data
def safe_read_csv(uploaded_file, fallback_path):
    try:
        if uploaded_file is not None:
            return pd.read_csv(uploaded_file)
        if fallback_path and os.path.exists(fallback_path):
            return pd.read_csv(fallback_path)
    except Exception:
        return pd.DataFrame()
    return pd.DataFrame()

att_df = safe_read_csv(uploaded_att, FALLBACK_ATT)
marks_df = safe_read_csv(uploaded_marks, FALLBACK_MARKS)

st.sidebar.markdown("---")
st.sidebar.header("UI / thresholds")
auto_expand_explanations = st.sidebar.checkbox("Auto-expand explanations", value=False)
flag_score_threshold = st.sidebar.number_input("Flag if avg score < (score)", min_value=0, max_value=100, value=40)
flag_att_threshold_pct = st.sidebar.slider("Flag if attendance < (%)", min_value=0, max_value=100, value=75)

# -------------------------
# Defensive cleaning
# -------------------------
# Ensure DataFrames exist
if att_df is None:
    att_df = pd.DataFrame()
if marks_df is None:
    marks_df = pd.DataFrame()

# Normalize column names
if not att_df.empty:
    att_df.columns = [c.strip() for c in att_df.columns]
if not marks_df.empty:
    marks_df.columns = [c.strip() for c in marks_df.columns]

# Safe parse Date
if not att_df.empty and "Date" in att_df.columns:
    att_df["Date"] = pd.to_datetime(att_df["Date"], dayfirst=True, errors="coerce")

# Create present flag robustly
if not att_df.empty:
    if "Status" in att_df.columns:
        att_df["_present_flag_"] = att_df["Status"].astype(str).str.upper().map({
            "P":1,"PRESENT":1,"1":1,"Y":1,"YES":1,
            "A":0,"ABSENT":0,"0":0,"N":0,"NO":0
        })
        att_df["_present_flag_"] = att_df["_present_flag_"].fillna(att_df["Status"].astype(str).str[0].map({"P":1,"A":0}))
    elif "Attendance" in att_df.columns:
        att_df["_present_flag_"] = att_df["Attendance"].astype(str).str.upper().map({"PRESENT":1,"ABSENT":0,"P":1,"A":0})
    else:
        att_df["_present_flag_"] = np.nan
else:
    att_df["_present_flag_"] = pd.Series(dtype=float)

# Marks numeric safety
if not marks_df.empty:
    if "Marks" in marks_df.columns:
        marks_df["Marks"] = pd.to_numeric(marks_df["Marks"], errors="coerce")
    else:
        marks_df["Marks"] = np.nan
    if "FullMarks" in marks_df.columns:
        marks_df["FullMarks"] = pd.to_numeric(marks_df["FullMarks"], errors="coerce")
else:
    marks_df = pd.DataFrame(columns=["ID","Roll","Name","Subject","ExamNumber","Exam","ExamType","Marks","FullMarks"])

# Subject color mapping
def assign_subject_colors(subjects):
    sub = sorted([s for s in subjects if pd.notna(s)])
    mapping = {}
    core = PALETTE
    for i,s in enumerate(sub):
        mapping[s] = core[i % len(core)]
    return mapping

SUBJECT_COLORS = assign_subject_colors(marks_df["Subject"].unique()) if "Subject" in marks_df.columns and not marks_df.empty else {}

# -------------------------
# Small helpers for exporting charts -> images
# -------------------------
def fig_to_png_bytes(fig, scale=2, width=None, height=None):
    """Try to get PNG bytes from a Plotly figure. Returns None if unavailable (no kaleido)."""
    try:
        if width and height:
            return fig.to_image(format="png", scale=scale, width=width, height=height)
        return fig.to_image(format="png", scale=scale)
    except Exception:
        try:
            # fallback through pio
            if width and height:
                return pio.to_image(fig, format="png", scale=scale, width=width, height=height)
            return pio.to_image(fig, format="png", scale=scale)
        except Exception:
            return None

def write_image_temp(fig, width=700, height=400):
    """Write fig to a temporary PNG file if possible, return filename or None."""
    img = fig_to_png_bytes(fig, width=width, height=height)
    if img:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        tmp.write(img); tmp.flush()
        tmp.close()
        return tmp.name
    return None

# -------------------------
# Export builders
# -------------------------
def build_class_excel_bytes(marks_df, att_df, student_level_df=None):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        if not marks_df.empty:
            marks_df.to_excel(writer, sheet_name="Marks", index=False)
        if not att_df.empty:
            att_df.to_excel(writer, sheet_name="Attendance", index=False)
        if student_level_df is not None and not student_level_df.empty:
            student_level_df.to_excel(writer, sheet_name="StudentSummary", index=False)
    buf.seek(0)
    return buf.getvalue()

def build_class_pdf_bytes(marks_df, att_df, student_level_df=None):
    out = io.BytesIO()
    doc = SimpleDocTemplate(out, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []
    story.append(Paragraph("Right iTech — Class Snapshot", styles["Title"]))
    story.append(Spacer(1,12))

    # summary text
    total_students = 0
    if "ID" in marks_df.columns and not marks_df.empty:
        total_students = marks_df["ID"].nunique()
    if "ID" in att_df.columns and not att_df.empty:
        total_students = max(total_students, att_df["ID"].nunique())
    story.append(Paragraph(f"Total students (estimated): {total_students}", styles["Normal"]))
    if "Gender" in att_df.columns and not att_df.empty:
        g = att_df.drop_duplicates(subset=["ID"])[["ID","Gender"]].groupby("Gender").size().to_dict()
        boys = g.get("M", g.get("Male", 0))
        girls = g.get("F", g.get("Female", 0))
        story.append(Paragraph(f"Boys: {boys} | Girls: {girls}", styles["Normal"]))
    # average attendance text
    if not att_df.empty and "_present_flag_" in att_df.columns:
        avg_att = att_df["_present_flag_"].mean()
        story.append(Paragraph(f"Average attendance: {avg_att*100:.1f}%", styles["Normal"]))
    story.append(Spacer(1,12))

    # try attach 3 key charts (pie, trend, top performers)
    temps = []
    try:
        # pie
        if not att_df.empty and "_present_flag_" in att_df.columns:
            att_summary = pd.DataFrame({"Status":["Present","Absent"], "Rate":[att_df["_present_flag_"].mean(), 1 - att_df["_present_flag_"].mean()]})
            fig_pie = px.pie(att_summary, names="Status", values="Rate", color="Status",
                             color_discrete_map={"Present":ATT_PRESENT_COLOR,"Absent":ATT_ABSENT_COLOR}, title="Attendance Ratio")
            tmp1 = write_image_temp(fig_pie)
            if tmp1:
                temps.append(tmp1)
                story.append(RLImage(tmp1, width=450, height=200))
                story.append(Spacer(1,12))
        # trend
        if not att_df.empty and "Date" in att_df.columns and not att_df["Date"].isna().all():
            att_trend = att_df.groupby(att_df["Date"].dt.date)["_present_flag_"].mean().reset_index()
            att_trend.columns = ["Date","attendance_rate"]
            fig_trend = px.line(att_trend, x="Date", y="attendance_rate", title="Daily Attendance %")
            fig_trend.update_yaxes(tickformat=".0%")
            tmp2 = write_image_temp(fig_trend)
            if tmp2:
                temps.append(tmp2)
                story.append(RLImage(tmp2, width=450, height=200))
                story.append(Spacer(1,12))
        # top performers
        if not marks_df.empty and "Marks" in marks_df.columns and "Name" in marks_df.columns:
            avg_marks = marks_df.groupby("Name")["Marks"].mean().reset_index().sort_values("Marks", ascending=False).head(5)
            fig_top = px.bar(avg_marks, x="Name", y="Marks", color="Marks", color_continuous_scale="greens", title="Top 5 by Avg Marks")
            tmp3 = write_image_temp(fig_top)
            if tmp3:
                temps.append(tmp3)
                story.append(RLImage(tmp3, width=450, height=200))
                story.append(Spacer(1,12))
    except Exception:
        # image embedding failed — continue with text only
        pass

    story.append(Paragraph("Notes & recommendations:", styles["Heading3"]))
    story.append(Paragraph("Use flagged list to identify students needing support. Prioritize low-attendance & low-score students.", styles["Normal"]))
    doc.build(story)

    # cleanup temps
    try:
        for t in temps:
            os.unlink(t)
    except Exception:
        pass

    out.seek(0)
    return out.getvalue()

def build_student_pdf_bytes_safe(student_name):
    # wrapper to call existing builder, uses per-student data
    return build_student_pdf_bytes_for_student(student_name)

def build_student_pdf_bytes_for_student(student_name):
    s_m = marks_df[marks_df["Name"]==student_name] if not marks_df.empty and "Name" in marks_df.columns else pd.DataFrame()
    s_a = att_df[att_df["Name"]==student_name] if not att_df.empty and "Name" in att_df.columns else pd.DataFrame()

    out = io.BytesIO()
    doc = SimpleDocTemplate(out, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []
    story.append(Paragraph(f"Student Report — {student_name}", styles["Title"]))
    story.append(Spacer(1,12))

    if not s_m.empty and "Marks" in s_m.columns:
        story.append(Paragraph(f"Average marks: {s_m['Marks'].mean():.2f}", styles["Normal"]))
    if not s_a.empty and "_present_flag_" in s_a.columns and not s_a["_present_flag_"].isna().all():
        p = int(s_a["_present_flag_"].sum()); t = len(s_a)
        story.append(Paragraph(f"Attendance: {p}/{t} ({(p/t*100):.1f}%)", styles["Normal"]))
    story.append(Spacer(1,12))

    temps = []
    try:
        if not s_m.empty and "Subject" in s_m.columns:
            subj = s_m.groupby("Subject")["Marks"].mean().reset_index()
            fig1 = px.bar(subj, x="Subject", y="Marks", color="Subject", color_discrete_map=SUBJECT_COLORS)
            tmp1 = write_image_temp(fig1)
            if tmp1:
                temps.append(tmp1); story.append(RLImage(tmp1, width=450, height=200)); story.append(Spacer(1,12))
        if not s_a.empty and "Date" in s_a.columns and not s_a["Date"].isna().all():
            s_a2 = s_a.copy(); s_a2["month"] = s_a2["Date"].dt.to_period("M").astype(str)
            monthly = s_a2.groupby("month")["_present_flag_"].mean().reset_index()
            fig2 = px.bar(monthly, x="month", y="_present_flag_")
            tmp2 = write_image_temp(fig2)
            if tmp2:
                temps.append(tmp2); story.append(RLImage(tmp2, width=450, height=150)); story.append(Spacer(1,12))
    except Exception:
        pass

    story.append(Paragraph("Notes & next steps:", styles["Heading3"]))
    story.append(Paragraph("Consider targeted support in weak subjects and monitor attendance closely.", styles["Normal"]))
    doc.build(story)

    try:
        for t in temps:
            os.unlink(t)
    except Exception:
        pass

    out.seek(0)
    return out.getvalue()

# -------------------------
# Show warning if no data loaded at all
# -------------------------
if att_df.empty and marks_df.empty:
    st.error("No attendance or marks data loaded. Use the sidebar to upload CSVs or place fallback files on the server.")
    st.stop()

# -------------------------
# Global filters in sidebar (improve UX)
# -------------------------
st.sidebar.header("Global filters")
# date range
if not att_df.empty and "Date" in att_df.columns and not att_df["Date"].isna().all():
    earliest = att_df["Date"].min().date()
    latest = att_df["Date"].max().date()
else:
    earliest = date.today()
    latest = date.today()
date_range = st.sidebar.date_input("Attendance date range", value=(earliest, latest))

# subject & exam filters
subject_options = sorted(marks_df["Subject"].dropna().unique().tolist()) if not marks_df.empty and "Subject" in marks_df.columns else []
subject_filter = st.sidebar.multiselect("Filter subjects (optional)", options=subject_options, default=subject_options)
exam_options = sorted(marks_df["ExamNumber"].dropna().unique().tolist()) if not marks_df.empty and "ExamNumber" in marks_df.columns else []
exam_filter = st.sidebar.multiselect("Filter exams (optional)", options=exam_options, default=exam_options)

# quick name search
name_search = st.sidebar.text_input("Search student name (partial)")

# -------------------------
# Tabs UI
# -------------------------
tabs = st.tabs(["Class overview","Individual Student Report","Compare Students","Attendance explorer","Insights & Export"])

# ===== TAB 0: Class overview =====
with tabs[0]:
    st.header("Class overview")

    # Snapshot metrics
    c1, c2, c3, c4 = st.columns(4)
    ids_all = set()
    if "ID" in marks_df.columns and not marks_df.empty:
        ids_all.update(marks_df["ID"].dropna().astype(str).tolist())
    if "ID" in att_df.columns and not att_df.empty:
        ids_all.update(att_df["ID"].dropna().astype(str).tolist())
    total_students = len(ids_all) if ids_all else (marks_df["Name"].nunique() if not marks_df.empty and "Name" in marks_df.columns else 0)
    c1.metric("Total students", total_students)

    # gender
    if "Gender" in att_df.columns and not att_df.empty:
        genders = att_df.drop_duplicates(subset=["ID"])[["ID","Gender"]].groupby("Gender").size().to_dict()
        boys = genders.get("M", genders.get("Male", 0))
        girls = genders.get("F", genders.get("Female", 0))
        c2.metric("Boys", int(boys)); c3.metric("Girls", int(girls))
    else:
        c2.metric("Boys", "N/A"); c3.metric("Girls", "N/A")

    avg_att = att_df["_present_flag_"].mean() if not att_df.empty else np.nan
    c4.metric("Avg attendance", f"{avg_att*100:.1f}%" if not np.isnan(avg_att) else "N/A")

    st.markdown("---")

    # Attendance categories pie (student-level)
    st.subheader("Attendance categories (student-level)")
    if not att_df.empty and "Name" in att_df.columns and "_present_flag_" in att_df.columns:
        try:
            sd, ed = (date_range[0], date_range[1]) if isinstance(date_range, (list,tuple)) and len(date_range)==2 else (date_range, date_range)
        except Exception:
            sd, ed = earliest, latest
        if "Date" in att_df.columns:
            mask = (att_df["Date"].dt.date >= sd) & (att_df["Date"].dt.date <= ed)
            att_in_range = att_df[mask]
        else:
            att_in_range = att_df.copy()
        student_att = att_in_range.groupby("Name")["_present_flag_"].mean().reset_index()
        student_att["attendance_pct"] = student_att["_present_flag_"] * 100
        def cat(rate):
            if rate >= 90: return "Excellent (90%+)"
            if rate >= 75: return "Good (75–90%)"
            if rate >= 50: return "Average (50–75%)"
            return "Poor (<50%)"
        student_att["Category"] = student_att["attendance_pct"].apply(cat)
        counts = student_att["Category"].value_counts().reindex(["Excellent (90%+)","Good (75–90%)","Average (50–75%)","Poor (<50%)"]).fillna(0).reset_index()
        counts.columns = ["Category","Count"]
        fig_cat = px.pie(counts, names="Category", values="Count",
                         color="Category",
                         color_discrete_map={
                             "Excellent (90%+)":"#2CA02C",
                             "Good (75–90%)":"#1F77B4",
                             "Average (50–75%)":"#FF7F0E",
                             "Poor (<50%)":"#D62728"
                         }, title="Students by Attendance Category")
        st.plotly_chart(fig_cat, use_container_width=True)
        with st.expander("ℹ️ Explanation", expanded=auto_expand_explanations):
            st.write(
                "Students are bucketed by their personal attendance percentage across the selected date range. "
                "This view shows what portion of the class is in each attendance-quality bracket — helpful to prioritize follow-ups."
            )
    else:
        st.info("Not enough attendance data to compute student-level categories.")

    st.markdown("---")

    # Attendance vs marks scatter (unique insight)
    st.subheader("Attendance vs Average Marks (per student)")
    if not marks_df.empty and not att_df.empty and "Name" in marks_df.columns and "Name" in att_df.columns:
        # compute per-student averages
        marks_avg = marks_df.groupby("Name")["Marks"].mean().reset_index().rename(columns={"Marks":"avg_marks"})
        att_avg = att_df.groupby("Name")["_present_flag_"].mean().reset_index().rename(columns={"_present_flag_":"att_rate"})
        merge = pd.merge(marks_avg, att_avg, on="Name", how="inner")
        if name_search:
            merge = merge[merge["Name"].str.contains(name_search, case=False, na=False)]
        if not merge.empty:
            fig_sc = px.scatter(merge, x="att_rate", y="avg_marks", hover_data=["Name"], title="Attendance % vs Avg Marks",
                                labels={"att_rate":"Attendance %","avg_marks":"Avg Marks"}, color_discrete_sequence=[PALETTE[0]])
            fig_sc.update_xaxes(tickformat=".0%")
            st.plotly_chart(fig_sc, use_container_width=True)
            with st.expander("ℹ️ Explanation", expanded=auto_expand_explanations):
                st.write(
                    "Scatter plots each student by attendance (x) and average marks (y). "
                    "Students in bottom-left are both absent and low-performing — priority for intervention. "
                    "Hover to see names; use filters to focus on subsets."
                )
        else:
            st.info("No students match the current filters.")
    else:
        st.info("Need both attendance and marks data to show correlation.")

    st.markdown("---")

    # Cumulative marks curve (simple & different)
    st.subheader("Cumulative distribution of marks (class)")
    if not marks_df.empty and "Marks" in marks_df.columns:
        m = marks_df["Marks"].dropna().sort_values()
        if not m.empty:
            cum = m.reset_index(drop=True).reset_index().assign(cumshare=lambda df: (df.index+1)/len(df))
            fig_cum = px.line(cum, x="index", y="Marks", title="Cumulative Marks (index order) - use as quick inequality check")
            st.plotly_chart(fig_cum, use_container_width=True)
            with st.expander("ℹ️ Explanation", expanded=auto_expand_explanations):
                st.write(
                    "This cumulative-style curve helps visualize how marks accumulate across records. "
                    "A steep left part shows many low scores; a gradual incline shows more evenly distributed scores."
                )
        else:
            st.info("No marks to plot.")
    else:
        st.info("No marks data available.")

# ===== TAB 1: Individual Student Report =====
with tabs[1]:
    st.header("Individual Student Report")

    # candidate list from marks then attendance
    name_candidates = []
    if not marks_df.empty and "Name" in marks_df.columns:
        name_candidates = sorted(marks_df["Name"].dropna().unique().tolist())
    elif not att_df.empty and "Name" in att_df.columns:
        name_candidates = sorted(att_df["Name"].dropna().unique().tolist())

    if not name_candidates:
        st.info("No student names found in the data.")
    else:
        student = st.selectbox("Select student", name_candidates)
        s_marks = marks_df[marks_df["Name"] == student] if not marks_df.empty else pd.DataFrame()
        s_att = att_df[att_df["Name"] == student] if not att_df.empty else pd.DataFrame()

        # profile card
        if not s_marks.empty:
            sid = s_marks["ID"].iloc[0] if "ID" in s_marks.columns else "N/A"
            sroll = s_marks["Roll"].iloc[0] if "Roll" in s_marks.columns else "N/A"
        elif not s_att.empty:
            sid = s_att["ID"].iloc[0] if "ID" in s_att.columns else "N/A"
            sroll = s_att["Roll"].iloc[0] if "Roll" in s_att.columns else "N/A"
        else:
            sid = "N/A"; sroll = "N/A"

        colA, colB = st.columns([2,3])
        with colA:
            st.markdown(f"### {student}")
            st.markdown(f"**ID:** {sid}  \n**Roll:** {sroll}")
        with colB:
            avg_mark = s_marks["Marks"].mean() if not s_marks.empty and s_marks["Marks"].notna().any() else np.nan
            att_rate = s_att["_present_flag_"].mean() if not s_att.empty and "_present_flag_" in s_att.columns else np.nan
            st.metric("Average mark", f"{avg_mark:.1f}" if not np.isnan(avg_mark) else "N/A")
            st.metric("Attendance rate", f"{att_rate*100:.1f}%" if not np.isnan(att_rate) else "N/A")

        st.markdown("---")

        # Subject performance simple bar
        st.subheader("Subject performance (average)")
        if not s_marks.empty and "Subject" in s_marks.columns:
            subj_avg = s_marks.groupby("Subject")["Marks"].mean().reset_index()
            fig_subj = px.bar(subj_avg, x="Subject", y="Marks", color="Subject", color_discrete_map=SUBJECT_COLORS)
            st.plotly_chart(fig_subj, use_container_width=True)
            with st.expander("ℹ️ Explanation", expanded=auto_expand_explanations):
                st.write("Shows student's average per subject — clear strengths and weaknesses.")
        else:
            st.info("No subject marks for this student.")

        # Marks across exams grouped
        st.subheader("Marks across exams (grouped by subject) — easier to read")
        if not s_marks.empty and "ExamNumber" in s_marks.columns and "Subject" in s_marks.columns:
            exam_plot = s_marks.groupby(["ExamNumber","Subject"])["Marks"].mean().reset_index()
            fig_exam = px.bar(exam_plot, x="ExamNumber", y="Marks", color="Subject", barmode="group", color_discrete_map=SUBJECT_COLORS)
            fig_exam.update_layout(xaxis_title="Exam number", yaxis_title="Marks")
            st.plotly_chart(fig_exam, use_container_width=True)
            with st.expander("ℹ️ Explanation", expanded=auto_expand_explanations):
                st.write("Grouped bars show per-subject marks per exam — easier to compare exam-by-exam.")
        else:
            st.info("Not enough exam/subject structure to plot grouped bars.")

        # Attendance by month (clean)
        st.subheader("Attendance by month")
        if not s_att.empty and "Date" in s_att.columns:
            s_att2 = s_att.sort_values("Date").copy()
            s_att2["month"] = s_att2["Date"].dt.to_period("M").astype(str)
            monthly = s_att2.groupby("month")["_present_flag_"].mean().reset_index()
            fig_attm = px.bar(monthly, x="month", y="_present_flag_")
            fig_attm.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig_attm, use_container_width=True)
            with st.expander("ℹ️ Explanation", expanded=auto_expand_explanations):
                st.write("Monthly attendance percentage for the student — smoother and easier to read than daily dots.")
        else:
            st.info("No attendance records for this student.")

        st.markdown("---")
        # Export student report
        c1, c2 = st.columns(2)
        with c1:
            excel_bytes = build_class_excel_bytes(s_marks, s_att, None)
            st.download_button("Download student Excel", data=excel_bytes, file_name=f"{student}_report.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        with c2:
            pdf_bytes = build_student_pdf_bytes_for_student(student)
            st.download_button("Download student PDF", data=pdf_bytes, file_name=f"{student}_report.pdf", mime="application/pdf")

# ===== TAB 2: Compare Students =====
with tabs[2]:
    st.header("Compare Students")
    candidate_names = sorted(set(marks_df["Name"].dropna().tolist())) if "Name" in marks_df.columns and not marks_df.empty else []
    selection = st.multiselect("Select up to 6 students", options=candidate_names, max_selections=6)
    exam_choice = st.selectbox("Exam filter (All for averages)", options=["All"] + ([str(e) for e in sorted(marks_df["ExamNumber"].dropna().unique().tolist())] if "ExamNumber" in marks_df.columns else []))

    if not selection or len(selection) < 2:
        st.info("Select two or more students to compare.")
    else:
        comp = marks_df[marks_df["Name"].isin(selection)]
        if exam_choice != "All":
            comp = comp[comp["ExamNumber"].astype(str) == exam_choice]

        st.subheader("Subject averages comparison")
        comp_avg = comp.groupby(["Name","Subject"])["Marks"].mean().reset_index()
        if not comp_avg.empty:
            subjects = sorted(comp_avg["Subject"].unique())
            fig = go.Figure()
            for name in selection:
                row = comp_avg[comp_avg["Name"]==name]
                y_vals = [row[row["Subject"]==s]["Marks"].values[0] if s in row["Subject"].values else 0 for s in subjects]
                fig.add_trace(go.Bar(name=name, x=subjects, y=y_vals))
            fig.update_layout(barmode="group", xaxis_title="Subject", yaxis_title="Avg marks", legend_title="Student")
            st.plotly_chart(fig, use_container_width=True)
            with st.expander("ℹ️ Explanation", expanded=auto_expand_explanations):
                st.write("Grouped bars compare subject averages between selected students. Use the exam filter to focus on a single exam.")
        else:
            st.info("No data for selected students/exam.")

        st.subheader("Exam trend")
        if "ExamNumber" in marks_df.columns:
            trend_df = marks_df[marks_df["Name"].isin(selection)].groupby(["ExamNumber","Name"])["Marks"].mean().reset_index()
            if not trend_df.empty:
                fig_trend = px.line(trend_df, x="ExamNumber", y="Marks", color="Name", markers=True)
                fig_trend.update_layout(yaxis_title="Avg marks")
                st.plotly_chart(fig_trend, use_container_width=True)
                with st.expander("ℹ️ Explanation", expanded=auto_expand_explanations):
                    st.write("Line chart shows how selected students' averages changed across exams.")
            else:
                st.info("Not enough exam data to draw trends.")
        else:
            st.info("No ExamNumber column in marks.")

# ===== TAB 3: Attendance explorer =====
with tabs[3]:
    st.header("Attendance explorer")

    if att_df.empty:
        st.info("No attendance data.")
    else:
        min_d = att_df["Date"].min().date() if "Date" in att_df.columns and not att_df["Date"].isna().all() else date.today()
        max_d = att_df["Date"].max().date() if "Date" in att_df.columns and not att_df["Date"].isna().all() else date.today()
        dr = st.date_input("Select date range", value=(min_d, max_d))
        try:
            sd, ed = (dr[0], dr[1]) if isinstance(dr, (list,tuple)) and len(dr)==2 else (dr, dr)
        except Exception:
            sd, ed = min_d, max_d

        mask = (att_df["Date"].dt.date >= sd) & (att_df["Date"].dt.date <= ed) if "Date" in att_df.columns else slice(None)
        att_filtered = att_df[mask] if mask is not slice(None) else att_df.copy()

        st.subheader("Daily class attendance %")
        if not att_filtered.empty:
            att_trend = att_filtered.groupby(att_filtered["Date"].dt.date)["_present_flag_"].mean().reset_index()
            att_trend.columns = ["Date","attendance_rate"]
            fig = px.line(att_trend, x="Date", y="attendance_rate", markers=True)
            fig.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)
            with st.expander("ℹ️ Explanation", expanded=auto_expand_explanations):
                st.write("Daily class attendance percentage in the selected date range. Hover to see exact values.")
        else:
            st.info("No attendance records in selected range.")

        st.markdown("---")
        st.subheader("Average attendance by weekday")
        try:
            att_filtered["weekday"] = att_filtered["Date"].dt.day_name()
            weekday_avg = att_filtered.groupby("weekday")["_present_flag_"].mean().reindex(
                ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
            ).reset_index().dropna()
            weekday_avg.columns = ["weekday","attendance_rate"]
            fig_wk = px.bar(weekday_avg, x="weekday", y="attendance_rate")
            fig_wk.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig_wk, use_container_width=True)
            with st.expander("ℹ️ Explanation", expanded=auto_expand_explanations):
                st.write("Average attendance per weekday — it helps find recurring low-attendance days.")
        except Exception:
            st.info("Unable to compute weekday breakdown for this dataset.")

        st.markdown("---")
        st.subheader("Attendance leaderboard (top N)")
        n_top = st.number_input("How many students to show:", min_value=5, max_value=200, value=20)
        att_leader = att_filtered.groupby("Name")["_present_flag_"].mean().reset_index().sort_values("_present_flag_", ascending=False)
        if not att_leader.empty:
            leader_display = att_leader.copy()
            leader_display["attendance_rate"] = leader_display["_present_flag_"].apply(lambda x: f"{x:.1%}")
            st.dataframe(leader_display.set_index("Name").head(n_top))
            with st.expander("ℹ️ Explanation", expanded=auto_expand_explanations):
                st.write("Top students by attendance percentage in the selected range.")
        else:
            st.info("No attendance summary available for the selected range.")

# ===== TAB 4: Insights & Export =====
with tabs[4]:
    st.header("Insights & Export")

    # student summary
    def student_summary_df(mdf):
        if mdf.empty:
            return pd.DataFrame()
        s = mdf.groupby(["ID","Roll","Name"]).agg(avg_score=("Marks","mean"), exams_taken=("ExamNumber","nunique"), records_count=("Marks","count")).reset_index()
        return s

    student_level = student_summary_df(marks_df)

    st.subheader("Top / Bottom performers")
    k = st.number_input("How many top/bottom to show", min_value=1, max_value=50, value=5)
    if student_level.empty:
        st.info("Not enough marks data to compute performers.")
    else:
        # merge attendance %
        if not att_df.empty and "ID" in att_df.columns:
            att_s = att_df.groupby("ID")["_present_flag_"].mean().reset_index().rename(columns={"_present_flag_":"attendance_rate"})
            student_level["ID"] = student_level["ID"].astype(str)
            att_s["ID"] = att_s["ID"].astype(str)
            student_level = student_level.merge(att_s, on="ID", how="left")
        else:
            student_level["attendance_rate"] = np.nan

        topK = student_level.sort_values("avg_score", ascending=False).head(k)[["Name","avg_score","attendance_rate"]]
        botK = student_level.sort_values("avg_score", ascending=True).head(k)[["Name","avg_score","attendance_rate"]]
        top_disp = topK.copy(); top_disp["avg_score"]=top_disp["avg_score"].round(2); top_disp["attendance_rate"]=top_disp["attendance_rate"].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")
        bot_disp = botK.copy(); bot_disp["avg_score"]=bot_disp["avg_score"].round(2); bot_disp["attendance_rate"]=bot_disp["attendance_rate"].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Top performers**")
            st.table(top_disp.set_index("Name") if not top_disp.empty else pd.DataFrame())
        with c2:
            st.markdown("**Bottom performers**")
            st.table(bot_disp.set_index("Name") if not bot_disp.empty else pd.DataFrame())
        with st.expander("Explanation", expanded=auto_expand_explanations):
            st.write("Top and bottom performers are computed by average marks across recorded exams. Combine with attendance to plan recognition or interventions.")

    st.markdown("---")
    st.subheader("Flagged students (low attendance or low marks)")
    if student_level.empty:
        st.info("No student-level data to flag.")
    else:
        flagged = student_level[(student_level["avg_score"] < flag_score_threshold) | (student_level["attendance_rate"].fillna(1) < (flag_att_threshold_pct/100.0))]
        if not flagged.empty:
            flagged_disp = flagged[["ID","Roll","Name","avg_score","attendance_rate"]].copy()
            flagged_disp["avg_score"] = flagged_disp["avg_score"].round(2)
            flagged_disp["attendance_rate"] = flagged_disp["attendance_rate"].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")
            st.dataframe(flagged_disp.set_index("Name"))
            csv_bytes = flagged.to_csv(index=False).encode("utf-8")
            st.download_button("Download flagged (CSV)", data=csv_bytes, file_name="flagged_students.csv", mime="text/csv")
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                flagged.to_excel(writer, index=False, sheet_name="Flagged")
            st.download_button("Download flagged (Excel)", data=buf.getvalue(), file_name="flagged_students.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        else:
            st.success("No flagged students under current thresholds.")

    st.markdown("---")
    st.subheader("Export full datasets")
    if not marks_df.empty:
        st.download_button("Download marks (CSV)", data=marks_df.to_csv(index=False).encode("utf-8"), file_name="marks_export.csv", mime="text/csv")
    if not att_df.empty:
        st.download_button("Download attendance (CSV)", data=att_df.to_csv(index=False).encode("utf-8"), file_name="attendance_export.csv", mime="text/csv")

    st.markdown("---")
    st.subheader("Generate class PDF snapshot (charts embedded when possible)")
    if st.button("Generate PDF snapshot"):
        pdf_bytes = build_class_pdf_bytes(marks_df, att_df, student_level)
        st.download_button("Download class snapshot PDF", data=pdf_bytes, file_name="class_snapshot.pdf", mime="application/pdf")
    with st.expander("ℹ️ Explanation", expanded=auto_expand_explanations):
        st.write("Exports include CSV/Excel and a PDF snapshot of the class with key charts. If the environment lacks Plotly image support, the PDF will still contain textual summary and downloadable data.")

st.caption("Right iTech — polished. If you want custom branding (logo, colors) or a dark theme, tell me which colors/logo and I will add them without touching other functionality.")
