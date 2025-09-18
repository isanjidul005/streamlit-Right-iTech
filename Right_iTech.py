# right_itech_full.py
# Right iTech - Full-featured Student Insights Dashboard (copy-paste ready)

import os
import io
import tempfile
from datetime import date

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# PDF / Excel
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4

# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="Right iTech", layout="wide", initial_sidebar_state="expanded")

# -------------------------
# Utility: robust CSV load
# -------------------------
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

# -------------------------
# Sidebar: upload + basic config
# -------------------------
st.sidebar.header("Data upload & settings")
uploaded_marks = st.sidebar.file_uploader("Upload marks CSV (cleanest_marks.csv)", type=["csv"])
uploaded_att = st.sidebar.file_uploader("Upload attendance CSV (combined_attendance.csv)", type=["csv"])

FALLBACK_MARKS = "/mnt/data/cleanest_marks.csv"
FALLBACK_ATT = "/mnt/data/combined_attendance.csv"

marks_df = safe_read_csv(uploaded_marks, FALLBACK_MARKS)
att_df = safe_read_csv(uploaded_att, FALLBACK_ATT)

st.sidebar.markdown("---")
st.sidebar.header("UI & thresholds")
auto_expand_explanations = st.sidebar.checkbox("Auto-expand explanations", value=False)
flag_score_threshold = st.sidebar.number_input("Flag students if avg score < (score)", min_value=0, max_value=100, value=40)
flag_att_threshold_pct = st.sidebar.slider("Flag students if attendance < (%)", min_value=0, max_value=100, value=75)

# color palette
PALETTE = px.colors.qualitative.Set2
ATT_PRESENT_COLOR = "#2ca02c"
ATT_ABSENT_COLOR = "#d62728"

# -------------------------
# Basic validation & cleaning
# -------------------------
# if both empty -> show message but don't crash
if marks_df.empty and att_df.empty:
    st.sidebar.warning("No data loaded. Upload at least one CSV to use the dashboard.")
# Normalize column names
if not marks_df.empty:
    marks_df.columns = [c.strip() for c in marks_df.columns]
if not att_df.empty:
    att_df.columns = [c.strip() for c in att_df.columns]

# Ensure numeric
if not marks_df.empty:
    if "Marks" in marks_df.columns:
        marks_df["Marks"] = pd.to_numeric(marks_df["Marks"], errors="coerce")
    else:
        marks_df["Marks"] = np.nan
    if "FullMarks" in marks_df.columns:
        marks_df["FullMarks"] = pd.to_numeric(marks_df["FullMarks"], errors="coerce")
    else:
        marks_df["FullMarks"] = np.nan
    if "WasAbsent" not in marks_df.columns:
        marks_df["WasAbsent"] = "False"
else:
    marks_df = pd.DataFrame(columns=["ID","Roll","Name","Subject","ExamNumber","ExamType","Marks","FullMarks","WasAbsent"])

# Attendance cleaning
if not att_df.empty:
    if "Date" in att_df.columns:
        att_df["Date"] = pd.to_datetime(att_df["Date"], dayfirst=True, errors="coerce")
    # create present flag (works for many tokens)
    if "Status" in att_df.columns:
        att_df["_present_flag_"] = att_df["Status"].astype(str).str.upper().map({
            "P":1,"PRESENT":1,"1":1,"Y":1,"YES":1,
            "A":0,"ABSENT":0,"0":0,"N":0,"NO":0
        })
        att_df["_present_flag_"] = att_df["_present_flag_"].fillna(att_df["Status"].astype(str).str[0].map({"P":1,"A":0}))
    else:
        # try 'Attendance' column if present
        if "Attendance" in att_df.columns:
            att_df["_present_flag_"] = att_df["Attendance"].astype(str).str.upper().map({"PRESENT":1,"ABSENT":0,"P":1,"A":0})
        else:
            att_df["_present_flag_"] = np.nan
else:
    att_df = pd.DataFrame(columns=["ID","Roll","Name","Date","Status","_present_flag_"])

# Subject color mapping
def assign_subject_colors(subjects):
    sub = sorted([s for s in subjects if pd.notna(s)])
    mapping = {}
    for i, s in enumerate(sub):
        mapping[s] = PALETTE[i % len(PALETTE)]
    return mapping

if "Subject" in marks_df.columns and not marks_df.empty:
    SUBJECT_COLORS = assign_subject_colors(marks_df["Subject"].unique())
else:
    SUBJECT_COLORS = {}

# -------------------------
# Helper functions
# -------------------------
def student_marks_summary(df):
    if df.empty:
        return pd.DataFrame()
    s = df.groupby(["ID","Roll","Name"]).agg(
        avg_score=("Marks","mean"),
        exams_taken=("ExamNumber","nunique"),
        records_count=("Marks","count"),
    ).reset_index()
    return s

student_summary = student_marks_summary(marks_df)

def fig_to_png_bytes(fig, scale=2):
    try:
        return fig.to_image(format="png", scale=scale)
    except Exception:
        return None

def build_student_excel_bytes(student_name):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        s_m = marks_df[marks_df["Name"]==student_name] if "Name" in marks_df.columns and not marks_df.empty else pd.DataFrame()
        s_a = att_df[att_df["Name"]==student_name] if "Name" in att_df.columns and not att_df.empty else pd.DataFrame()
        s_m.to_excel(writer, sheet_name="Marks", index=False)
        s_a.to_excel(writer, sheet_name="Attendance", index=False)
    buf.seek(0)
    return buf.getvalue()

def build_student_pdf_bytes(student_name):
    s_m = marks_df[marks_df["Name"]==student_name] if "Name" in marks_df.columns and not marks_df.empty else pd.DataFrame()
    s_a = att_df[att_df["Name"]==student_name] if "Name" in att_df.columns and not att_df.empty else pd.DataFrame()
    out_buf = io.BytesIO()
    doc = SimpleDocTemplate(out_buf, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []
    story.append(Paragraph(f"Student Report — {student_name}", styles["Title"]))
    story.append(Spacer(1,12))
    if not s_m.empty:
        story.append(Paragraph(f"Average marks: {s_m['Marks'].mean():.2f}", styles["Normal"]))
    if not s_a.empty and "_present_flag_" in s_a.columns and len(s_a)>0:
        p = int(s_a["_present_flag_"].sum()); t = len(s_a)
        story.append(Paragraph(f"Attendance: {p}/{t} ({(p/t*100):.1f}%)", styles["Normal"]))
    story.append(Spacer(1,12))

    temps = []
    try:
        if not s_m.empty and "Subject" in s_m.columns:
            subj_avg = s_m.groupby("Subject")["Marks"].mean().reset_index()
            fig1 = px.bar(subj_avg, x="Subject", y="Marks", color="Subject", color_discrete_map=SUBJECT_COLORS)
            img1 = fig_to_png_bytes(fig1)
            if img1:
                tmp1 = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                tmp1.write(img1); tmp1.flush()
                temps.append(tmp1.name)
                story.append(RLImage(tmp1.name, width=450, height=200))
                story.append(Spacer(1,12))
        if not s_a.empty and "Date" in s_a.columns:
            s_a_month = s_a.copy()
            s_a_month["month"] = s_a_month["Date"].dt.to_period("M").astype(str)
            monthly = s_a_month.groupby("month")["_present_flag_"].mean().reset_index()
            fig2 = px.bar(monthly, x="month", y="_present_flag_")
            img2 = fig_to_png_bytes(fig2)
            if img2:
                tmp2 = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                tmp2.write(img2); tmp2.flush()
                temps.append(tmp2.name)
                story.append(RLImage(tmp2.name, width=450, height=150))
                story.append(Spacer(1,12))
    except Exception:
        # image export may be unavailable
        pass

    story.append(Paragraph("Notes & recommendations:", styles["Heading3"]))
    story.append(Paragraph("Review subjects with low averages and monitor attendance for flagged students.", styles["Normal"]))
    doc.build(story)

    # cleanup
    try:
        for p in temps:
            os.unlink(p)
    except Exception:
        pass

    out_buf.seek(0)
    return out_buf.getvalue()

# -------------------------
# Top-level header
# -------------------------
st.markdown("<h1 style='margin-bottom:2px'>Right iTech — Student Insights</h1>", unsafe_allow_html=True)
st.markdown("<div style='color:#666;margin-bottom:10px'>Interactive dashboard: marks, attendance, comparisons, and exports.</div>", unsafe_allow_html=True)
st.write("---")

# -------------------------
# Controls: global filters
# -------------------------
st.sidebar.header("Global filters")
# Date range filter for attendance-aware charts
if not att_df.empty and "Date" in att_df.columns:
    min_date = att_df["Date"].min().date()
    max_date = att_df["Date"].max().date()
else:
    min_date = date.today()
    max_date = date.today()
date_range = st.sidebar.date_input("Attendance date range", value=(min_date, max_date))

# Subject filter
subject_options = sorted(marks_df["Subject"].dropna().unique().tolist()) if "Subject" in marks_df.columns and not marks_df.empty else []
subject_filter = st.sidebar.multiselect("Filter subjects (optional)", options=subject_options, default=subject_options)

# Exam filter
exam_options = sorted(marks_df["ExamNumber"].dropna().unique().tolist()) if "ExamNumber" in marks_df.columns and not marks_df.empty else []
exam_filter = st.sidebar.multiselect("Filter exams (optional)", options=exam_options, default=exam_options)

# Name search quick
name_search = st.sidebar.text_input("Search student name (partial)")

# -------------------------
# Layout: Tabs
# -------------------------
tabs = st.tabs(["Class overview","Individual Student Report","Compare Students","Attendance explorer","Insights & Export"])

# ===== TAB 0: Class overview =====
with tabs[0]:
    st.header("Class overview")
    # Snapshot metrics
    col_a, col_b, col_c, col_d = st.columns(4)
    # total students from either marks or attendance
    ids = set()
    if "ID" in marks_df.columns:
        ids.update(marks_df["ID"].dropna().astype(str).tolist())
    if "ID" in att_df.columns:
        ids.update(att_df["ID"].dropna().astype(str).tolist())
    total_students = len(ids) if ids else (marks_df["Name"].nunique() if not marks_df.empty else 0)
    col_a.metric("Total students", total_students)

    # gender breakdown if exists
    if "Gender" in att_df.columns and not att_df.empty:
        g = att_df.drop_duplicates(subset=["ID"])[["ID","Gender"]].groupby("Gender").size().to_dict()
        boys = g.get("M", g.get("Male", 0)) + 0
        girls = g.get("F", g.get("Female", 0)) + 0
        col_b.metric("Boys", int(boys))
        col_c.metric("Girls", int(girls))
    else:
        col_b.metric("Boys", "N/A")
        col_c.metric("Girls", "N/A")

    # average attendance %
    avg_att = att_df["_present_flag_"].mean() if not att_df.empty else np.nan
    col_d.metric("Avg attendance", f"{avg_att*100:.1f}%" if not np.isnan(avg_att) else "N/A")

    st.markdown("---")

    # Subject ranking by class average
    st.subheader("Subject ranking (class average)")
    if not subj := (marks_df.groupby("Subject")["Marks"].mean().reset_index() if not marks_df.empty and "Subject" in marks_df.columns else pd.DataFrame()).empty:
        subj_summary = marks_df.groupby("Subject")["Marks"].mean().reset_index().rename(columns={"Marks":"avg_score"})
        subj_sorted = subj_summary.sort_values("avg_score", ascending=False)
        fig_sub = px.bar(subj_sorted, x="avg_score", y="Subject", orientation="h", color="Subject", color_discrete_map=SUBJECT_COLORS)
        fig_sub.update_layout(xaxis_title="Average score", yaxis_title="")
        st.plotly_chart(fig_sub, use_container_width=True)
        with st.expander("Explanation", expanded=auto_expand_explanations):
            best = subj_sorted.iloc[0]["Subject"]
            worst = subj_sorted.iloc[-1]["Subject"]
            st.write(f"Subjects are ranked by class average. Best: **{best}** — Lowest: **{worst}**. Prioritize teaching focus accordingly.")
    else:
        st.info("Not enough subject data to compute ranking.")

    st.markdown("---")

    # Attendance categories pie (useful)
    st.subheader("Attendance categories (students grouped by their attendance%)")
    if not att_df.empty and "Name" in att_df.columns and "_present_flag_" in att_df.columns:
        # filter by date range
        try:
            sd, ed = (date_range[0], date_range[1]) if isinstance(date_range, (list,tuple)) and len(date_range)==2 else (date_range, date_range)
        except Exception:
            sd, ed = min_date, max_date
        mask = (att_df["Date"].dt.date >= sd) & (att_df["Date"].dt.date <= ed)
        att_in_range = att_df[mask]
        student_att = att_in_range.groupby("Name")["_present_flag_"].mean().reset_index()
        student_att["attendance_pct"] = student_att["_present_flag_"] * 100

        def cat(rate):
            if rate >= 90:
                return "Excellent (90%+)"
            elif rate >= 75:
                return "Good (75–90%)"
            elif rate >= 50:
                return "Average (50–75%)"
            else:
                return "Poor (<50%)"

        student_att["Category"] = student_att["attendance_pct"].apply(cat)
        counts = student_att["Category"].value_counts().reindex(["Excellent (90%+)","Good (75–90%)","Average (50–75%)","Poor (<50%)"]).fillna(0).reset_index()
        counts.columns = ["Category","Count"]
        fig_cat = px.pie(counts, names="Category", values="Count",
                         color="Category",
                         color_discrete_map={
                             "Excellent (90%+)": "#2ca02c",
                             "Good (75–90%)": "#1f77b4",
                             "Average (50–75%)": "#ff7f0e",
                             "Poor (<50%)": "#d62728"
                         },
                         title="Students by Attendance Category")
        st.plotly_chart(fig_cat, use_container_width=True)
        with st.expander("Explanation", expanded=auto_expand_explanations):
            st.write("Students are bucketed by their personal attendance percentage over the selected date range. This chart shows proportions of students in each bucket — a quick way to spot how many are at risk due to poor attendance.")
    else:
        st.info("Not enough attendance data to compute student-level categories.")

    st.markdown("---")

    # Score distribution with filters
    st.subheader("Score distribution (filtered)")
    filtered_marks = marks_df.copy()
    # apply subject filter and exam filter if any selected
    if subject_filter:
        filtered_marks = filtered_marks[filtered_marks["Subject"].isin(subject_filter)]
    if exam_filter:
        filtered_marks = filtered_marks[filtered_marks["ExamNumber"].isin(exam_filter)]
    if name_search:
        filtered_marks = filtered_marks[filtered_marks["Name"].str.contains(name_search, case=False, na=False)]

    if not filtered_marks.empty:
        fig_hist = px.histogram(filtered_marks, x="Marks", nbins=30, color="Subject" if "Subject" in filtered_marks.columns else None, color_discrete_map=SUBJECT_COLORS or None)
        st.plotly_chart(fig_hist, use_container_width=True)
        with st.expander("Explanation", expanded=auto_expand_explanations):
            st.write("Histogram shows the distribution of marks (respecting the active filters). Use this to understand class spread and detect many low scores.")
    else:
        st.info("No marks after applying filters.")

# ===== TAB 1: Individual Student Report =====
with tabs[1]:
    st.header("Individual Student Report")
    # candidate names
    name_candidates = []
    if not marks_df.empty and "Name" in marks_df.columns:
        name_candidates = sorted(marks_df["Name"].dropna().unique().tolist())
    elif not att_df.empty and "Name" in att_df.columns:
        name_candidates = sorted(att_df["Name"].dropna().unique().tolist())

    if not name_candidates:
        st.info("No student names found. Upload data with a Name column.")
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

        # nice header
        c1, c2 = st.columns([2,3])
        with c1:
            st.markdown(f"### {student}")
            st.markdown(f"**ID:** {sid}  \n**Roll:** {sroll}")
        with c2:
            avg_mark = s_marks["Marks"].mean() if not s_marks.empty and s_marks["Marks"].notna().any() else np.nan
            att_rate = s_att["_present_flag_"].mean() if not s_att.empty and "_present_flag_" in s_att.columns else np.nan
            st.metric("Average mark", f"{avg_mark:.1f}" if not np.isnan(avg_mark) else "N/A")
            st.metric("Attendance rate", f"{att_rate*100:.1f}%" if not np.isnan(att_rate) else "N/A")

        st.markdown("---")

        # Subject performance
        st.subheader("Subject performance (average)")
        if not s_marks.empty and "Subject" in s_marks.columns:
            subj_avg = s_marks.groupby("Subject")["Marks"].mean().reset_index()
            fig_subj = px.bar(subj_avg, x="Subject", y="Marks", color="Subject", color_discrete_map=SUBJECT_COLORS)
            st.plotly_chart(fig_subj, use_container_width=True)
            with st.expander("Explanation", expanded=auto_expand_explanations):
                st.write("Average marks per subject for this student — highlights strengths and weaknesses.")
        else:
            st.info("No subject-level marks for this student.")

        # Marks across exams: grouped bars for clarity
        st.subheader("Marks across exams (grouped by subject)")
        if not s_marks.empty and "ExamNumber" in s_marks.columns and "Subject" in s_marks.columns:
            exam_plot = s_marks.groupby(["ExamNumber","Subject"])["Marks"].mean().reset_index()
            fig_exam = px.bar(exam_plot, x="ExamNumber", y="Marks", color="Subject", barmode="group", color_discrete_map=SUBJECT_COLORS)
            fig_exam.update_layout(xaxis_title="Exam number", yaxis_title="Marks")
            st.plotly_chart(fig_exam, use_container_width=True)
            with st.expander("Explanation", expanded=auto_expand_explanations):
                st.write("Grouped bars show subject-level performance per exam — easier to compare across subjects for each exam.")
        else:
            st.info("Not enough exam/subject structure to draw grouped bars.")

        # Attendance monthly
        st.subheader("Attendance by month")
        if not s_att.empty and "Date" in s_att.columns:
            s_att = s_att.sort_values("Date")
            s_att["month"] = s_att["Date"].dt.to_period("M").astype(str)
            monthly = s_att.groupby("month")["_present_flag_"].mean().reset_index()
            fig_attm = px.bar(monthly, x="month", y="_present_flag_")
            fig_attm.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig_attm, use_container_width=True)
            with st.expander("Explanation", expanded=auto_expand_explanations):
                st.write("Monthly attendance for the student; good replacement for day-level dots which are binary.")
        else:
            st.info("No attendance records for this student.")

        # Export buttons
        st.markdown("---")
        st.subheader("Export student report")
        ex_col, pdf_col = st.columns(2)
        with ex_col:
            excel_bytes = build_student_excel_bytes(student)
            st.download_button("Download Excel", data=excel_bytes, file_name=f"{student}_report.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        with pdf_col:
            pdf_bytes = build_student_pdf_bytes(student)
            st.download_button("Download PDF", data=pdf_bytes, file_name=f"{student}_report.pdf", mime="application/pdf")

# ===== TAB 2: Compare Students =====
with tabs[2]:
    st.header("Compare Students")
    candidate_names = sorted(set(marks_df["Name"].dropna().tolist())) if "Name" in marks_df.columns and not marks_df.empty else []
    selected = st.multiselect("Select up to 6 students", options=candidate_names, max_selections=6)
    exam_sel = st.selectbox("Select exam (All for averages)", options=["All"] + ([str(e) for e in sorted(marks_df["ExamNumber"].dropna().unique().tolist())] if "ExamNumber" in marks_df.columns else []))

    if not selected or len(selected) < 2:
        st.info("Select two or more students to compare.")
    else:
        comp = marks_df[marks_df["Name"].isin(selected)].copy()
        if exam_sel != "All":
            comp = comp[comp["ExamNumber"].astype(str) == exam_sel]

        # Subject-wise grouped bars
        st.subheader("Subject averages comparison")
        comp_avg = comp.groupby(["Name","Subject"])["Marks"].mean().reset_index()
        if not comp_avg.empty:
            subjects = sorted(comp_avg["Subject"].unique())
            fig_comp = go.Figure()
            for name in selected:
                row = comp_avg[comp_avg["Name"]==name]
                y_vals = [row[row["Subject"]==s]["Marks"].values[0] if s in row["Subject"].values else 0 for s in subjects]
                fig_comp.add_trace(go.Bar(name=name, x=subjects, y=y_vals))
            fig_comp.update_layout(barmode="group", xaxis_title="Subject", yaxis_title="Avg marks")
            st.plotly_chart(fig_comp, use_container_width=True)
            with st.expander("Explanation", expanded=auto_expand_explanations):
                st.write("Compare subject-wise averages between selected students. Use the exam selector to focus on a single exam.")
        else:
            st.info("No data for selected students/exam.")

        # Exam trend line across selected students
        st.subheader("Exam trend for selected students")
        if "ExamNumber" in marks_df.columns:
            trend_df = marks_df[marks_df["Name"].isin(selected)].groupby(["ExamNumber","Name"])["Marks"].mean().reset_index()
            if not trend_df.empty:
                fig_trend = px.line(trend_df, x="ExamNumber", y="Marks", color="Name", markers=True)
                fig_trend.update_layout(yaxis_title="Avg marks")
                st.plotly_chart(fig_trend, use_container_width=True)
                with st.expander("Explanation", expanded=auto_expand_explanations):
                    st.write("Lines show how each student's average changed across exams — useful to detect improving or declining performance.")
            else:
                st.info("Not enough exam data to build trends.")
        else:
            st.info("ExamNumber not found in dataset.")

# ===== TAB 3: Attendance explorer =====
with tabs[3]:
    st.header("Attendance explorer")

    if att_df.empty:
        st.info("No attendance data.")
    else:
        # date selector
        min_d = att_df["Date"].min().date() if "Date" in att_df.columns and not att_df["Date"].isna().all() else date.today()
        max_d = att_df["Date"].max().date() if "Date" in att_df.columns and not att_df["Date"].isna().all() else date.today()
        dr = st.date_input("Select date range", value=(min_d, max_d))
        try:
            sd, ed = (dr[0], dr[1]) if isinstance(dr, (list,tuple)) and len(dr)==2 else (dr, dr)
        except Exception:
            sd, ed = min_d, max_d
        mask = (att_df["Date"].dt.date >= sd) & (att_df["Date"].dt.date <= ed)
        att_filtered = att_df[mask]

        # Class daily attendance %
        st.subheader("Class attendance trend (daily %)")
        if not att_filtered.empty:
            att_trend = att_filtered.groupby(att_filtered["Date"].dt.date)["_present_flag_"].mean().reset_index()
            att_trend.columns = ["Date","attendance_rate"]
            fig = px.line(att_trend, x="Date", y="attendance_rate", markers=True)
            fig.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)
            with st.expander("Explanation", expanded=auto_expand_explanations):
                st.write("Daily attendance percentage for the selected date range. Hover to see exact rates.")
        else:
            st.info("No records in selected range.")

        st.markdown("---")
        # Weekday attendance
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
            with st.expander("Explanation", expanded=auto_expand_explanations):
                st.write("Average attendance by weekday — useful to spot recurring low-attendance weekdays.")
        except Exception:
            st.info("Unable to compute weekday breakdown for this data.")

        st.markdown("---")
        # Attendance leaderboard (per-student)
        st.subheader("Attendance leaderboard (top N)")
        n_top = st.number_input("How many students to show:", min_value=5, max_value=200, value=20)
        att_leader = att_filtered.groupby("Name")["_present_flag_"].mean().reset_index().sort_values("_present_flag_", ascending=False)
        if not att_leader.empty:
            leader_disp = att_leader.copy()
            leader_disp["attendance_rate"] = leader_disp["_present_flag_"].apply(lambda x: f"{x:.1%}")
            st.dataframe(leader_disp.set_index("Name").head(n_top))
            with st.expander("Explanation", expanded=auto_expand_explanations):
                st.write("Leaderboard lists students with highest attendance percentage in the selected range.")
        else:
            st.info("No attendance summary available.")

# ===== TAB 4: Insights & Export =====
with tabs[4]:
    st.header("Insights & Export")

    # Top/Bottom performers controls
    st.subheader("Top / Bottom performers")
    k = st.number_input("How many top/bottom performers to show:", min_value=1, max_value=50, value=5)
    if student_summary.empty:
        st.info("Not enough marks data to compute performers.")
    else:
        student_level = student_summary.copy()
        # merge attendance %
        if not att_df.empty and "ID" in att_df.columns:
            att_s = att_df.groupby("ID")["_present_flag_"].mean().reset_index().rename(columns={"_present_flag_":"attendance_rate"})
            # ensure type match for ID
            student_level["ID"] = student_level["ID"].astype(str)
            att_s["ID"] = att_s["ID"].astype(str)
            student_level = student_level.merge(att_s, on="ID", how="left")
        else:
            student_level["attendance_rate"] = np.nan

        topK = student_level.sort_values("avg_score", ascending=False).head(k)[["Name","avg_score","attendance_rate"]]
        botK = student_level.sort_values("avg_score", ascending=True).head(k)[["Name","avg_score","attendance_rate"]]

        # format display
        top_disp = topK.copy()
        top_disp["avg_score"] = top_disp["avg_score"].round(2)
        top_disp["attendance_rate"] = top_disp["attendance_rate"].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")
        bot_disp = botK.copy()
        bot_disp["avg_score"] = bot_disp["avg_score"].round(2)
        bot_disp["attendance_rate"] = bot_disp["attendance_rate"].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")

        ctop, cbot = st.columns(2)
        with ctop:
            st.markdown("**Top performers**")
            st.table(top_disp.set_index("Name"))
        with cbot:
            st.markdown("**Bottom performers**")
            st.table(bot_disp.set_index("Name"))

        with st.expander("Explanation", expanded=auto_expand_explanations):
            st.write("Top/bottom performers are computed by average marks across all recorded exams. Combine with attendance to decide recognition or intervention.")

    st.markdown("---")
    # Flagged students
    st.subheader("Flagged students (low attendance or low average score)")
    if student_level.empty:
        st.info("No student-level data to flag.")
    else:
        flagged = student_level[(student_level["avg_score"] < flag_score_threshold) | (student_level["attendance_rate"].fillna(1) < (flag_att_threshold_pct/100.0))]
        if not flagged.empty:
            flagged_disp = flagged[["ID","Roll","Name","avg_score","attendance_rate"]].copy()
            flagged_disp["avg_score"] = flagged_disp["avg_score"].round(2)
            flagged_disp["attendance_rate"] = flagged_disp["attendance_rate"].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")
            st.dataframe(flagged_disp.set_index("Name"))
            # downloads
            csv_bytes = flagged.to_csv(index=False).encode("utf-8")
            st.download_button("Download flagged (CSV)", data=csv_bytes, file_name="flagged_students.csv", mime="text/csv")
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                flagged.to_excel(writer, index=False, sheet_name="Flagged")
            st.download_button("Download flagged (Excel)", data=buf.getvalue(), file_name="flagged_students.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        else:
            st.success("No flagged students with current thresholds.")

    st.markdown("---")
    # Full exports
    st.subheader("Full dataset export")
    if not marks_df.empty:
        csv_marks = marks_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download marks (CSV)", csv_marks, "marks_export.csv", "text/csv")
    if not att_df.empty:
        csv_att = att_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download attendance (CSV)", csv_att, "attendance_export.csv", "text/csv")

    st.markdown("---")
    st.subheader("Generate class PDF snapshot")
    if st.button("Generate PDF snapshot"):
        # build a simple PDF with class highlights; charts may not embed due to environment
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        doc = SimpleDocTemplate(tmp.name, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []
        story.append(Paragraph("Right iTech — Class snapshot", styles["Title"]))
        story.append(Spacer(1,12))
        story.append(Paragraph(f"Total students: {total_students}", styles["Normal"]))
        if "Gender" in att_df.columns and not att_df.empty:
            story.append(Paragraph(f"Boys: {boys} — Girls: {girls}", styles["Normal"]))
        if not np.isnan(avg_att):
            story.append(Paragraph(f"Average attendance: {avg_att*100:.1f}%", styles["Normal"]))
        story.append(Spacer(1,12))
        doc.build(story)
        with open(tmp.name,"rb") as f:
            pdf_bytes = f.read()
        st.download_button("Download class snapshot PDF", data=pdf_bytes, file_name="class_snapshot.pdf", mime="application/pdf")
        try:
            os.unlink(tmp.name)
        except Exception:
            pass

st.caption("Right iTech — full-featured. If you want theming, logo, or scheduled exports, tell me which next.")
