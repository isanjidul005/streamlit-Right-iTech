# Right_iTech_final.py
# Full-featured Right iTech Streamlit app — copy-paste ready

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

st.set_page_config(page_title="Right iTech", layout="wide", initial_sidebar_state="expanded")

# -------------------------
# Sidebar: uploads + settings
# -------------------------
st.sidebar.header("Data upload & settings")

uploaded_att = st.sidebar.file_uploader("Attendance CSV (optional)", type=["csv"])
uploaded_marks = st.sidebar.file_uploader("Marks CSV (optional)", type=["csv"])

# fallback paths (use only if file exists)
FALLBACK_ATT = "/mnt/data/combined_attendance.csv"
FALLBACK_MARKS = "/mnt/data/cleanest_marks.csv"

@st.cache_data
def safe_read_csv(uploaded_file, fallback_path):
    """Read uploaded file or fallback if exists. Return empty DataFrame if neither available."""
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

# UX toggles & thresholds
auto_expand_explanations = st.sidebar.checkbox("Auto-expand explanations", value=False)
flag_score_threshold = st.sidebar.number_input("Flag if avg score < (score)", min_value=0, max_value=100, value=40)
flag_att_threshold_pct = st.sidebar.slider("Flag if attendance < (%)", min_value=0, max_value=100, value=75)

# color palette and helpers
PALETTE = px.colors.qualitative.Set2
ATT_PRESENT_COLOR = "#2ca02c"
ATT_ABSENT_COLOR = "#d62728"

def assign_subject_colors(subjects):
    subs = sorted([s for s in subjects if pd.notna(s)])
    mapping = {}
    for i, s in enumerate(subs):
        mapping[s] = PALETTE[i % len(PALETTE)]
    return mapping

# -------------------------
# Basic cleaning & safety
# -------------------------
# Ensure dataframes exist
if att_df is None:
    att_df = pd.DataFrame()
if marks_df is None:
    marks_df = pd.DataFrame()

# Normalize column names
att_df.columns = [c.strip() for c in att_df.columns] if not att_df.empty else att_df.columns
marks_df.columns = [c.strip() for c in marks_df.columns] if not marks_df.empty else marks_df.columns

# Safe date parsing
if not att_df.empty and "Date" in att_df.columns:
    att_df["Date"] = pd.to_datetime(att_df["Date"], dayfirst=True, errors="coerce")

# Create unified present flag
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

# marks numeric
if not marks_df.empty:
    if "Marks" in marks_df.columns:
        marks_df["Marks"] = pd.to_numeric(marks_df["Marks"], errors="coerce")
    else:
        marks_df["Marks"] = np.nan
    if "FullMarks" in marks_df.columns:
        marks_df["FullMarks"] = pd.to_numeric(marks_df["FullMarks"], errors="coerce")
else:
    marks_df = pd.DataFrame(columns=["ID","Roll","Name","Subject","ExamNumber","Exam","ExamType","Marks","FullMarks"])

# Subject colors
SUBJECT_COLORS = assign_subject_colors(marks_df["Subject"].unique()) if "Subject" in marks_df.columns and not marks_df.empty else {}

# -------------------------
# Utility functions
# -------------------------
def fig_to_png_bytes(fig, scale=2):
    """Return PNG bytes for a plotly figure if possible, else None."""
    try:
        return fig.to_image(format="png", scale=scale)  # requires kaleido
    except Exception:
        try:
            return pio.to_image(fig, format="png", scale=scale)
        except Exception:
            return None

def build_student_excel_bytes(student_name):
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        s_m = marks_df[marks_df["Name"]==student_name] if not marks_df.empty and "Name" in marks_df.columns else pd.DataFrame()
        s_a = att_df[att_df["Name"]==student_name] if not att_df.empty and "Name" in att_df.columns else pd.DataFrame()
        s_m.to_excel(writer, sheet_name="Marks", index=False)
        s_a.to_excel(writer, sheet_name="Attendance", index=False)
    out.seek(0)
    return out.getvalue()

def build_student_pdf_bytes(student_name):
    """Return PDF bytes for a single student: text summary + embedded charts (when possible)."""
    s_m = marks_df[marks_df["Name"]==student_name] if not marks_df.empty and "Name" in marks_df.columns else pd.DataFrame()
    s_a = att_df[att_df["Name"]==student_name] if not att_df.empty and "Name" in att_df.columns else pd.DataFrame()

    out_buf = io.BytesIO()
    doc = SimpleDocTemplate(out_buf, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []
    story.append(Paragraph(f"Student Report — {student_name}", styles["Title"]))
    story.append(Spacer(1,12))

    if not s_m.empty and "Marks" in s_m.columns:
        story.append(Paragraph(f"Average marks: {s_m['Marks'].mean():.2f}", styles["Normal"]))
    if not s_a.empty and "_present_flag_" in s_a.columns and len(s_a) > 0 and not s_a["_present_flag_"].isna().all():
        p = int(s_a["_present_flag_"].sum()); t = len(s_a)
        story.append(Paragraph(f"Attendance: {p}/{t} ({(p/t*100):.1f}%)", styles["Normal"]))
    story.append(Spacer(1,12))

    temps = []
    # try embed subject avg chart
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
            s_a2 = s_a.copy()
            s_a2["month"] = s_a2["Date"].dt.to_period("M").astype(str)
            monthly = s_a2.groupby("month")["_present_flag_"].mean().reset_index()
            fig2 = px.bar(monthly, x="month", y="_present_flag_")
            img2 = fig_to_png_bytes(fig2)
            if img2:
                tmp2 = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                tmp2.write(img2); tmp2.flush()
                temps.append(tmp2.name)
                story.append(RLImage(tmp2.name, width=450, height=150))
                story.append(Spacer(1,12))
    except Exception:
        # don't fail PDF if images can't be created
        pass

    story.append(Paragraph("Notes & recommendations:", styles["Heading3"]))
    story.append(Paragraph("Consider remedial sessions for subjects with low averages and contact parents for students flagged under Insights.", styles["Normal"]))
    doc.build(story)

    # cleanup temps
    try:
        for p in temps:
            os.unlink(p)
    except Exception:
        pass

    out_buf.seek(0)
    return out_buf.getvalue()

# -------------------------
# App header
# -------------------------
st.markdown("<h1 style='margin-bottom:4px'>Right iTech — Student Insights</h1>", unsafe_allow_html=True)
st.markdown("<div style='color:#666;margin-bottom:14px'>Upload attendance & marks CSVs (or place files at /mnt/data/combined_attendance.csv and /mnt/data/cleanest_marks.csv on the server).</div>", unsafe_allow_html=True)
st.write("---")

# -------------------------
# Basic requirement: if no data, show guidance (but do not crash)
# -------------------------
if att_df.empty and marks_df.empty:
    st.error("No data found. Upload Attendance and Marks CSVs in the sidebar or place fallback files at /mnt/data/*.csv.")
    st.stop()

# -------------------------
# Filters (global)
# -------------------------
st.sidebar.header("Global filters")
# date range
if not att_df.empty and "Date" in att_df.columns and not att_df["Date"].isna().all():
    min_date = att_df["Date"].min().date()
    max_date = att_df["Date"].max().date()
else:
    min_date = date.today()
    max_date = date.today()
date_range = st.sidebar.date_input("Attendance date range", value=(min_date, max_date))

# subject filter
subject_options = sorted(marks_df["Subject"].dropna().unique().tolist()) if not marks_df.empty and "Subject" in marks_df.columns else []
subject_filter = st.sidebar.multiselect("Filter subjects (optional)", options=subject_options, default=subject_options)

# exam filter
exam_options = sorted(marks_df["ExamNumber"].dropna().unique().tolist()) if not marks_df.empty and "ExamNumber" in marks_df.columns else []
exam_filter = st.sidebar.multiselect("Filter exams (optional)", options=exam_options, default=exam_options)

# name quick search
name_search = st.sidebar.text_input("Search student name (partial)")

# -------------------------
# Tabs layout
# -------------------------
tabs_ui = st.tabs(["Class overview","Individual Student Report","Compare Students","Attendance explorer","Insights & Export"])

# ===== TAB 0: Class overview =====
with tabs_ui[0]:
    st.header("Class overview")

    # snapshot
    col1, col2, col3, col4 = st.columns(4)
    ids_set = set()
    if "ID" in marks_df.columns and not marks_df.empty:
        ids_set.update(marks_df["ID"].dropna().astype(str).tolist())
    if "ID" in att_df.columns and not att_df.empty:
        ids_set.update(att_df["ID"].dropna().astype(str).tolist())
    total_students = len(ids_set) if ids_set else (marks_df["Name"].nunique() if not marks_df.empty and "Name" in marks_df.columns else 0)
    col1.metric("Total students", total_students)

    # gender
    if "Gender" in att_df.columns and not att_df.empty:
        gender_counts = att_df.drop_duplicates(subset=["ID"])[["ID","Gender"]].groupby("Gender").size().to_dict()
        boys = gender_counts.get("M", gender_counts.get("Male", 0))
        girls = gender_counts.get("F", gender_counts.get("Female", 0))
        col2.metric("Boys", int(boys))
        col3.metric("Girls", int(girls))
    else:
        col2.metric("Boys", "N/A"); col3.metric("Girls", "N/A")

    avg_att = att_df["_present_flag_"].mean() if not att_df.empty else np.nan
    col4.metric("Avg attendance", f"{avg_att*100:.1f}%" if not np.isnan(avg_att) else "N/A")

    st.markdown("---")

    # Subject ranking (no walrus)
    st.subheader("Subject ranking — class average")
    if not marks_df.empty and "Subject" in marks_df.columns:
        subj_summary = marks_df.groupby("Subject")["Marks"].mean().reset_index().rename(columns={"Marks":"avg_score"})
        if not subj_summary.empty:
            subj_sorted = subj_summary.sort_values("avg_score", ascending=False)
            fig_sub = px.bar(subj_sorted, x="avg_score", y="Subject", orientation="h", color="avg_score", color_continuous_scale="Blues")
            fig_sub.update_layout(xaxis_title="Average score", yaxis_title="")
            st.plotly_chart(fig_sub, use_container_width=True)
            with st.expander("Explanation", expanded=auto_expand_explanations):
                best = subj_sorted.iloc[0]["Subject"]
                worst = subj_sorted.iloc[-1]["Subject"]
                st.write(f"Subjects ranked by class average. Best: **{best}** — Lowest: **{worst}**.")
        else:
            st.info("Not enough subject data to compute ranking.")
    else:
        st.info("Not enough subject data to compute ranking.")

    st.markdown("---")

    # Attendance categories distribution (meaningful pie)
    st.subheader("Attendance categories (student-level)")
    if not att_df.empty and "Name" in att_df.columns and "_present_flag_" in att_df.columns:
        # filter by date range
        try:
            sd, ed = (date_range[0], date_range[1]) if isinstance(date_range, (list,tuple)) and len(date_range)==2 else (date_range, date_range)
        except Exception:
            sd, ed = min_date, max_date
        mask = (att_df["Date"].dt.date >= sd) & (att_df["Date"].dt.date <= ed) if "Date" in att_df.columns else slice(None)
        att_in_range = att_df[mask] if mask is not slice(None) else att_df.copy()
        student_att = att_in_range.groupby("Name")["_present_flag_"].mean().reset_index()
        student_att["attendance_pct"] = student_att["_present_flag_"] * 100

        def categorize(rate):
            if rate >= 90:
                return "Excellent (90%+)"
            elif rate >= 75:
                return "Good (75–90%)"
            elif rate >= 50:
                return "Average (50–75%)"
            else:
                return "Poor (<50%)"

        student_att["Category"] = student_att["attendance_pct"].apply(categorize)
        counts = student_att["Category"].value_counts().reindex(["Excellent (90%+)","Good (75–90%)","Average (50–75%)","Poor (<50%)"]).fillna(0).reset_index()
        counts.columns = ["Category","Count"]
        fig_cat = px.pie(counts, names="Category", values="Count",
                         color="Category",
                         color_discrete_map={
                             "Excellent (90%+)":"#2ca02c",
                             "Good (75–90%)":"#1f77b4",
                             "Average (50–75%)":"#ff7f0e",
                             "Poor (<50%)":"#d62728"
                         },
                         title="Students by Attendance Category")
        st.plotly_chart(fig_cat, use_container_width=True)
        with st.expander("ℹ️ Explanation", expanded=auto_expand_explanations):
            st.write("Buckets students by their personal attendance percent (selected date range). See how many are at-risk.")
    else:
        st.info("Not enough attendance data to compute student-level categories.")

    st.markdown("---")

    # Score distribution with simple filters applied
    st.subheader("Score distribution (filtered)")
    filtered_marks = marks_df.copy()
    if subject_filter:
        filtered_marks = filtered_marks[filtered_marks["Subject"].isin(subject_filter)]
    if exam_filter:
        filtered_marks = filtered_marks[filtered_marks["ExamNumber"].isin(exam_filter)]
    if name_search:
        filtered_marks = filtered_marks[filtered_marks["Name"].str.contains(name_search, case=False, na=False)]

    if not filtered_marks.empty:
        fig_hist = px.histogram(filtered_marks, x="Marks", nbins=30,
                                color="Subject" if "Subject" in filtered_marks.columns else None,
                                color_discrete_map=SUBJECT_COLORS or None)
        st.plotly_chart(fig_hist, use_container_width=True)
        with st.expander("ℹ️ Explanation", expanded=auto_expand_explanations):
            st.write("Distribution of marks after applying filters; helps detect many low performers or skew.")
    else:
        st.info("No marks after applying filters.")

# ===== TAB 1: Individual Student Report =====
with tabs_ui[1]:
    st.header("Individual Student Report")
    name_candidates = []
    if not marks_df.empty and "Name" in marks_df.columns:
        name_candidates = sorted(marks_df["Name"].dropna().unique().tolist())
    elif not att_df.empty and "Name" in att_df.columns:
        name_candidates = sorted(att_df["Name"].dropna().unique().tolist())

    if not name_candidates:
        st.info("No student names found in provided data.")
    else:
        student = st.selectbox("Select student", name_candidates)
        s_marks = marks_df[marks_df["Name"] == student] if not marks_df.empty else pd.DataFrame()
        s_att = att_df[att_df["Name"] == student] if not att_df.empty else pd.DataFrame()

        # Header card
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

        # Subject performance
        st.subheader("Subject performance (average)")
        if not s_marks.empty and "Subject" in s_marks.columns:
            subj_avg = s_marks.groupby("Subject")["Marks"].mean().reset_index()
            fig_subj = px.bar(subj_avg, x="Subject", y="Marks", color="Subject", color_discrete_map=SUBJECT_COLORS)
            st.plotly_chart(fig_subj, use_container_width=True)
            with st.expander("Explanation", expanded=auto_expand_explanations):
                st.write("Average marks per subject for this student — highlights strengths/weaknesses.")
        else:
            st.info("No subject-level marks for this student.")

        # Marks across exams grouped
        st.subheader("Marks across exams (grouped by subject)")
        if not s_marks.empty and "ExamNumber" in s_marks.columns and "Subject" in s_marks.columns:
            exam_plot = s_marks.groupby(["ExamNumber","Subject"])["Marks"].mean().reset_index()
            fig_exam = px.bar(exam_plot, x="ExamNumber", y="Marks", color="Subject", barmode="group", color_discrete_map=SUBJECT_COLORS)
            fig_exam.update_layout(xaxis_title="Exam number", yaxis_title="Marks")
            st.plotly_chart(fig_exam, use_container_width=True)
            with st.expander("Explanation", expanded=auto_expand_explanations):
                st.write("Grouped bars show subject-level performance per exam.")
        else:
            st.info("Not enough exam/subject structure to show grouped bars.")

        # Attendance monthly
        st.subheader("Attendance by month")
        if not s_att.empty and "Date" in s_att.columns:
            s_att2 = s_att.sort_values("Date").copy()
            s_att2["month"] = s_att2["Date"].dt.to_period("M").astype(str)
            monthly = s_att2.groupby("month")["_present_flag_"].mean().reset_index()
            fig_attm = px.bar(monthly, x="month", y="_present_flag_")
            fig_attm.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig_attm, use_container_width=True)
            with st.expander("Explanation", expanded=auto_expand_explanations):
                st.write("Monthly attendance percentage for the student.")
        else:
            st.info("No attendance records for this student.")

        st.markdown("---")
        # Exports for student
        c1, c2 = st.columns(2)
        with c1:
            excel_bytes = build_student_excel_bytes(student)
            st.download_button("Download Excel report", data=excel_bytes, file_name=f"{student}_report.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        with c2:
            pdf_bytes = build_student_pdf_bytes(student)
            st.download_button("Download PDF report", data=pdf_bytes, file_name=f"{student}_report.pdf", mime="application/pdf")

# ===== TAB 2: Compare Students =====
with tabs_ui[2]:
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

        st.subheader("Average Marks by Subject (Comparison)")
        comp_avg = comp.groupby(["Name","Subject"])["Marks"].mean().reset_index()
        if not comp_avg.empty:
            subjects = sorted(comp_avg["Subject"].unique())
            fig = go.Figure()
            for name in selection:
                row = comp_avg[comp_avg["Name"]==name]
                y_vals = [row[row["Subject"]==s]["Marks"].values[0] if s in row["Subject"].values else 0 for s in subjects]
                fig.add_trace(go.Bar(name=name, x=subjects, y=y_vals))
            fig.update_layout(barmode="group", xaxis_title="Subject", yaxis_title="Avg marks")
            st.plotly_chart(fig, use_container_width=True)
            with st.expander("Explanation", expanded=auto_expand_explanations):
                st.write("Grouped bars compare subject averages between selected students.")
        else:
            st.info("No subject averages available for selected students/exam.")

        st.subheader("Exam trend (selected students)")
        if "ExamNumber" in marks_df.columns:
            trend_df = marks_df[marks_df["Name"].isin(selection)].groupby(["ExamNumber","Name"])["Marks"].mean().reset_index()
            if not trend_df.empty:
                fig_trend = px.line(trend_df, x="ExamNumber", y="Marks", color="Name", markers=True)
                fig_trend.update_layout(yaxis_title="Avg marks")
                st.plotly_chart(fig_trend, use_container_width=True)
                with st.expander("Explanation", expanded=auto_expand_explanations):
                    st.write("Shows how each student's average changed across exams.")
            else:
                st.info("Not enough exam data to draw trends.")
        else:
            st.info("ExamNumber column not found in marks data.")

# ===== TAB 3: Attendance explorer =====
with tabs_ui[3]:
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

        st.subheader("Class attendance trend (daily %)")
        if not att_filtered.empty:
            att_trend = att_filtered.groupby(att_filtered["Date"].dt.date)["_present_flag_"].mean().reset_index()
            att_trend.columns = ["Date","attendance_rate"]
            fig = px.line(att_trend, x="Date", y="attendance_rate", markers=True)
            fig.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)
            with st.expander("Explanation", expanded=auto_expand_explanations):
                st.write("Daily class attendance percentage in selected range.")
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
            with st.expander("Explanation", expanded=auto_expand_explanations):
                st.write("Average attendance per weekday — reveals patterns like low attendance on certain weekdays.")
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
            with st.expander("Explanation", expanded=auto_expand_explanations):
                st.write("Leaderboard shows students with highest attendance in the selected period.")
        else:
            st.info("No attendance summary available for the selected range.")

# ===== TAB 4: Insights & Export =====
with tabs_ui[4]:
    st.header("Insights & Export")

    # student-level summary (avg marks)
    def student_marks_summary(df):
        if df.empty:
            return pd.DataFrame()
        s = df.groupby(["ID","Roll","Name"]).agg(
            avg_score=("Marks","mean"),
            exams_taken=("ExamNumber","nunique"),
            records_count=("Marks","count")
        ).reset_index()
        return s

    student_level = student_marks_summary(marks_df)

    st.subheader("Top / Bottom performers")
    k = st.number_input("How many top/bottom performers to show", min_value=1, max_value=50, value=5)
    if student_level.empty:
        st.info("Not enough marks data to show performers.")
    else:
        if not att_df.empty and "ID" in att_df.columns:
            att_s = att_df.groupby("ID")["_present_flag_"].mean().reset_index().rename(columns={"_present_flag_":"attendance_rate"})
            # ensure string IDs match
            student_level["ID"] = student_level["ID"].astype(str)
            att_s["ID"] = att_s["ID"].astype(str)
            student_level = student_level.merge(att_s, on="ID", how="left")
        else:
            student_level["attendance_rate"] = np.nan

        topK = student_level.sort_values("avg_score", ascending=False).head(k)[["Name","avg_score","attendance_rate"]]
        botK = student_level.sort_values("avg_score", ascending=True).head(k)[["Name","avg_score","attendance_rate"]]

        top_disp = topK.copy()
        top_disp["avg_score"] = top_disp["avg_score"].round(2)
        top_disp["attendance_rate"] = top_disp["attendance_rate"].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")
        bot_disp = botK.copy()
        bot_disp["avg_score"] = bot_disp["avg_score"].round(2)
        bot_disp["attendance_rate"] = bot_disp["attendance_rate"].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Top performers**")
            if not top_disp.empty:
                st.table(top_disp.set_index("Name"))
            else:
                st.info("No top performers to show.")
        with c2:
            st.markdown("**Bottom performers**")
            if not bot_disp.empty:
                st.table(bot_disp.set_index("Name"))
            else:
                st.info("No bottom performers to show.")
        with st.expander("Explanation", expanded=auto_expand_explanations):
            st.write("Top and bottom performers are computed by average marks. Use this list to plan recognition or intervention.")

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
            st.success("No flagged students with current thresholds.")

    st.markdown("---")
    st.subheader("Full dataset export")
    if not marks_df.empty:
        st.download_button("Download marks (CSV)", data=marks_df.to_csv(index=False).encode("utf-8"), file_name="marks_export.csv", mime="text/csv")
    if not att_df.empty:
        st.download_button("Download attendance (CSV)", data=att_df.to_csv(index=False).encode("utf-8"), file_name="attendance_export.csv", mime="text/csv")

    st.markdown("---")
    st.subheader("Generate class PDF snapshot (includes charts where possible)")
    if st.button("Generate PDF snapshot"):
        # build simple PDF with charts embedded if possible
        tmp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        doc = SimpleDocTemplate(tmp_pdf.name, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []
        story.append(Paragraph("Right iTech — Class snapshot", styles["Title"]))
        story.append(Spacer(1,12))
        story.append(Paragraph(f"Total students: {total_students}", styles["Normal"]))
        if 'boys' in locals() and 'girls' in locals():
            story.append(Paragraph(f"Boys: {boys} — Girls: {girls}", styles["Normal"]))
        if not np.isnan(avg_att):
            story.append(Paragraph(f"Average attendance: {avg_att*100:.1f}%", styles["Normal"]))
        story.append(Spacer(1,12))

        temps = []
        try:
            # Attendance pie
            att_summary = pd.DataFrame({"Status":["Present","Absent"], "Rate":[avg_att, 1-avg_att]})
            fig_p = px.pie(att_summary, names="Status", values="Rate", color="Status", color_discrete_map={"Present":ATT_PRESENT_COLOR,"Absent":ATT_ABSENT_COLOR})
            img = fig_to_png_bytes(fig_p)
            if img:
                tmp1 = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                tmp1.write(img); tmp1.flush()
                temps.append(tmp1.name)
                story.append(RLImage(tmp1.name, width=450, height=200))
                story.append(Spacer(1,12))
            # Attendance trend
            if not att_df.empty:
                att_trend = att_df.groupby('Date')["_present_flag_"].mean().reset_index()
                fig_t = px.line(att_trend, x="Date", y="_present_flag_")
                fig_t.update_yaxes(tickformat=".0%")
                img2 = fig_to_png_bytes(fig_t)
                if img2:
                    tmp2 = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                    tmp2.write(img2); tmp2.flush()
                    temps.append(tmp2.name)
                    story.append(RLImage(tmp2.name, width=450, height=200))
                    story.append(Spacer(1,12))
            # Top performers chart
            if not marks_df.empty:
                avg_marks = marks_df.groupby('Name')['Marks'].mean().reset_index().sort_values('Marks', ascending=False).head(5)
                fig_top = px.bar(avg_marks, x='Name', y='Marks', color='Marks', color_continuous_scale='greens')
                img3 = fig_to_png_bytes(fig_top)
                if img3:
                    tmp3 = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                    tmp3.write(img3); tmp3.flush()
                    temps.append(tmp3.name)
                    story.append(RLImage(tmp3.name, width=450, height=200))
                    story.append(Spacer(1,12))
        except Exception:
            pass

        story.append(Paragraph("Notes: Use Flags to identify students needing intervention.", styles["Normal"]))
        doc.build(story)
        with open(tmp_pdf.name, "rb") as f:
            pdf_bytes = f.read()
        st.download_button("Download class snapshot PDF", data=pdf_bytes, file_name="class_snapshot.pdf", mime="application/pdf")
        try:
            os.unlink(tmp_pdf.name)
            for p in temps:
                os.unlink(p)
        except Exception:
            pass

st.caption("Right iTech — final. If anything errors, paste the exact traceback and I will patch that exact line immediately.")
