# Right_iTech_streamlit.py
# Full, packed Streamlit app — Plotly visualizations, safe colors, broad explanations, no exports.

import os
import io
from datetime import date
import tempfile

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# -------------------------
# Page config + theme
# -------------------------
st.set_page_config(page_title="Right iTech", layout="wide", initial_sidebar_state="expanded")
px.defaults.template = "plotly_white"

# Calm, professional palette (consistent)
PALETTE = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#17becf"]
ATT_PRESENT_COLOR = "#2ca02c"
ATT_ABSENT_COLOR = "#d62728"
NEUTRAL = "#6c757d"
CARD_BG = "#ffffff"

# -------------------------
# Sidebar: upload + settings
# -------------------------
st.sidebar.header("Upload data & preferences")

att_upload = st.sidebar.file_uploader("Upload attendance CSV (optional)", type=["csv"])
marks_upload = st.sidebar.file_uploader("Upload marks CSV (optional)", type=["csv"])

# fallback files on server (only used if present)
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

att_df = safe_read_csv(att_upload, FALLBACK_ATT)
marks_df = safe_read_csv(marks_upload, FALLBACK_MARKS)

st.sidebar.markdown("---")
auto_expand = st.sidebar.checkbox("Auto-expand explanations", value=False)
pass_threshold = st.sidebar.number_input("Pass threshold (marks)", min_value=0, max_value=100, value=40)
flag_score_threshold = st.sidebar.number_input("Flag if avg score < (score)", min_value=0, max_value=100, value=40)
flag_att_threshold_pct = st.sidebar.slider("Flag if attendance < (%)", 0, 100, 75)

# -------------------------
# Small CSS to ensure title doesn't have white block and looks readable
# -------------------------
st.markdown(
    """
    <style>
      .app-title {
        font-family: "Segoe UI", Roboto, Arial, sans-serif;
        color: #0b3d91;
        padding: 8px 0px;
        margin-bottom: 6px;
      }
      .sub-text {
        color: #555;
        margin-bottom: 12px;
      }
      .card {
        background: #ffffff;
        border-radius: 8px;
        padding: 10px;
        box-shadow: 0 4px 14px rgba(31,119,180,0.06);
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div class='card'><div class='app-title'><h1 style='margin:0'>Right iTech</h1></div><div class='sub-text'>Clean, readable analytics for marks & attendance — interactive visuals.</div></div>", unsafe_allow_html=True)
st.write("---")

# -------------------------
# Defensive cleaning
# -------------------------
if att_df is None:
    att_df = pd.DataFrame()
if marks_df is None:
    marks_df = pd.DataFrame()

# Normalize column names (strip)
if not att_df.empty:
    att_df.columns = [c.strip() for c in att_df.columns]
if not marks_df.empty:
    marks_df.columns = [c.strip() for c in marks_df.columns]

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

# Ensure numeric marks
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
    subs = sorted([s for s in subjects if pd.notna(s)])
    mapping = {}
    for i,s in enumerate(subs):
        mapping[s] = PALETTE[i % len(PALETTE)]
    return mapping

SUBJECT_COLORS = assign_subject_colors(marks_df["Subject"].unique()) if ("Subject" in marks_df.columns and not marks_df.empty) else {}

# -------------------------
# Helper functions
# -------------------------
def student_summary_df(mdf):
    if mdf.empty:
        return pd.DataFrame()
    s = mdf.groupby(["ID","Roll","Name"], as_index=False).agg(avg_score=("Marks","mean"), exams_taken=("ExamNumber","nunique"), records=("Marks","count"))
    return s

def pct(x):
    try:
        return f"{x*100:.1f}%"
    except Exception:
        return "N/A"

# -------------------------
# Stop early if no data at all
# -------------------------
if att_df.empty and marks_df.empty:
    st.warning("No data detected. Upload Attendance and Marks CSVs in the sidebar, or put fallback files at /mnt/data/*.csv")
    st.stop()

# -------------------------
# Global filters (sidebar)
# -------------------------
st.sidebar.header("Global Filters")
if not att_df.empty and "Date" in att_df.columns and not att_df["Date"].isna().all():
    min_date = att_df["Date"].min().date()
    max_date = att_df["Date"].max().date()
else:
    min_date = date.today(); max_date = date.today()

date_range = st.sidebar.date_input("Attendance date range", value=(min_date, max_date))

subject_opts = sorted(marks_df["Subject"].dropna().unique().tolist()) if (not marks_df.empty and "Subject" in marks_df.columns) else []
subject_filter = st.sidebar.multiselect("Filter subjects", options=subject_opts, default=subject_opts)

exam_opts = sorted(marks_df["ExamNumber"].dropna().unique().tolist()) if (not marks_df.empty and "ExamNumber" in marks_df.columns) else []
exam_filter = st.sidebar.multiselect("Filter exams", options=exam_opts, default=exam_opts)

name_search = st.sidebar.text_input("Search student name (partial)")

# -------------------------
# Tabs
# -------------------------
tabs = st.tabs(["Class Overview","Student Dashboard","Compare Students","Attendance","Marks","Insights"])

# ===== Tab: Class Overview =====
with tabs[0]:
    st.header("Class Overview (snapshot)")

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    ids_set = set()
    if "ID" in marks_df.columns and not marks_df.empty:
        ids_set.update(marks_df["ID"].dropna().astype(str).tolist())
    if "ID" in att_df.columns and not att_df.empty:
        ids_set.update(att_df["ID"].dropna().astype(str).tolist())
    total_students = len(ids_set) if ids_set else (marks_df["Name"].nunique() if not marks_df.empty and "Name" in marks_df.columns else 0)
    col1.metric("Total students", total_students)

    # gender counts (if available)
    if not att_df.empty and "Gender" in att_df.columns:
        gender_df = att_df.drop_duplicates(subset=["ID"]).groupby("Gender").size().to_dict()
        boys = gender_df.get("M", gender_df.get("Male", 0))
        girls = gender_df.get("F", gender_df.get("Female", 0))
        col2.metric("Boys", int(boys))
        col3.metric("Girls", int(girls))
    else:
        col2.metric("Boys", "N/A"); col3.metric("Girls", "N/A")

    avg_att = att_df["_present_flag_"].mean() if not att_df.empty else np.nan
    col4.metric("Avg attendance", f"{avg_att*100:.1f}%" if not np.isnan(avg_att) else "N/A")

    st.markdown("---")

    # Score distribution (histogram)
    st.subheader("Score distribution")
    filtered_marks = marks_df.copy()
    if subject_filter:
        filtered_marks = filtered_marks[filtered_marks["Subject"].isin(subject_filter)]
    if exam_filter:
        filtered_marks = filtered_marks[filtered_marks["ExamNumber"].isin(exam_filter)]
    if name_search:
        filtered_marks = filtered_marks[filtered_marks["Name"].str.contains(name_search, case=False, na=False)]

    if not filtered_marks.empty and "Marks" in filtered_marks.columns:
        fig_hist = px.histogram(filtered_marks, x="Marks", nbins=25, title="Distribution of marks (filtered)", color_discrete_sequence=[PALETTE[0]])
        st.plotly_chart(fig_hist, use_container_width=True)
        with st.expander("Explanation", expanded=auto_expand):
            st.write(
                "Histogram shows how marks are distributed across records (respecting filters). "
                "Look for clustering (many low or many high scores). Use filters to focus by subject or exam."
            )
    else:
        st.info("Not enough marks data to show distribution.")

    st.markdown("---")

    # Attendance categories pie (student-level buckets)
    st.subheader("Attendance quality (student buckets)")
    if not att_df.empty and "Name" in att_df.columns and "_present_flag_" in att_df.columns:
        try:
            sd, ed = (date_range[0], date_range[1]) if isinstance(date_range, (list,tuple)) and len(date_range)==2 else (date_range, date_range)
        except Exception:
            sd, ed = min_date, max_date
        if "Date" in att_df.columns:
            mask = (att_df["Date"].dt.date >= sd) & (att_df["Date"].dt.date <= ed)
            att_range = att_df[mask]
        else:
            att_range = att_df.copy()

        per_student = att_range.groupby("Name")["_present_flag_"].mean().reset_index().rename(columns={"_present_flag_":"att_mean"})
        per_student["pct"] = per_student["att_mean"] * 100
        def bucket(v):
            if v >= 90: return "Excellent (90%+)"
            if v >= 75: return "Good (75–90%)"
            if v >= 50: return "Average (50–75%)"
            return "Poor (<50%)"
        per_student["bucket"] = per_student["pct"].apply(bucket)
        counts = per_student["bucket"].value_counts().reindex(["Excellent (90%+)","Good (75–90%)","Average (50–75%)","Poor (<50%)"]).fillna(0).reset_index()
        counts.columns = ["Category","Count"]
        fig_pie = px.pie(counts, names="Category", values="Count", title="Students by attendance quality",
                         color_discrete_sequence=["#2ca02c","#17becf","#ff7f0e","#d62728"])
        st.plotly_chart(fig_pie, use_container_width=True)
        with st.expander("Explanation", expanded=auto_expand):
            st.write(
                "Students are bucketed by personal attendance percentage over the selected date range. "
                "This shows how many are in each risk bracket and helps prioritize follow-up for 'Poor' students."
            )
    else:
        st.info("Not enough attendance data for student-level attendance buckets.")

    st.markdown("---")

    # Subject-level overview (avg, median, pass rate)
    st.subheader("Subject-level summary")
    if not marks_df.empty and "Subject" in marks_df.columns and "Marks" in marks_df.columns:
        subj = marks_df.groupby("Subject").agg(avg_score=("Marks","mean"), median_score=("Marks","median"), count=("Marks","count")).reset_index()
        subj = subj.sort_values("avg_score", ascending=False)
        fig_sub_avg = px.bar(subj, x="Subject", y="avg_score", title="Average score by subject", color="avg_score", color_continuous_scale="Blues")
        st.plotly_chart(fig_sub_avg, use_container_width=True)
        with st.expander("Explanation", expanded=auto_expand):
            st.write(
                "Average scores per subject show which subjects the class performs best/worst in. "
                "Median score (not shown) can be used to check for skew from outliers."
            )

        # pass rate
        if "Marks" in marks_df.columns:
            subj_pass = marks_df.copy()
            subj_pass["pass"] = subj_pass["Marks"] >= pass_threshold
            pass_summary = subj_pass.groupby("Subject")["pass"].mean().reset_index().rename(columns={"pass":"pass_rate"}).sort_values("pass_rate", ascending=False)
            fig_pass = px.bar(pass_summary, x="Subject", y="pass_rate", title=f"Pass rate per subject (>= {pass_threshold})")
            fig_pass.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig_pass, use_container_width=True)
            with st.expander("Explanation", expanded=auto_expand):
                st.write("Pass rate shows the share of recorded marks at-or-above the pass threshold. Subjects with low pass rate need targeted support.")
    else:
        st.info("Not enough marks data to compute subject-level summary.")

# ===== Tab: Student Dashboard (single student overview, renamed) =====
with tabs[1]:
    st.header("Student Dashboard")
    # list of students
    students = []
    if "Name" in marks_df.columns and not marks_df.empty:
        students = sorted(marks_df["Name"].dropna().unique().tolist())
    elif "Name" in att_df.columns and not att_df.empty:
        students = sorted(att_df["Name"].dropna().unique().tolist())

    if not students:
        st.info("No student names found in dataset.")
    else:
        student = st.selectbox("Choose student", students)
        s_marks = marks_df[marks_df["Name"]==student] if not marks_df.empty else pd.DataFrame()
        s_att = att_df[att_df["Name"]==student] if not att_df.empty else pd.DataFrame()

        # profile
        st.subheader(f"{student}")
        colA, colB = st.columns([2,3])
        with colA:
            sid = s_marks["ID"].iloc[0] if (not s_marks.empty and "ID" in s_marks.columns) else (s_att["ID"].iloc[0] if (not s_att.empty and "ID" in s_att.columns) else "N/A")
            sroll = s_marks["Roll"].iloc[0] if (not s_marks.empty and "Roll" in s_marks.columns) else (s_att["Roll"].iloc[0] if (not s_att.empty and "Roll" in s_att.columns) else "N/A")
            st.markdown(f"**ID:** {sid}  \n**Roll:** {sroll}")
        with colB:
            avg_mark = s_marks["Marks"].mean() if not s_marks.empty and "Marks" in s_marks.columns else np.nan
            att_rate = s_att["_present_flag_"].mean() if not s_att.empty and "_present_flag_" in s_att.columns else np.nan
            st.metric("Average mark", f"{avg_mark:.1f}" if not np.isnan(avg_mark) else "N/A")
            st.metric("Attendance rate", f"{att_rate*100:.1f}%" if not np.isnan(att_rate) else "N/A")

        st.markdown("---")

        # Subject strengths (simple bar)
        st.subheader("Subject strengths & weaknesses")
        if not s_marks.empty and "Subject" in s_marks.columns:
            subj_avg = s_marks.groupby("Subject")["Marks"].mean().reset_index().sort_values("Marks", ascending=False)
            fig_subj = px.bar(subj_avg, x="Subject", y="Marks", color="Marks", color_continuous_scale=["#ff7f0e","#1f77b4"])
            st.plotly_chart(fig_subj, use_container_width=True)
            with st.expander("Explanation", expanded=auto_expand):
                st.write("Shows the student's average per subject. Highest bars indicate strengths; lowest indicate subjects to improve.")
        else:
            st.info("No marks for this student.")

        st.markdown("---")

        # Marks across exams (clear line chart per subject)
        st.subheader("Marks across exams (trend by subject)")
        if not s_marks.empty and "ExamNumber" in s_marks.columns and "Subject" in s_marks.columns:
            trend_df = s_marks.groupby(["ExamNumber","Subject"])["Marks"].mean().reset_index()
            fig_trends = px.line(trend_df, x="ExamNumber", y="Marks", color="Subject", markers=True, title="Marks by exam (subject lines)")
            st.plotly_chart(fig_trends, use_container_width=True)
            with st.expander("Explanation", expanded=auto_expand):
                st.write("Each line is a subject — shows how student performed across exams in every subject.")
        else:
            st.info("Not enough exam-level marks to draw trends.")

        st.markdown("---")

        # Attendance by month (bar)
        st.subheader("Attendance by month")
        if not s_att.empty and "Date" in s_att.columns:
            s_att2 = s_att.copy()
            s_att2["month"] = s_att2["Date"].dt.to_period("M").astype(str)
            monthly = s_att2.groupby("month")["_present_flag_"].mean().reset_index()
            fig_attm = px.bar(monthly, x="month", y="_present_flag_", title="Monthly attendance")
            fig_attm.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig_attm, use_container_width=True)
            with st.expander("Explanation", expanded=auto_expand):
                st.write("Monthly attendance percentage for the selected student — more interpretable than day-level dots.")
        else:
            st.info("No attendance records for this student.")

# ===== Tab: Compare Students =====
with tabs[2]:
    st.header("Compare Students")
    candidate_names = sorted(set(marks_df["Name"].dropna().tolist())) if ("Name" in marks_df.columns and not marks_df.empty) else []
    selected = st.multiselect("Select students (up to 6)", options=candidate_names, max_selections=6)
    exam_choice = st.selectbox("Exam filter (All for averages)", options=["All"] + ([str(e) for e in sorted(marks_df["ExamNumber"].dropna().unique().tolist())] if ("ExamNumber" in marks_df.columns and not marks_df.empty) else []))

    if not selected or len(selected) < 2:
        st.info("Select two or more students to compare (use ctrl/cmd to multi-select).")
    else:
        comp = marks_df[marks_df["Name"].isin(selected)].copy()
        if exam_choice != "All":
            comp = comp[comp["ExamNumber"].astype(str) == exam_choice]

        st.subheader("Subject-wise comparison")
        comp_avg = comp.groupby(["Name","Subject"])["Marks"].mean().reset_index()
        if not comp_avg.empty:
            fig_cmp = px.bar(comp_avg, x="Subject", y="Marks", color="Name", barmode="group")
            st.plotly_chart(fig_cmp, use_container_width=True)
            with st.expander("Explanation", expanded=auto_expand):
                st.write("Compares subject averages across the selected students. Helpful to see who is stronger in which subject.")
        else:
            st.info("No subject averages for the selected students/exam.")

        st.markdown("---")
        st.subheader("Exam-wise trend comparison")
        if "ExamNumber" in marks_df.columns:
            trend_df = marks_df[marks_df["Name"].isin(selected)].groupby(["ExamNumber","Name"])["Marks"].mean().reset_index()
            if not trend_df.empty:
                fig_trend = px.line(trend_df, x="ExamNumber", y="Marks", color="Name", markers=True)
                fig_trend.update_layout(yaxis_title="Avg marks")
                st.plotly_chart(fig_trend, use_container_width=True)
                with st.expander("Explanation", expanded=auto_expand):
                    st.write("Shows changes in average marks across exams for the selected students — useful to spot improvement or decline.")
            else:
                st.info("Insufficient exam data.")
        else:
            st.info("No ExamNumber column available.")

# ===== Tab: Attendance =====
with tabs[3]:
    st.header("Attendance Exploration")

    if att_df.empty:
        st.info("No attendance data.")
    else:
        # date range selector inside tab (defaults from sidebar)
        min_d = att_df["Date"].min().date() if ("Date" in att_df.columns and not att_df["Date"].isna().all()) else date.today()
        max_d = att_df["Date"].max().date() if ("Date" in att_df.columns and not att_df["Date"].isna().all()) else date.today()
        dr = st.date_input("Select date range", value=(min_d, max_d))
        try:
            sd, ed = (dr[0], dr[1]) if isinstance(dr, (list,tuple)) and len(dr)==2 else (dr, dr)
        except Exception:
            sd, ed = min_d, max_d

        if "Date" in att_df.columns:
            mask = (att_df["Date"].dt.date >= sd) & (att_df["Date"].dt.date <= ed)
            att_filtered = att_df[mask]
        else:
            att_filtered = att_df.copy()

        st.subheader("Daily class attendance %")
        if not att_filtered.empty and "_present_flag_" in att_filtered.columns:
            att_trend = att_filtered.groupby(att_filtered["Date"].dt.date)["_present_flag_"].mean().reset_index().rename(columns={"_present_flag_":"attendance_rate"})
            fig = px.line(att_trend, x="Date", y="attendance_rate", title="Daily attendance %", markers=True)
            fig.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)
            with st.expander("Explanation", expanded=auto_expand):
                st.write("Daily attendance percentage across the selected date range. Hover to inspect specific days.")
        else:
            st.info("No daily attendance records for the selected range.")

        st.markdown("---")
        st.subheader("Attendance by weekday")
        try:
            att_filtered["weekday"] = att_filtered["Date"].dt.day_name()
            weekday_avg = att_filtered.groupby("weekday")["_present_flag_"].mean().reindex(["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]).reset_index().dropna()
            weekday_avg.columns = ["weekday","attendance_rate"]
            fig_w = px.bar(weekday_avg, x="weekday", y="attendance_rate", title="Average attendance by weekday")
            fig_w.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig_w, use_container_width=True)
            with st.expander("Explanation", expanded=auto_expand):
                st.write("Shows which weekdays have consistently lower attendance — useful when scheduling important lessons.")
        except Exception:
            st.info("Unable to compute weekday breakdown for your dataset.")

        st.markdown("---")
        st.subheader("Attendance leaderboard")
        n = st.number_input("Show top N students by attendance", min_value=5, max_value=200, value=20)
        if not att_filtered.empty:
            leader = att_filtered.groupby("Name")["_present_flag_"].mean().reset_index().sort_values("_present_flag_", ascending=False).reset_index(drop=True)
            leader["attendance_pct"] = leader["_present_flag_"].apply(lambda x: f"{x*100:.1f}%")
            st.dataframe(leader[["Name","attendance_pct"]].head(n).set_index("Name"))
            with st.expander("Explanation", expanded=auto_expand):
                st.write("Top students by attendance — use to reward or set examples.")
        else:
            st.info("No attendance summary available.")

# ===== Tab: Marks =====
with tabs[4]:
    st.header("Marks Exploration")

    # distribution (hist already in class overview, add boxplot & top/bottom lists)
    st.subheader("Marks boxplot by subject")
    if not marks_df.empty and "Marks" in marks_df.columns and "Subject" in marks_df.columns:
        box_df = marks_df.copy()
        if subject_filter:
            box_df = box_df[box_df["Subject"].isin(subject_filter)]
        fig_box = px.box(box_df, x="Subject", y="Marks", color="Subject", color_discrete_map=SUBJECT_COLORS, title="Distribution by subject (boxplot)")
        st.plotly_chart(fig_box, use_container_width=True)
        with st.expander("Explanation", expanded=auto_expand):
            st.write("Boxplots show median, quartiles, and outliers per subject — good for spotting variability and extreme scores.")
    else:
        st.info("Insufficient marks data to show subject boxplots.")

    st.markdown("---")
    st.subheader("Top & bottom performers (configurable)")
    if not marks_df.empty and "Name" in marks_df.columns:
        k = st.slider("How many top/bottom students", min_value=1, max_value=30, value=5)
        avg_by_name = marks_df.groupby("Name")["Marks"].mean().reset_index().dropna().sort_values("Marks", ascending=False)
        topk = avg_by_name.head(k)
        botk = avg_by_name.tail(k).sort_values("Marks")
        fig_top = px.bar(topk, x="Name", y="Marks", title=f"Top {k} students (avg marks)", color_discrete_sequence=[PALETTE[1]])
        fig_bot = px.bar(botk, x="Name", y="Marks", title=f"Bottom {k} students (avg marks)", color_discrete_sequence=[PALETTE[3]])
        st.plotly_chart(fig_top, use_container_width=True)
        st.plotly_chart(fig_bot, use_container_width=True)
        with st.expander("Explanation", expanded=auto_expand):
            st.write("Top and bottom lists (by average marks) help identify students for recognition or intervention.")
    else:
        st.info("No marks data to compute performers.")

# ===== Tab: Insights =====
with tabs[5]:
    st.header("Curated Insights")

    # 1) Who to intervene (low attendance & low score)
    st.subheader("Students needing attention (attendance + average score)")
    if not marks_df.empty and not att_df.empty and "Name" in marks_df.columns and "Name" in att_df.columns:
        marks_avg = marks_df.groupby("Name")["Marks"].mean().reset_index()
        att_avg = att_df.groupby("Name")["_present_flag_"].mean().reset_index()
        merge = pd.merge(marks_avg, att_avg, on="Name", how="inner").rename(columns={"Marks":"avg_marks","_present_flag_":"att_rate"})
        if name_search:
            merge = merge[merge["Name"].str.contains(name_search, case=False, na=False)]
        # flagged: low marks and low attendance
        flagged = merge[(merge["avg_marks"] < flag_score_threshold) | (merge["att_rate"] < (flag_att_threshold_pct/100.0))]
        if not flagged.empty:
            flagged = flagged.sort_values(["att_rate","avg_marks"])
            flagged["attendance_pct"] = flagged["att_rate"].apply(lambda x: f"{x*100:.1f}%")
            st.dataframe(flagged[["Name","avg_marks","attendance_pct"]].rename(columns={"avg_marks":"Avg Marks"}).set_index("Name"))
            with st.expander("Explanation", expanded=auto_expand):
                st.write("These students are flagged either for low average marks or low attendance. Prioritize them for follow-up.")
        else:
            st.success("No students flagged with current thresholds.")
    else:
        st.info("Need both marks and attendance data to compute attention list.")

    st.markdown("---")

    # 2) Class-level quick bullet insights (auto-generated)
    st.subheader("Auto quick insights (class-level)")
    bullets = []
    if not marks_df.empty and "Marks" in marks_df.columns:
        overall_avg = marks_df["Marks"].mean()
        overall_median = marks_df["Marks"].median()
        bullets.append(f"Class average mark: {overall_avg:.1f}")
        bullets.append(f"Class median mark: {overall_median:.1f}")
        high_var = marks_df.groupby("Subject")["Marks"].std().reset_index().sort_values("Marks", ascending=False)
        if not high_var.empty:
            top_var = high_var.iloc[0]
            bullets.append(f"Highest variability appears in subject: {top_var['Subject']} (std ≈ {top_var['Marks']:.1f})")
    if not att_df.empty and "_present_flag_" in att_df.columns:
        avg_att_overall = att_df["_present_flag_"].mean()
        bullets.append(f"Average attendance overall: {avg_att_overall*100:.1f}%")
    if not bullets:
        st.info("Not enough data to generate quick insights.")
    else:
        for b in bullets:
            st.write("•", b)

    st.markdown("---")
    st.subheader("Action suggestions (based on data)")
    st.write(
        "• Prioritize students in the red bucket (low attendance + low average marks).  \n"
        "• For subjects with low pass rates, schedule targeted remedial lessons and formative quizzes.  \n"
        "• Recognize top attendance students publicly to encourage regularity.  \n"
        "• Use monthly attendance trends to avoid scheduling important exams on low-attendance days."
    )

st.caption("Right iTech — safe colors, more visualizations, broader explanations. Tell me any single small tweak (e.g., change palette, move a plot) and I will update only that portion.")
