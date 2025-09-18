# Right_iTech_app.py
# Streamlit dashboard — clean, production-ready

import os
from datetime import date
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="Right iTech", layout="wide", initial_sidebar_state="expanded")
px.defaults.template = "plotly_white"

# -------------------------
# Distinct palette (used after subjects are known)
# -------------------------
DISTINCT_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#d62728",
    "#17becf", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#393b79"
]
PRESENT_COLOR = "#2ca02c"
ABSENT_COLOR = "#d62728"

# -------------------------
# Sidebar: uploads + controls
# -------------------------
st.sidebar.header("Upload data")

att_upload = st.sidebar.file_uploader("Attendance CSV", type=["csv"])
marks_upload = st.sidebar.file_uploader("Marks CSV", type=["csv"])

# fallback paths (if server has default files)
FALLBACK_ATT = "/mnt/data/combined_attendance.csv"
FALLBACK_MARKS = "/mnt/data/cleanest_marks.csv"

@st.cache_data
def safe_read_csv(uploaded_file, fallback_path):
    try:
        if uploaded_file is not None:
            return pd.read_csv(uploaded_file)
        if os.path.exists(fallback_path):
            return pd.read_csv(fallback_path)
    except Exception:
        return pd.DataFrame()
    return pd.DataFrame()

att_df = safe_read_csv(att_upload, FALLBACK_ATT)
marks_df = safe_read_csv(marks_upload, FALLBACK_MARKS)

# Sidebar preferences
st.sidebar.markdown("---")
auto_expand = st.sidebar.checkbox("Auto-expand explanations", value=False)
pass_threshold = st.sidebar.number_input("Pass threshold (marks)", 0, 100, 40)
flag_score_threshold = st.sidebar.number_input("Flag if avg score < (score)", 0, 100, 40)
flag_att_threshold_pct = st.sidebar.slider("Flag if attendance < (%)", 0, 100, 75)

# -------------------------
# Title
# -------------------------
st.markdown("<h1 style='text-align:center; color:#1f77b4;'>Right iTech</h1>", unsafe_allow_html=True)
st.write("---")

# -------------------------
# Normalisation & cleaning
# -------------------------
if not att_df.empty:
    att_df.columns = [c.strip() for c in att_df.columns]
    if "Date" in att_df.columns:
        att_df["Date"] = pd.to_datetime(att_df["Date"], dayfirst=True, errors="coerce")
    if "Status" in att_df.columns:
        att_df["_present_flag_"] = att_df["Status"].astype(str).str.upper().map({
            "P":1,"PRESENT":1,"1":1,"Y":1,"YES":1,
            "A":0,"ABSENT":0,"0":0,"N":0,"NO":0
        })
    else:
        att_df["_present_flag_"] = np.nan

if not marks_df.empty:
    marks_df.columns = [c.strip() for c in marks_df.columns]
    if "Marks" in marks_df.columns:
        marks_df["Marks"] = pd.to_numeric(marks_df["Marks"], errors="coerce")
    if "FullMarks" in marks_df.columns:
        marks_df["FullMarks"] = pd.to_numeric(marks_df["FullMarks"], errors="coerce")

# assign subject colors dynamically
def assign_subject_colors(subjects):
    subs = sorted([s for s in subjects if pd.notna(s)])
    mapping = {}
    for i,s in enumerate(subs):
        mapping[s] = DISTINCT_PALETTE[i % len(DISTINCT_PALETTE)]
    return mapping

SUBJECT_COLORS = assign_subject_colors(marks_df["Subject"].unique()) if ("Subject" in marks_df.columns and not marks_df.empty) else {}

# -------------------------
# Helpers
# -------------------------
def safe_count_gender(df):
    """Count boys/girls robustly from Gender column if available"""
    if df.empty or "Gender" not in df.columns:
        return 0,0
    g = df["Gender"].astype(str).str.strip().str.lower().fillna("")
    boys = int(g.str.contains(r"^m|male|boy").sum())
    girls = int(g.str.contains(r"^f|female|girl").sum())
    return boys,girls

# -------------------------
# Stop if no data at all
# -------------------------
if att_df.empty and marks_df.empty:
    st.warning("No data detected. Upload Attendance and Marks CSVs in the sidebar.")
    st.stop()

# -------------------------
# Global filters
# -------------------------
st.sidebar.header("Global filters")
if not att_df.empty and "Date" in att_df.columns:
    min_date, max_date = att_df["Date"].min().date(), att_df["Date"].max().date()
else:
    min_date, max_date = date.today(), date.today()

date_range = st.sidebar.date_input("Attendance date range", value=(min_date, max_date))
subject_options = sorted(marks_df["Subject"].dropna().unique()) if "Subject" in marks_df.columns else []
subject_filter = st.sidebar.multiselect("Subjects", subject_options, default=subject_options)
exam_options = sorted(marks_df["ExamNumber"].dropna().unique()) if "ExamNumber" in marks_df.columns else []
exam_filter = st.sidebar.multiselect("Exams", exam_options, default=exam_options)
name_search = st.sidebar.text_input("Search student name")

# -------------------------
# Tabs
# -------------------------
tabs = st.tabs(["Class Overview","Student Dashboard","Compare Students","Attendance","Marks","Insights"])

# -------------------------
# Class Overview
# -------------------------
with tabs[0]:
    st.header("Class Overview")

    # total students
    total_students = marks_df["Name"].nunique() if not marks_df.empty else att_df["Name"].nunique()
    boys,girls = safe_count_gender(att_df if not att_df.empty else marks_df)

    avg_att = att_df["_present_flag_"].mean() if "_present_flag_" in att_df.columns else np.nan

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total students", total_students)
    c2.metric("Boys", boys)
    c3.metric("Girls", girls)
    c4.metric("Avg attendance", f"{avg_att*100:.1f}%" if not np.isnan(avg_att) else "N/A")

    st.markdown("---")

    # subject averages
    st.subheader("Average marks by subject")
    if not marks_df.empty:
        dfm = marks_df.copy()
        if subject_filter: dfm = dfm[dfm["Subject"].isin(subject_filter)]
        if exam_filter: dfm = dfm[dfm["ExamNumber"].isin(exam_filter)]
        if name_search: dfm = dfm[dfm["Name"].str.contains(name_search, case=False, na=False)]
        subj_avg = dfm.groupby("Subject")["Marks"].mean().reset_index()
        if not subj_avg.empty:
            fig = px.bar(subj_avg, x="Subject", y="Marks", color="Subject", color_discrete_map=SUBJECT_COLORS)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # threshold breakdown
    st.subheader("Students ≥ threshold vs < threshold")
    if not marks_df.empty:
        per_stu_sub = marks_df.groupby(["Subject","Name"])["Marks"].mean().reset_index()
        subj_counts = per_stu_sub.groupby("Subject").apply(
            lambda g: pd.Series({
                "n_total": g["Name"].nunique(),
                "n_above": (g["Marks"]>=pass_threshold).sum()
            })
        ).reset_index()
        subj_counts["n_below"] = subj_counts["n_total"] - subj_counts["n_above"]

        long = []
        for _,r in subj_counts.iterrows():
            long.append({"Subject":r["Subject"],"Category":f"≥{pass_threshold}","Value":r["n_above"]})
            long.append({"Subject":r["Subject"],"Category":f"<{pass_threshold}","Value":r["n_below"]})
        df_long = pd.DataFrame(long)
        fig = px.bar(df_long, x="Subject", y="Value", color="Category", barmode="stack")
        st.plotly_chart(fig, use_container_width=True)

# -------------------------
# Student Dashboard
# -------------------------
with tabs[1]:
    st.header("Student Dashboard")
    students = sorted(marks_df["Name"].dropna().unique()) if "Name" in marks_df.columns else []
    if not students:
        st.info("No student names available.")
    else:
        stu = st.selectbox("Select student", students)
        s_marks = marks_df[marks_df["Name"]==stu]
        s_att = att_df[att_df["Name"]==stu]

        st.subheader("Performance by subject")
        subj_avg = s_marks.groupby("Subject")["Marks"].mean().reset_index()
        if not subj_avg.empty:
            fig = px.bar(subj_avg, x="Subject", y="Marks", color="Subject", color_discrete_map=SUBJECT_COLORS)
            st.plotly_chart(fig, use_container_width=True)

# -------------------------
# Compare Students
# -------------------------
with tabs[2]:
    st.header("Compare Students")
    candidates = sorted(marks_df["Name"].dropna().unique()) if "Name" in marks_df.columns else []
    chosen = st.multiselect("Select students", candidates, max_selections=6)

    if len(chosen)>=2:
        comp = marks_df[marks_df["Name"].isin(chosen)]

        # subject-wise averages
        st.subheader("Subject-wise averages")
        avg = comp.groupby(["Name","Subject"])["Marks"].mean().reset_index()
        fig = px.bar(avg, x="Subject", y="Marks", color="Name", barmode="group")
        st.plotly_chart(fig, use_container_width=True)

        # exam averages
        st.subheader("Average by exam")
        exam_avg = comp.groupby(["Name","ExamNumber"])["Marks"].mean().reset_index()
        fig2 = px.line(exam_avg, x="ExamNumber", y="Marks", color="Name", markers=True)
        st.plotly_chart(fig2, use_container_width=True)

# -------------------------
# Attendance
# -------------------------
with tabs[3]:
    st.header("Attendance")

    if not att_df.empty:
        # weekend toggle
        toggle = st.radio("Include weekends?", ["Include","Exclude"], horizontal=True)
        df_att = att_df.copy()
        if toggle=="Exclude":
            df_att = df_att[df_att["Date"].dt.day_name()!="Friday"]

        # daily trend
        st.subheader("Daily attendance %")
        daily = df_att.groupby(df_att["Date"].dt.date)["_present_flag_"].mean().reset_index()
        fig = px.line(daily, x="Date", y="_present_flag_", markers=True)
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

        # monthly bar
        st.subheader("Monthly attendance")
        df_att["month"] = df_att["Date"].dt.to_period("M").astype(str)
        monthly = df_att.groupby("month")["_present_flag_"].mean().reset_index()
        fig2 = px.bar(monthly, x="month", y="_present_flag_")
        fig2.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig2, use_container_width=True)

        # present vs absent pie
        st.subheader("Overall present vs absent")
        counts = df_att["_present_flag_"].value_counts().rename({1:"Present",0:"Absent"})
        fig3 = px.pie(values=counts.values, names=counts.index, color=counts.index,
                      color_discrete_map={"Present":PRESENT_COLOR,"Absent":ABSENT_COLOR})
        st.plotly_chart(fig3, use_container_width=True)

# -------------------------
# Marks
# -------------------------
with tabs[4]:
    st.header("Marks")
    if not marks_df.empty:
        fig = px.histogram(marks_df, x="Marks", nbins=20)
        st.plotly_chart(fig, use_container_width=True)

# -------------------------
# Insights
# -------------------------
with tabs[5]:
    st.header("Insights")
    if not marks_df.empty:
        st.write(f"Class average: {marks_df['Marks'].mean():.1f}")
    if not att_df.empty:
        st.write(f"Avg attendance: {att_df['_present_flag_'].mean()*100:.1f}%")
