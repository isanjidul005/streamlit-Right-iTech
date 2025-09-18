# Right_iTech.py
# Streamlit student analytics dashboard (clean inline comments only)

import os
from datetime import date
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Page setup
st.set_page_config(page_title="Right iTech", layout="wide", initial_sidebar_state="expanded")
px.defaults.template = "plotly_white"

# Color palettes
DISTINCT_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#d62728",
    "#17becf", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#393b79"
]
PRESENT_COLOR, ABSENT_COLOR, NEUTRAL = "#2ca02c", "#d62728", "#4a4a4a"

# Sidebar upload widgets
st.sidebar.header("Upload data & preferences")
att_upload = st.sidebar.file_uploader("Attendance CSV (optional)", type=["csv"])
marks_upload = st.sidebar.file_uploader("Marks CSV (optional)", type=["csv"])

# Fallback paths
FALLBACK_ATT = "/mnt/data/combined_attendance.csv"
FALLBACK_MARKS = "/mnt/data/cleanest_marks.csv"

@st.cache_data
def safe_read_csv(uploaded_file, fallback_path):
    """Read CSV from upload or fallback, return DataFrame."""
    try:
        if uploaded_file is not None:
            return pd.read_csv(uploaded_file)
        if fallback_path and os.path.exists(fallback_path):
            return pd.read_csv(fallback_path)
    except Exception:
        return pd.DataFrame()
    return pd.DataFrame()

# Load data
att_df = safe_read_csv(att_upload, FALLBACK_ATT)
marks_df = safe_read_csv(marks_upload, FALLBACK_MARKS)

# Sidebar controls
st.sidebar.markdown("---")
auto_expand = st.sidebar.checkbox("Auto-expand explanations", value=False)
pass_threshold = st.sidebar.number_input("Pass threshold (marks)", 0, 100, 40)
flag_score_threshold = st.sidebar.number_input("Flag avg score < (score)", 0, 100, 40)
flag_att_threshold_pct = st.sidebar.slider("Flag attendance < (%)", 0, 100, 75)

# App title
st.markdown("<h1 style='text-align:center; color:#1f77b4; margin-bottom:4px;'>Right iTech</h1>", unsafe_allow_html=True)
st.write("---")

# Ensure DataFrames exist
if att_df is None: att_df = pd.DataFrame()
if marks_df is None: marks_df = pd.DataFrame()

# Clean column names
if not att_df.empty: att_df.columns = [c.strip() for c in att_df.columns]
if not marks_df.empty: marks_df.columns = [c.strip() for c in marks_df.columns]

# Parse dates in attendance
if not att_df.empty and "Date" in att_df.columns:
    att_df["Date"] = pd.to_datetime(att_df["Date"], dayfirst=True, errors="coerce")

# Add present/absent flag in attendance
if not att_df.empty:
    if "Status" in att_df.columns:
        att_df["_present_flag_"] = att_df["Status"].astype(str).str.upper().map({
            "P":1,"PRESENT":1,"1":1,"Y":1,"YES":1,
            "A":0,"ABSENT":0,"0":0,"N":0,"NO":0
        })
        att_df["_present_flag_"] = att_df["_present_flag_"].fillna(
            att_df["Status"].astype(str).str[0].map({"P":1,"A":0})
        )
    elif "Attendance" in att_df.columns:
        att_df["_present_flag_"] = att_df["Attendance"].astype(str).str.upper().map({
            "PRESENT":1,"ABSENT":0,"P":1,"A":0
        })
    else:
        att_df["_present_flag_"] = np.nan
else:
    att_df["_present_flag_"] = pd.Series(dtype=float)

# Marks cleanup
if not marks_df.empty:
    if "Marks" in marks_df.columns:
        marks_df["Marks"] = pd.to_numeric(marks_df["Marks"], errors="coerce")
    else:
        marks_df["Marks"] = np.nan
    if "FullMarks" in marks_df.columns:
        marks_df["FullMarks"] = pd.to_numeric(marks_df["FullMarks"], errors="coerce")
else:
    marks_df = pd.DataFrame(columns=["ID","Roll","Name","Subject","ExamNumber","Exam","ExamType","Marks","FullMarks"])

# Assign subject colors dynamically
def assign_subject_colors(subjects):
    subs = sorted([s for s in subjects if pd.notna(s)])
    return {s: DISTINCT_PALETTE[i % len(DISTINCT_PALETTE)] for i,s in enumerate(subs)}

SUBJECT_COLORS = assign_subject_colors(marks_df["Subject"].unique()) if ("Subject" in marks_df.columns and not marks_df.empty) else {}

# Count boys and girls safely
def safe_count_gender(df):
    if df is None or df.empty: return 0, 0
    if "ID" in df.columns:
        uniq = df.drop_duplicates(subset=["ID"]).copy()
    elif "Name" in df.columns:
        uniq = df.drop_duplicates(subset=["Name"]).copy()
    else:
        return 0, 0
    if "Gender" not in uniq.columns: return 0, 0

    g = uniq["Gender"].astype(str).str.strip().str.lower().fillna("")
    male_mask = g.str.match(r'^(m|male|boy|man)\b', na=False)
    female_mask = g.str.match(r'^(f|female|girl|woman)\b', na=False)
    boys, girls = int(male_mask.sum()), int(female_mask.sum())

    if boys == 0 and girls == 0:  
        boys = int(g.str.contains(r'male|^m\b|boy', na=False).sum())
        girls = int(g.str.contains(r'female|^f\b|girl', na=False).sum())
    return boys, girls

def ensure_list(x): 
    return x if isinstance(x, list) else [x]

# Stop if no data
if att_df.empty and marks_df.empty:
    st.warning("No data detected. Upload Attendance and Marks CSVs in the sidebar or place fallback files at /mnt/data/*.csv")
    st.stop()

# Sidebar global filters
st.sidebar.header("Global filters")
if not att_df.empty and "Date" in att_df.columns and not att_df["Date"].isna().all():
    min_date, max_date = att_df["Date"].min().date(), att_df["Date"].max().date()
else:
    min_date = max_date = date.today()

date_range = st.sidebar.date_input("Attendance date range", value=(min_date, max_date))
subject_options = sorted(marks_df["Subject"].dropna().unique().tolist()) if ("Subject" in marks_df.columns and not marks_df.empty) else []
subject_filter = st.sidebar.multiselect("Filter subjects (global)", options=subject_options, default=subject_options)
exam_options = sorted(marks_df["ExamNumber"].dropna().unique().tolist()) if ("ExamNumber" in marks_df.columns and not marks_df.empty) else []
exam_filter = st.sidebar.multiselect("Filter exams (global)", options=exam_options, default=exam_options)
name_search = st.sidebar.text_input("Quick search student name (partial)")

# Tabs
tabs = st.tabs(["Class Overview","Student Dashboard","Compare Students","Attendance","Marks","Insights"])

# === Class Overview ===
with tabs[0]:
    st.header("Class Overview")

    # Total students
    total_students = 0
    if "ID" in marks_df.columns and not marks_df.empty:
        total_students = marks_df["ID"].nunique()
    elif "Name" in marks_df.columns and not marks_df.empty:
        total_students = marks_df["Name"].nunique()
    elif "ID" in att_df.columns and not att_df.empty:
        total_students = att_df["ID"].nunique()
    elif "Name" in att_df.columns and not att_df.empty:
        total_students = att_df["Name"].nunique()

    # Gender counts
    boys, girls = (safe_count_gender(att_df) if not att_df.empty else (0,0))
    if boys==0 and girls==0 and not marks_df.empty: 
        boys,girls = safe_count_gender(marks_df)

    # Avg attendance
    avg_att = att_df["_present_flag_"].mean() if not att_df.empty and "_present_flag_" in att_df.columns else np.nan

    # Metric row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total students", total_students)
    col2.metric("Boys", int(boys) if boys is not None else "N/A")
    col3.metric("Girls", int(girls) if girls is not None else "N/A")
    col4.metric("Avg attendance", f"{avg_att*100:.1f}%" if not np.isnan(avg_att) else "N/A")

    # Bar chart of average marks by subject
    if not marks_df.empty:
        avg_by_subject = marks_df.groupby("Subject")["Marks"].mean().reset_index()
        fig = px.bar(
            avg_by_subject, x="Subject", y="Marks", 
            title="Average Marks by Subject",
            color="Subject", color_discrete_map=SUBJECT_COLORS
        )
        st.plotly_chart(fig, use_container_width=True)

# === Student Dashboard ===
with tabs[1]:
    st.header("Student Dashboard")
    st.info("Details per student (filterable).")

# === Compare Students ===
with tabs[2]:
    st.header("Compare Students")
    st.info("Side-by-side comparison of students.")

# === Attendance ===
with tabs[3]:
    st.header("Attendance")
    st.info("Attendance visualizations and breakdowns.")

# === Marks ===
with tabs[4]:
    st.header("Marks")
    st.info("Exam and subject performance charts.")

# === Insights ===
with tabs[5]:
    st.header("Insights")
    st.info("Key takeaways and actionable insights.")
