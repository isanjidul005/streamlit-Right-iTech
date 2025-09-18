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

# -------------------------
# Page setup
# -------------------------
st.set_page_config(page_title="Right iTech", layout="wide", initial_sidebar_state="expanded")
pio.templates.default = "plotly_white"

# -------------------------
# Helpful colors & palette (calm, professional)
# -------------------------
PALETTE = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#17becf"]
ATT_PRESENT_COLOR = "#2ca02c"
ATT_ABSENT_COLOR = "#d62728"

# -------------------------
# Sidebar: upload + settings
# -------------------------
st.sidebar.header("Upload data & settings")
uploaded_att = st.sidebar.file_uploader("Attendance CSV (optional)", type=["csv"]) 
uploaded_marks = st.sidebar.file_uploader("Marks CSV (optional)", type=["csv"]) 

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
auto_expand_explanations = st.sidebar.checkbox("Auto-expand explanations", value=False)
pass_threshold = st.sidebar.number_input("Pass threshold (marks)", min_value=0, max_value=100, value=40)
flag_score_threshold = st.sidebar.number_input("Flag if avg score < (score)", min_value=0, max_value=100, value=40)
flag_att_threshold_pct = st.sidebar.slider("Flag if attendance < (%)", 0, 100, 75)

# -------------------------
# Basic safety & cleaning
# -------------------------
if att_df is None: att_df = pd.DataFrame()
if marks_df is None: marks_df = pd.DataFrame()

# normalize column names
if not att_df.empty:
    att_df.columns = [c.strip() for c in att_df.columns]
if not marks_df.empty:
    marks_df.columns = [c.strip() for c in marks_df.columns]

# safe parse dates
if not att_df.empty and "Date" in att_df.columns:
    att_df["Date"] = pd.to_datetime(att_df["Date"], dayfirst=True, errors="coerce")

# unify present flag
if not att_df.empty:
    if "Status" in att_df.columns:
        att_df["_present_flag_"] = att_df["Status"].astype(str).str.upper().map({
            "P":1, "PRESENT":1, "1":1, "Y":1, "YES":1,
            "A":0, "ABSENT":0, "0":0, "N":0, "NO":0
        })
        att_df["_present_flag_"] = att_df["_present_flag_"].fillna(att_df["Status"].astype(str).str[0].map({"P":1,"A":0}))
    elif "Attendance" in att_df.columns:
        att_df["_present_flag_"] = att_df["Attendance"].astype(str).str.upper().map({"PRESENT":1,"ABSENT":0,"P":1,"A":0})
    else:
        att_df["_present_flag_"] = np.nan
else:
    att_df["_present_flag_"] = pd.Series(dtype=float)

# ensure marks numeric
if not marks_df.empty:
    if "Marks" in marks_df.columns:
        marks_df["Marks"] = pd.to_numeric(marks_df["Marks"], errors="coerce")
    else:
        marks_df["Marks"] = np.nan
    if "FullMarks" in marks_df.columns:
        marks_df["FullMarks"] = pd.to_numeric(marks_df["FullMarks"], errors="coerce")
else:
    marks_df = pd.DataFrame(columns=["ID","Roll","Name","Subject","ExamNumber","Exam","ExamType","Marks","FullMarks"]) 

# subject colors
def assign_subject_colors(subjects):
    subs = sorted([s for s in subjects if pd.notna(s)])
    mapping = {}
    for i,s in enumerate(subs):
        mapping[s] = PALETTE[i % len(PALETTE)]
    return mapping

SUBJECT_COLORS = assign_subject_colors(marks_df["Subject"].unique()) if ("Subject" in marks_df.columns and not marks_df.empty) else {}

# -------------------------
# UI header styles (professional, readable)
# -------------------------
st.markdown("""
<style>
.header {font-family: 'Segoe UI', Tahoma, sans-serif; color: #0a2540;}
.card {background:#ffffff; padding:16px; border-radius:8px; box-shadow: rgba(0, 0, 0, 0.08) 0px 2px 8px;}
.small {color:#444; font-size:0.9rem;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class='card'>
  <h1 class='header'>Right iTech</h1>
  <div class='small'>Marks & attendance dashboard with deeper insights</div>
</div>
""", unsafe_allow_html=True)

st.write("---")

# -------------------------
# Stop if no data
# -------------------------
if att_df.empty and marks_df.empty:
    st.error("No data loaded. Upload files in the sidebar or place fallback CSVs at /mnt/data/*.csv")
    st.stop()

# -------------------------
# Tabs (simplified UI)
# -------------------------
tabs = st.tabs(["Class Overview","Student Report","Compare Students","Attendance","Insights"]) 

# ===== Tab 0: Class overview =====
with tabs[0]:
    st.header("Class overview")

    col1, col2, col3 = st.columns(3)
    ids = set()
    if not marks_df.empty and "ID" in marks_df.columns:
        ids.update(marks_df["ID"].dropna().astype(str).tolist())
    if not att_df.empty and "ID" in att_df.columns:
        ids.update(att_df["ID"].dropna().astype(str).tolist())
    total_students = len(ids)
    col1.metric("Total students", total_students)

    avg_att = att_df["_present_flag_"].mean() if not att_df.empty else np.nan
    col2.metric("Avg attendance", f"{avg_att*100:.1f}%" if not np.isnan(avg_att) else "N/A")

    avg_marks = marks_df["Marks"].mean() if not marks_df.empty else np.nan
    col3.metric("Avg marks", f"{avg_marks:.1f}" if not np.isnan(avg_marks) else "N/A")

    st.markdown("---")
    st.subheader("Distribution of marks")
    if not marks_df.empty:
        fig_hist = px.histogram(marks_df, x="Marks", nbins=20, color_discrete_sequence=["#1f77b4"])
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.info("Marks data not available.")

    st.markdown("---")
    st.subheader("Attendance distribution")
    if not att_df.empty and "Name" in att_df.columns:
        att_rates = att_df.groupby("Name")["_present_flag_"].mean().reset_index()
        fig_att = px.histogram(att_rates, x="_present_flag_", nbins=10, color_discrete_sequence=["#2ca02c"])
        fig_att.update_xaxes(title="Attendance rate")
        st.plotly_chart(fig_att, use_container_width=True)
    else:
        st.info("Attendance data not available.")

# ===== Tab 1: Individual Student Report =====
with tabs[1]:
    st.header("Individual Student Report")
    name_candidates = sorted(marks_df["Name"].dropna().unique().tolist()) if ("Name" in marks_df.columns and not marks_df.empty) else []

    if not name_candidates:
        st.info("No student names found in data.")
    else:
        student = st.selectbox("Select student", name_candidates)
        s_marks = marks_df[marks_df["Name"]==student] if not marks_df.empty else pd.DataFrame()
        s_att = att_df[att_df["Name"]==student] if not att_df.empty else pd.DataFrame()

        st.subheader("Subject performance")
        if not s_marks.empty:
            subj_avg = s_marks.groupby("Subject")["Marks"].mean().reset_index()
            fig = px.bar(subj_avg, x="Subject", y="Marks", color="Subject", color_discrete_map=SUBJECT_COLORS)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No marks data for this student.")

        st.subheader("Attendance trend")
        if not s_att.empty and "Date" in s_att.columns:
            trend = s_att.groupby(s_att["Date"].dt.date)["_present_flag_"].mean().reset_index()
            fig_trend = px.line(trend, x="Date", y="_present_flag_", markers=True)
            fig_trend.update_yaxes(tickformat='.0%')
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.info("No attendance data for this student.")

# ===== Tab 2: Compare Students =====
with tabs[2]:
    st.header("Compare Students")
    candidate_names = sorted(set(marks_df["Name"].dropna().tolist())) if ("Name" in marks_df.columns and not marks_df.empty) else []
    selection = st.multiselect("Select students", options=candidate_names)

    if len(selection) >= 2:
        comp = marks_df[marks_df["Name"].isin(selection)]
        subj_avg = comp.groupby(["Name","Subject"])["Marks"].mean().reset_index()
        fig = px.bar(subj_avg, x="Subject", y="Marks", color="Name", barmode="group")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Select two or more students to compare.")

# ===== Tab 3: Attendance explorer =====
with tabs[3]:
    st.header("Attendance Explorer")
    if not att_df.empty:
        trend = att_df.groupby(att_df["Date"].dt.date)["_present_flag_"].mean().reset_index()
        fig = px.line(trend, x="Date", y="_present_flag_", markers=True)
        fig.update_yaxes(tickformat='.0%')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No attendance data available.")

# ===== Tab 4: Insights =====
with tabs[4]:
    st.header("Insights")

    if not marks_df.empty:
        top_subjects = marks_df.groupby("Subject")["Marks"].mean().reset_index().sort_values("Marks", ascending=False)
        fig_top = px.bar(top_subjects, x="Subject", y="Marks", color="Subject", color_discrete_map=SUBJECT_COLORS)
        st.subheader("Average marks by subject")
        st.plotly_chart(fig_top, use_container_width=True)

    if not att_df.empty:
        avg_att_by_student = att_df.groupby("Name")["_present_flag_"].mean().reset_index()
        st.subheader("Lowest attendance students")
        low_att = avg_att_by_student.sort_values("_present_flag_").head(10)
        st.table(low_att)

st.caption("Right iTech — polished insights. Colors are professional, UI is simple and clean.")
