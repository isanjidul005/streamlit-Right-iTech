# Right_iTech.py
import os
from datetime import date
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Right iTech", layout="wide", initial_sidebar_state="expanded")
px.defaults.template = "plotly_white"

# palette
DISTINCT_PALETTE = [
    "#1f77b4","#ff7f0e","#2ca02c","#9467bd","#d62728",
    "#17becf","#8c564b","#e377c2","#7f7f7f","#bcbd22","#393b79"
]
PRESENT_COLOR, ABSENT_COLOR = "#2ca02c","#d62728"

# sidebar inputs
st.sidebar.header("Upload data & preferences")
att_upload = st.sidebar.file_uploader("Attendance CSV (optional)", type=["csv"])
marks_upload = st.sidebar.file_uploader("Marks CSV (optional)", type=["csv"])
FALLBACK_ATT, FALLBACK_MARKS = "/mnt/data/combined_attendance.csv","/mnt/data/cleanest_marks.csv"

@st.cache_data
def safe_read_csv(uploaded_file, fallback_path):
    try:
        if uploaded_file is not None: return pd.read_csv(uploaded_file)
        if os.path.exists(fallback_path): return pd.read_csv(fallback_path)
    except Exception: return pd.DataFrame()
    return pd.DataFrame()

att_df = safe_read_csv(att_upload, FALLBACK_ATT)
marks_df = safe_read_csv(marks_upload, FALLBACK_MARKS)

# sidebar options
st.sidebar.markdown("---")
auto_expand = st.sidebar.checkbox("Auto-expand explanations", value=False)
pass_threshold = st.sidebar.number_input("Pass threshold", 0, 100, 40)
flag_score_threshold = st.sidebar.number_input("Flag if avg score <", 0, 100, 40)
flag_att_threshold_pct = st.sidebar.slider("Flag if attendance < (%)", 0, 100, 75)

# title
st.markdown("<h1 style='text-align:center; color:#1f77b4;'>Right iTech</h1>", unsafe_allow_html=True)
st.write("---")

# clean cols
if not att_df.empty: att_df.columns = [c.strip() for c in att_df.columns]
if not marks_df.empty: marks_df.columns = [c.strip() for c in marks_df.columns]

# date col
if "Date" in att_df.columns: att_df["Date"] = pd.to_datetime(att_df["Date"], dayfirst=True, errors="coerce")

# attendance flag
if not att_df.empty:
    if "Status" in att_df.columns:
        att_df["_present_flag_"] = att_df["Status"].astype(str).str.upper().map({
            "P":1,"PRESENT":1,"1":1,"Y":1,"YES":1,
            "A":0,"ABSENT":0,"0":0,"N":0,"NO":0})
    elif "Attendance" in att_df.columns:
        att_df["_present_flag_"] = att_df["Attendance"].astype(str).str.upper().map({"PRESENT":1,"ABSENT":0,"P":1,"A":0})
    else: att_df["_present_flag_"] = np.nan
else: att_df["_present_flag_"] = pd.Series(dtype=float)

# numeric marks
if not marks_df.empty:
    if "Marks" in marks_df.columns: marks_df["Marks"] = pd.to_numeric(marks_df["Marks"], errors="coerce")
    if "FullMarks" in marks_df.columns: marks_df["FullMarks"] = pd.to_numeric(marks_df["FullMarks"], errors="coerce")
else:
    marks_df = pd.DataFrame(columns=["ID","Roll","Name","Subject","ExamNumber","Exam","ExamType","Marks","FullMarks"])

# subject colors
def assign_subject_colors(subjects):
    subs = sorted([s for s in subjects if pd.notna(s)])
    return {s: DISTINCT_PALETTE[i % len(DISTINCT_PALETTE)] for i,s in enumerate(subs)}
SUBJECT_COLORS = assign_subject_colors(marks_df["Subject"].unique()) if "Subject" in marks_df.columns else {}

# gender counts
def safe_count_gender(df):
    if df.empty or "Gender" not in df.columns: return 0,0
    uniq = df.drop_duplicates(subset=["ID"]) if "ID" in df.columns else df.drop_duplicates(subset=["Name"])
    g = uniq["Gender"].astype(str).str.lower()
    boys = g.str.match(r'^(m|male|boy)').sum()
    girls = g.str.match(r'^(f|female|girl)').sum()
    return int(boys), int(girls)

# no data stop
if att_df.empty and marks_df.empty:
    st.warning("No data detected. Upload CSVs or place fallback files in /mnt/data/")
    st.stop()

# sidebar filters
st.sidebar.header("Global filters")
if "Date" in att_df.columns and not att_df["Date"].isna().all():
    min_date, max_date = att_df["Date"].min().date(), att_df["Date"].max().date()
else: min_date, max_date = date.today(), date.today()
date_range = st.sidebar.date_input("Attendance date range", (min_date, max_date))
subject_filter = st.sidebar.multiselect("Filter subjects", sorted(marks_df["Subject"].dropna().unique()) if "Subject" in marks_df else [])
exam_filter = st.sidebar.multiselect("Filter exams", sorted(marks_df["ExamNumber"].dropna().unique()) if "ExamNumber" in marks_df else [])
name_search = st.sidebar.text_input("Search student name")

# tabs
tabs = st.tabs(["Class Overview","Student Dashboard","Compare Students","Attendance","Marks","Insights"])

# --- Overview ---
with tabs[0]:
    st.header("Class Overview")
    if "ID" in marks_df: total_students = marks_df["ID"].nunique()
    elif "Name" in marks_df: total_students = marks_df["Name"].nunique()
    elif "ID" in att_df: total_students = att_df["ID"].nunique()
    else: total_students = att_df["Name"].nunique()
    boys,girls = safe_count_gender(att_df if not att_df.empty else marks_df)
    avg_att = att_df["_present_flag_"].mean() if "_present_flag_" in att_df else np.nan
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total students", total_students)
    c2.metric("Boys", boys)
    c3.metric("Girls", girls)
    c4.metric("Avg attendance", f"{avg_att*100:.1f}%" if not np.isnan(avg_att) else "N/A")
    st.subheader("Attendance trend")
    if "Date" in att_df and "_present_flag_" in att_df:
        sd, ed = date_range if isinstance(date_range,(list,tuple)) else (date_range,date_range)
        mask = (att_df["Date"].dt.date>=sd)&(att_df["Date"].dt.date<=ed)
        att_range = att_df[mask]
        if not att_range.empty:
            att_over_time = att_range.groupby(att_range["Date"].dt.date)["_present_flag_"].mean().reset_index()
            fig = px.line(att_over_time, x="Date", y="_present_flag_", markers=True)
            fig.update_yaxes(tickformat=".0%"); st.plotly_chart(fig,use_container_width=True)
    st.subheader("Average marks by subject")
    if "Marks" in marks_df and "Subject" in marks_df:
        dfm = marks_df.copy()
        if subject_filter: dfm = dfm[dfm["Subject"].isin(subject_filter)]
        if exam_filter: dfm = dfm[dfm["ExamNumber"].isin(exam_filter)]
        if name_search: dfm = dfm[dfm["Name"].str.contains(name_search,case=False,na=False)]
        subj_avg = dfm.groupby("Subject")["Marks"].mean().reset_index()
        if not subj_avg.empty:
            fig = px.bar(subj_avg, x="Subject", y="Marks", color="Subject", color_discrete_map=SUBJECT_COLORS)
            st.plotly_chart(fig,use_container_width=True)

# --- Student Dashboard ---
with tabs[1]:
    st.header("Student Dashboard")
    if not marks_df.empty:
        students = sorted(marks_df["Name"].dropna().unique())
        selected = st.selectbox("Select student", students)
        sdata = marks_df[marks_df["Name"]==selected]
        st.subheader(f"Performance: {selected}")
        if not sdata.empty:
            fig = px.line(sdata, x="ExamNumber", y="Marks", color="Subject",
                          color_discrete_map=SUBJECT_COLORS, markers=True)
            st.plotly_chart(fig,use_container_width=True)
            polar = sdata.groupby("Subject")["Marks"].mean().reset_index()
            if not polar.empty:
                polar = pd.concat([polar, polar.iloc[[0]]])
                fig = go.Figure(data=go.Scatterpolar(r=polar["Marks"], theta=polar["Subject"], fill="toself"))
                fig.update_layout(polar=dict(radialaxis=dict(visible=True,range=[0,100])))
                st.plotly_chart(fig,use_container_width=True)

# --- Compare Students ---
with tabs[2]:
    st.header("Compare Students")
    if not marks_df.empty:
        s_multi = st.multiselect("Select students", sorted(marks_df["Name"].dropna().unique()))
        if s_multi:
            dfm = marks_df[marks_df["Name"].isin(s_multi)]
            fig = px.line(dfm, x="ExamNumber", y="Marks", color="Name", line_group="Subject", markers=True)
            st.plotly_chart(fig,use_container_width=True)
            avg_sub = dfm.groupby(["Name","Subject"])["Marks"].mean().reset_index()
            fig = px.bar(avg_sub, x="Subject", y="Marks", color="Name", barmode="group")
            st.plotly_chart(fig,use_container_width=True)

# --- Attendance ---
with tabs[3]:
    st.header("Attendance")
    if not att_df.empty and "Date" in att_df:
        toggle_weekend = st.radio("Attendance view", ["Include weekends","Exclude weekends"], horizontal=True)
        att_df["_weekday_"] = att_df["Date"].dt.day_name()
        data = att_df.copy()
        if toggle_weekend=="Exclude weekends": data = data[data["_weekday_"]!="Friday"]
        daily = data.groupby("Date")["_present_flag_"].mean().reset_index()
        fig = px.line(daily, x="Date", y="_present_flag_", markers=True, title="Daily attendance rate")
        fig.update_yaxes(tickformat=".0%"); st.plotly_chart(fig,use_container_width=True)
        avg_by_day = data.groupby("_weekday_")["_present_flag_"].mean().reindex(
            ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
        avg_by_day = avg_by_day.reset_index().dropna()
        fig = px.bar(avg_by_day, x="_weekday_", y="_present_flag_", title="Avg attendance by weekday")
        fig.update_yaxes(tickformat=".0%"); st.plotly_chart(fig,use_container_width=True)

# --- Marks ---
with tabs[4]:
    st.header("Marks")
    if not marks_df.empty:
        if subject_filter: marks_df = marks_df[marks_df["Subject"].isin(subject_filter)]
        if exam_filter: marks_df = marks_df[marks_df["ExamNumber"].isin(exam_filter)]
        if name_search: marks_df = marks_df[marks_df["Name"].str.contains(name_search,case=False,na=False)]
        st.subheader("Distribution by subject")
        fig = px.box(marks_df, x="Subject", y="Marks", color="Subject", color_discrete_map=SUBJECT_COLORS)
        st.plotly_chart(fig,use_container_width=True)
        st.subheader("Marks vs Full marks")
        if "FullMarks" in marks_df.columns:
            fig = px.scatter(marks_df, x="FullMarks", y="Marks", color="Subject", hover_data=["Name"],
                             color_discrete_map=SUBJECT_COLORS)
            st.plotly_chart(fig,use_container_width=True)

# --- Insights ---
with tabs[5]:
    st.header("Insights")
    if not marks_df.empty:
        avg_marks = marks_df.groupby("Name")["Marks"].mean()
        low_perf = avg_marks[avg_marks<flag_score_threshold]
        if not low_perf.empty:
            st.subheader("Low performing students")
            st.write(low_perf.reset_index().rename(columns={"Marks":"Avg Marks"}))
    if not att_df.empty and "_present_flag_" in att_df:
        avg_att = att_df.groupby("Name")["_present_flag_"].mean()
        low_att = avg_att[avg_att<flag_att_threshold_pct/100]
        if not low_att.empty:
            st.subheader("Low attendance students")
            st.write(low_att.reset_index().rename(columns={"_present_flag_":"Attendance Rate"}))
