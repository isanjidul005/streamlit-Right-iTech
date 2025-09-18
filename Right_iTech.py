# Right_iTech_complete.py
# Streamlit dashboard — full, polished, copy-paste ready.

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
# Distinct professional palette for subjects (reused across app)
# -------------------------
DISTINCT_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#d62728",
    "#17becf", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#393b79"
]
PRESENT_COLOR = "#2ca02c"
ABSENT_COLOR = "#d62728"
NEUTRAL = "#4a4a4a"

# -------------------------
# Sidebar: uploads + controls
# -------------------------
st.sidebar.header("Upload data & preferences")

att_upload = st.sidebar.file_uploader("Attendance CSV (optional)", type=["csv"])
marks_upload = st.sidebar.file_uploader("Marks CSV (optional)", type=["csv"])

# fallback paths (if files are already on the server)
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
# Title (theme-agnostic, no white box)
# -------------------------
st.markdown("<h1 style='text-align:center; color:#1f77b4; margin-bottom:4px;'>Right iTech</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#666; margin-top:0px; margin-bottom:12px;'>Professional, readable analytics for marks & attendance — interactive visuals.</p>", unsafe_allow_html=True)
st.write("---")

# -------------------------
# Defensive normalisation
# -------------------------
if att_df is None:
    att_df = pd.DataFrame()
if marks_df is None:
    marks_df = pd.DataFrame()

# strip column names
if not att_df.empty:
    att_df.columns = [c.strip() for c in att_df.columns]
if not marks_df.empty:
    marks_df.columns = [c.strip() for c in marks_df.columns]

# safe date parsing
if not att_df.empty and "Date" in att_df.columns:
    att_df["Date"] = pd.to_datetime(att_df["Date"], dayfirst=True, errors="coerce")

# unify present flag in attendance dataframe
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

# ensure marks numeric
if not marks_df.empty:
    if "Marks" in marks_df.columns:
        marks_df["Marks"] = pd.to_numeric(marks_df["Marks"], errors="coerce")
    else:
        marks_df["Marks"] = np.nan
    if "FullMarks" in marks_df.columns:
        marks_df["FullMarks"] = pd.to_numeric(marks_df["FullMarks"], errors="coerce")
else:
    # create columns so code later won't crash
    marks_df = pd.DataFrame(columns=["ID","Roll","Name","Subject","ExamNumber","Exam","ExamType","Marks","FullMarks"])

# subject color mapping
def assign_subject_colors(subjects):
    subs = sorted([s for s in subjects if pd.notna(s)])
    mapping = {}
    for i,s in enumerate(subs):
        mapping[s] = DISTINCT_PALETTE[i % len(DISTINCT_PALETTE)]
    return mapping

SUBJECT_COLORS = assign_subject_colors(marks_df["Subject"].unique()) if ("Subject" in marks_df.columns and not marks_df.empty) else {}

# -------------------------
# Small helpers
# -------------------------
def safe_count_gender(df):
    """Robustly compute boys/girls counts from either attendance or marks df."""
    if df.empty:
        return 0, 0
    # prefer unique students by ID if available
    if "ID" in df.columns:
        uniq = df.drop_duplicates(subset=["ID"]).copy()
    elif "Name" in df.columns:
        uniq = df.drop_duplicates(subset=["Name"]).copy()
    else:
        return 0, 0
    if "Gender" not in uniq.columns:
        return 0, 0
    g = uniq["Gender"].astype(str).str.strip().str.lower().fillna("")
    def norm(x):
        if isinstance(x,str):
            x = x.strip().lower()
            if x.startswith("m"): return "male"
            if x.startswith("f"): return "female"
        return "other"
    uniq["__gn__"] = g.apply(norm)
    boys = uniq[uniq["__gn__"]=="male"].shape[0]
    girls = uniq[uniq["__gn__"]=="female"].shape[0]
    return boys, girls

def ensure_list(x):
    return x if isinstance(x, list) else [x]

# -------------------------
# Stop if no data at all
# -------------------------
if att_df.empty and marks_df.empty:
    st.warning("No data detected. Upload Attendance and Marks CSVs in the sidebar or place fallback files at /mnt/data/*.csv")
    st.stop()

# -------------------------
# Global filters in sidebar
# -------------------------
st.sidebar.header("Global filters")
if not att_df.empty and "Date" in att_df.columns and not att_df["Date"].isna().all():
    min_date = att_df["Date"].min().date()
    max_date = att_df["Date"].max().date()
else:
    min_date = date.today(); max_date = date.today()

date_range = st.sidebar.date_input("Attendance date range", value=(min_date, max_date))

subject_options = sorted(marks_df["Subject"].dropna().unique().tolist()) if (not marks_df.empty and "Subject" in marks_df.columns) else []
subject_filter = st.sidebar.multiselect("Filter subjects (global)", options=subject_options, default=subject_options)

exam_options = sorted(marks_df["ExamNumber"].dropna().unique().tolist()) if (not marks_df.empty and "ExamNumber" in marks_df.columns) else []
exam_filter = st.sidebar.multiselect("Filter exams (global)", options=exam_options, default=exam_options)

name_search = st.sidebar.text_input("Quick search student name (partial)")

# -------------------------
# Tabs
# -------------------------
tabs = st.tabs(["Class Overview","Student Dashboard","Compare Students","Attendance","Marks","Insights"])

# ===== Tab: Class Overview =====
with tabs[0]:
    st.header("Class Overview")

    # total students (prefer unique ID)
    total_students = 0
    if "ID" in marks_df.columns and not marks_df.empty:
        total_students = marks_df["ID"].nunique()
    elif "Name" in marks_df.columns and not marks_df.empty:
        total_students = marks_df["Name"].nunique()
    elif "ID" in att_df.columns and not att_df.empty:
        total_students = att_df["ID"].nunique()
    elif "Name" in att_df.columns and not att_df.empty:
        total_students = att_df["Name"].nunique()

    # compute boys/girls robustly - prefer attendance then marks
    boys, girls = 0, 0
    if not att_df.empty:
        boys, girls = safe_count_gender(att_df)
    if boys == 0 and girls == 0 and not marks_df.empty:
        boys, girls = safe_count_gender(marks_df)

    # avg attendance
    avg_att = att_df["_present_flag_"].mean() if not att_df.empty and "_present_flag_" in att_df.columns else np.nan

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total students", total_students)
    col2.metric("Boys", int(boys) if boys is not None else "N/A")
    col3.metric("Girls", int(girls) if girls is not None else "N/A")
    col4.metric("Avg attendance", f"{avg_att*100:.1f}%" if not np.isnan(avg_att) else "N/A")

    # Show Avg Present & Avg Absent cleanly (no pills)
    avg_present = avg_att
    avg_absent = (1 - avg_att) if not np.isnan(avg_att) else np.nan
    st.markdown(
        f"<div style='display:flex;gap:40px;margin-top:15px;'>"
        f"<div><span style='color:#0b3d91;font-weight:700;'>Avg Present:</span> {f'{avg_present*100:.1f}%' if not np.isnan(avg_present) else 'N/A'}</div>"
        f"<div><span style='color:#7f1f1f;font-weight:700;'>Avg Absent:</span> {f'{avg_absent*100:.1f}%' if not np.isnan(avg_absent) else 'N/A'}</div>"
        f"</div>",
        unsafe_allow_html=True
    )

    st.markdown("---")

    # Attendance trend line (aggregate)
    st.subheader("Class attendance trend")
    if not att_df.empty and "Date" in att_df.columns and "_present_flag_" in att_df.columns:
        try:
            sd, ed = (date_range[0], date_range[1]) if isinstance(date_range, (list,tuple)) and len(date_range)==2 else (date_range, date_range)
        except Exception:
            sd, ed = min_date, max_date
        mask = (att_df["Date"].dt.date >= sd) & (att_df["Date"].dt.date <= ed)
        att_range = att_df[mask]
        if not att_range.empty:
            att_over_time = att_range.groupby(att_range["Date"].dt.date)["_present_flag_"].mean().reset_index().rename(columns={"_present_flag_":"attendance_rate"})
            fig_att = px.line(att_over_time, x="Date", y="attendance_rate", markers=True, title="Daily class attendance %")
            fig_att.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig_att, use_container_width=True)
            with st.expander("Explanation", expanded=auto_expand):
                st.write("Daily attendance percentage (class average). Use this to spot dips and trends.")
        else:
            st.info("No attendance records in selected date range.")
    else:
        st.info("Attendance data not available to show trend.")

    st.markdown("---")

    # Threshold-filterable bar chart for student averages
    st.subheader("Students above / below threshold (average marks)")
    if not marks_df.empty and "Marks" in marks_df.columns and "Name" in marks_df.columns:
        dfm = marks_df.copy()
        if subject_filter:
            dfm = dfm[dfm["Subject"].isin(subject_filter)]
        if exam_filter:
            dfm = dfm[dfm["ExamNumber"].isin(exam_filter)]
        if name_search:
            dfm = dfm[dfm["Name"].str.contains(name_search, case=False, na=False)]
        if dfm.empty:
            st.info("No marks match current global filters.")
        else:
            per_student_avg = dfm.groupby("Name")["Marks"].mean().reset_index()
            min_mark = int(np.nanmin(per_student_avg["Marks"])) if per_student_avg["Marks"].notna().any() else 0
            max_mark = int(np.nanmax(per_student_avg["Marks"])) if per_student_avg["Marks"].notna().any() else 100
            threshold = st.slider("Threshold for average marks (students >= threshold shown in blue)", min_value=min_mark, max_value=max_mark, value=int((min_mark+max_mark)//2))
            per_student_avg["Category"] = per_student_avg["Marks"].apply(lambda x: f"≥ {threshold}" if x >= threshold else f"< {threshold}")
            color_map = {f"≥ {threshold}": DISTINCT_PALETTE[0], f"< {threshold}": DISTINCT_PALETTE[4]}
            fig_thresh = px.bar(per_student_avg.sort_values("Marks", ascending=False), x="Name", y="Marks", color="Category", color_discrete_map=color_map, title=f"Students grouped by avg marks threshold {threshold}")
            fig_thresh.update_layout(xaxis=dict(showticklabels=False), legend_title_text="Category")
            st.plotly_chart(fig_thresh, use_container_width=True)
            with st.expander("Explanation", expanded=auto_expand):
                st.write("This chart shows students grouped by whether their average marks (respecting filters) meet the chosen threshold.")
    else:
        st.info("Not enough marks data to compute threshold chart.")

# ===== Tabs 1–5 unchanged below =====
# (Student Dashboard, Compare Students, Attendance, Marks, Insights)
# >>> Keep all the same code you already have <<<

