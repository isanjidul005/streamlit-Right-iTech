# right_itech_app.py
# Streamlit App: Right iTech Student Insights (with color guide)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date

st.set_page_config(page_title='Right iTech Student Insights', layout='wide', initial_sidebar_state='expanded')

# Unified color palette for all charts
PALETTE = px.colors.qualitative.Set2

# Subject-color mapping for consistency
SUBJECT_COLORS = {}
def assign_colors(subjects):
    for i, sub in enumerate(sorted(subjects)):
        SUBJECT_COLORS[sub] = PALETTE[i % len(PALETTE)]

# ----------------------
# Utility functions
# ----------------------
@st.cache_data
def safe_read_csv(path_or_buffer):
    try:
        return pd.read_csv(path_or_buffer)
    except Exception:
        return pd.DataFrame()


def parse_attendance_dates(df, date_col='Date'):
    df = df.copy()
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors='coerce')
    return df


def standardize_attendance_status(df, status_col='Status'):
    df = df.copy()
    if status_col in df.columns:
        df['_present_flag_'] = df[status_col].astype(str).str.upper().map({'P':1,'PRESENT':1,'1':1,'A':0,'ABSENT':0,'0':0})
        df['_present_flag_'] = df['_present_flag_'].fillna(df[status_col].astype(str).str[0].map({'P':1,'A':0}))
    else:
        df['_present_flag_'] = np.nan
    return df


def subject_summary(marks_df):
    if marks_df.empty:
        return pd.DataFrame()
    a = marks_df.groupby('Subject').agg(
        exams=('ExamNumber','nunique'),
        avg_score=('Marks', 'mean'),
        avg_full=('FullMarks','mean'),
        entries=('Marks','count')
    ).reset_index()
    return a


def student_overview_marks(marks_df):
    if marks_df.empty:
        return pd.DataFrame()
    s = marks_df.groupby(['ID','Roll','Name']).agg(
        total_exams=('ExamNumber','nunique'),
        avg_score=('Marks','mean'),
        total_entries=('Marks','count'),
        absent_count=('WasAbsent', lambda x: x.astype(str).str.lower().isin(['true','1','yes']).sum())
    ).reset_index()
    return s

# ----------------------
# Header & Uploads
# ----------------------
st.markdown("""
<style>
.card {background: linear-gradient(180deg, #ffffff, #fbfbff); padding: 14px; border-radius: 12px; box-shadow: 0 6px 20px rgba(0,0,0,0.06);}
.h1 {font-size:28px; font-weight:700}
.small {color: #666}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="h1">📊 Right iTech Student Insights</div>', unsafe_allow_html=True)
st.markdown('<div class="small">Sleek, interactive visualizations for marks & attendance. Explanations are hidden under each chart.</div>', unsafe_allow_html=True)
st.write('---')

with st.sidebar.expander('Upload CSVs (optional)'):
    uploaded_marks = st.file_uploader('Upload `cleanest_marks.csv` (marks, long format)', type=['csv'], key='marks')
    uploaded_att = st.file_uploader('Upload `combined_attendance.csv` (attendance)', type=['csv'], key='att')

marks_path_fallback = "/mnt/data/cleanest_marks.csv"
att_path_fallback = "/mnt/data/combined_attendance.csv"

marks_df = safe_read_csv(uploaded_marks if uploaded_marks is not None else marks_path_fallback)
att_df = safe_read_csv(uploaded_att if uploaded_att is not None else att_path_fallback)

if marks_df.empty and att_df.empty:
    st.error('No data available. Please upload at least one of the CSVs.')
    st.stop()

marks_df.columns = [c.strip() for c in marks_df.columns]
att_df.columns = [c.strip() for c in att_df.columns]

marks_df.rename(columns={'fullname':'Name','Student':'Name','student_name':'Name'}, inplace=True)
att_df.rename(columns={'fullname':'Name','Student':'Name','student_name':'Name'}, inplace=True)

# Clean & derive
if 'Marks' in marks_df.columns:
    marks_df['Marks'] = pd.to_numeric(marks_df['Marks'], errors='coerce')
else:
    marks_df['Marks'] = np.nan

if 'FullMarks' in marks_df.columns:
    marks_df['FullMarks'] = pd.to_numeric(marks_df['FullMarks'], errors='coerce')
else:
    marks_df['FullMarks'] = np.nan

if 'WasAbsent' in marks_df.columns:
    marks_df['WasAbsent'] = marks_df['WasAbsent'].astype(str)
else:
    marks_df['WasAbsent'] = 'False'

if not att_df.empty:
    att_df = parse_attendance_dates(att_df, date_col='Date')
    att_df = standardize_attendance_status(att_df, status_col='Status')

if not att_df.empty and 'ID' in att_df.columns:
    att_summary = att_df.groupby(['ID','Roll','Name']).agg(
        total_days=('Date','nunique'),
        present_count=('_present_flag_','sum')
    ).reset_index()
    att_summary['attendance_rate'] = att_summary['present_count']/att_summary['total_days']
else:
    att_summary = pd.DataFrame()

subj_summary = subject_summary(marks_df)
student_mark_summary = student_overview_marks(marks_df)

if not marks_df.empty and not att_summary.empty:
    if 'ID' in marks_df.columns:
        marks_with_att = marks_df.merge(att_summary[['ID','attendance_rate']], on='ID', how='left')
    else:
        marks_with_att = marks_df.copy()
else:
    marks_with_att = marks_df.copy()

# Assign subject colors if possible
if 'Subject' in marks_df.columns:
    assign_colors(marks_df['Subject'].dropna().unique())

# ----------------------
# Sidebar controls
# ----------------------
st.sidebar.header('Controls & Filters')
with st.sidebar.form('filters_form'):
    score_min = st.number_input('Minimum score', value=0, min_value=0, max_value=100)
    score_max = st.number_input('Maximum score', value=100, min_value=0, max_value=100)
    min_att_pct = st.slider('Minimum attendance rate (%)', min_value=0, max_value=100, value=0)
    st.form_submit_button('Apply filters')

if 'Name' in marks_df.columns:
    student_list = marks_df['Name'].dropna().unique().tolist()
elif 'Name' in att_df.columns:
    student_list = att_df['Name'].dropna().unique().tolist()
else:
    student_list = []

# ----------------------
# Tabs
# ----------------------
tabs = st.tabs(['Class overview','Single student','Compare students','Attendance explorer','Insights & Export'])

# ----------------------
# Color guide card
# ----------------------
if SUBJECT_COLORS:
    st.markdown("### 🎨 Color Guide")
    for sub, col in SUBJECT_COLORS.items():
        st.markdown(f"<div style='display:inline-block;width:20px;height:20px;background:{col};margin-right:6px;border-radius:4px;'></div> {sub}", unsafe_allow_html=True)

# ----------------------
# Tab: Class overview
# ----------------------
with tabs[0]:
    st.header('Class overview')

    if not marks_df.empty:
        st.subheader('Overall score distribution')
        hist_df = marks_df[(marks_df['Marks']>=score_min) & (marks_df['Marks']<=score_max)]
        fig = px.histogram(hist_df, x='Marks', nbins=30, color_discrete_sequence=PALETTE)
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("Explanation"):
            st.write("This histogram shows how student scores are spread across exams.")

    if not subj_summary.empty:
        st.subheader('Subject averages')
        fig = px.bar(subj_summary, x='Subject', y='avg_score', color='Subject', color_discrete_map=SUBJECT_COLORS)
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("Explanation"):
            st.write("This bar chart compares average scores across subjects.")

    if not marks_df.empty:
        st.subheader('Correlation between subjects')
        pivot = marks_df.groupby(['ID','Name','Subject'])['Marks'].mean().reset_index()
        wide = pivot.pivot_table(index=['ID','Name'], columns='Subject', values='Marks')
        if wide.shape[1] >= 2:
            corr = wide.corr()
            fig = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu', title='Correlation matrix')
            st.plotly_chart(fig, use_container_width=True)
            with st.expander("Explanation"):
                st.write("This matrix shows how performance in one subject relates to another.")

# ----------------------
# (The rest of the code stays unchanged from previous update, using SUBJECT_COLORS or PALETTE consistently)
# ----------------------
