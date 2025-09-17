import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from io import StringIO

# -----------------------------
# Student Analytics Dashboard
# Single-file Streamlit app (robust, interactive, uploadable)
# -----------------------------

st.set_page_config(page_title="Student Analytics Dashboard", layout="wide")
st.title("📊 Student Analytics Dashboard — Robust Edition")

# -----------------------------
# Utilities
# -----------------------------
@st.cache_data
def safe_read_csv(uploaded) -> pd.DataFrame:
    """Read a CSV from uploaded file-like or path string robustly."""
    if uploaded is None:
        return pd.DataFrame()
    try:
        return pd.read_csv(uploaded)
    except Exception:
        # try with different encodings / separators
        try:
            return pd.read_csv(uploaded, encoding='utf-8', sep=None, engine='python')
        except Exception:
            return pd.read_csv(uploaded, encoding='latin1')


def ensure_date(df, col_candidates=["Date", "date", "day", "attendance_date"]):
    for c in col_candidates:
        if c in df.columns:
            try:
                df[c] = pd.to_datetime(df[c], errors='coerce')
                return df, c
            except Exception:
                continue
    # fallback: try to infer any datetime-like column
    for c in df.columns:
        if df[c].dtype == object:
            parsed = pd.to_datetime(df[c], errors='coerce')
            if parsed.notna().sum() > 0:
                df[c] = parsed
                return df, c
    return df, None


def safe_numeric(df, col):
    if col not in df.columns:
        return df
    df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


# -----------------------------
# Sidebar: file upload or use example files
# -----------------------------
st.sidebar.header("Data")
use_upload = st.sidebar.radio("Load data from", ["Upload CSVs", "Use server CSVs (if present)"])

attendance = pd.DataFrame()
results = pd.DataFrame()

if use_upload == "Upload CSVs":
    att_file = st.sidebar.file_uploader("Upload attendance CSV", type=["csv"]) 
    res_file = st.sidebar.file_uploader("Upload results CSV", type=["csv"]) 
    if att_file is not None:
        attendance = safe_read_csv(att_file)
    if res_file is not None:
        results = safe_read_csv(res_file)
else:
    # fallback to local known filenames (useful when running on server)
    try:
        attendance = safe_read_csv("/mnt/data/cleaned_attendance_data.csv")
        results = safe_read_csv("/mnt/data/cleaned_result_data.csv")
    except Exception:
        attendance = pd.DataFrame()
        results = pd.DataFrame()

# Quick validation / preview
st.sidebar.markdown("---")
if attendance.empty and results.empty:
    st.warning("No data loaded yet. Upload two CSV files: attendance and results, or place them in /mnt/data with names cleaned_attendance_data.csv and cleaned_result_data.csv")
else:
    st.sidebar.success("Data loaded — see main view for previews")

# -----------------------------
# Data cleaning & normalization
# -----------------------------
@st.cache_data
def prepare_data(att: pd.DataFrame, res: pd.DataFrame):
    att = att.copy()
    res = res.copy()

    # normalize column names
    att.columns = [c.strip() for c in att.columns]
    res.columns = [c.strip() for c in res.columns]

    # Find student id column (common names)
    sid_candidates = ["StudentID", "student_id", "id", "studentid", "roll"]
    sid_att = next((c for c in att.columns if c in sid_candidates), None)
    sid_res = next((c for c in res.columns if c in sid_candidates), None)

    # If not found, try first column
    if sid_att is None and len(att.columns) > 0:
        sid_att = att.columns[0]
    if sid_res is None and len(res.columns) > 0:
        sid_res = res.columns[0]

    # rename them to StudentID
    if sid_att:
        att = att.rename(columns={sid_att: 'StudentID'})
    if sid_res:
        res = res.rename(columns={sid_res: 'StudentID'})

    # Ensure numeric attendance (if stored as 1/0 or Present/Absent)
    # common attendance column names
    att_col_candidates = ['Attendance', 'attendance', 'Present', 'present']
    att_col = next((c for c in att.columns if c in att_col_candidates), None)
    if att_col is None:
        # try to infer a binary/numeric column
        for c in att.columns:
            if att[c].dropna().isin([0,1]).any() or att[c].dropna().isin(['P','A','Present','Absent']).any():
                att_col = c
                break
    if att_col:
        # convert textual to numeric
        def map_att(x):
            if pd.isna(x):
                return np.nan
            if str(x).strip().lower() in ['p','present','1','yes','y','true']:
                return 1
            if str(x).strip().lower() in ['a','absent','0','no','n','false']:
                return 0
            try:
                return float(x)
            except Exception:
                return np.nan
        att['Attendance'] = att[att_col].apply(map_att)
    else:
        # If no attendance-like column, create an attendance column with NaN
        att['Attendance'] = np.nan

    # Results: find marks column
    mark_candidates = ['Marks', 'marks', 'Score', 'score', 'Total']
    mark_col = next((c for c in res.columns if c in mark_candidates), None)
    if mark_col:
        res['Marks'] = pd.to_numeric(res[mark_col], errors='coerce')
    else:
        # try to find numeric columns
        nums = [c for c in res.columns if pd.api.types.is_numeric_dtype(res[c])]
        if nums:
            res['Marks'] = pd.to_numeric(res[nums[0]], errors='coerce')
        else:
            res['Marks'] = np.nan

    # Dates
    att, att_date_col = ensure_date(att)
    if att_date_col is None:
        # create synthetic date index if absent
        att['Date'] = pd.NaT
    else:
        if att_date_col != 'Date':
            att = att.rename(columns={att_date_col: 'Date'})

    # common exam column
    exam_candidates = ['Exam', 'exam', 'Term', 'term', 'Subject']
    exam_col = next((c for c in res.columns if c in exam_candidates), None)
    if exam_col:
        res = res.rename(columns={exam_col: 'Exam'})
    else:
        res['Exam'] = res.get('Exam', 'Exam')

    # standardize StudentID to string
    for df in [att, res]:
        if 'StudentID' in df.columns:
            df['StudentID'] = df['StudentID'].astype(str)

    return att, res

attendance, results = prepare_data(attendance, results)

# -----------------------------
# KPIs and quick stats
# -----------------------------
st.markdown("---")
col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)

with col_kpi1:
    st.metric("Total Students (attendance)", int(attendance['StudentID'].nunique()) if not attendance.empty else 0)
with col_kpi2:
    st.metric("Total Students (results)", int(results['StudentID'].nunique()) if not results.empty else 0)
with col_kpi3:
    avg_att = attendance['Attendance'].mean() if 'Attendance' in attendance.columns and not attendance.empty else np.nan
    st.metric("Average Attendance", f"{(avg_att*100):.2f}%" if not np.isnan(avg_att) else "N/A")
with col_kpi4:
    avg_marks = results['Marks'].mean() if 'Marks' in results.columns and not results.empty else np.nan
    st.metric("Average Marks", f"{avg_marks:.2f}" if not np.isnan(avg_marks) else "N/A")

# -----------------------------
# Main Tabs
# -----------------------------
tabs = st.tabs(["Class Overview", "Student Profile", "Attendance Heatmap", "Comparison", "Results Analysis", "Alerts & Predictions"]) 

# -----------------------------
# 1) Class Overview
# -----------------------------
with tabs[0]:
    st.header("🏫 Class Overview")

    if attendance.empty and results.empty:
        st.info("Upload data to see class overview")
    else:
        # Attendance distribution across students
        if not attendance.empty and 'Attendance' in attendance.columns:
            att_by_student = attendance.groupby('StudentID')['Attendance'].mean().fillna(0).sort_values(ascending=False)
            st.subheader("Attendance by student (avg)")
            st.dataframe(att_by_student.reset_index().rename(columns={'Attendance':'AvgAttendance'}).head(50))
            st.bar_chart(att_by_student)

        # Marks distribution
        if not results.empty and 'Marks' in results.columns:
            st.subheader("Marks distribution")
            fig, ax = plt.subplots()
            sns.histplot(results['Marks'].dropna(), kde=True, ax=ax)
            st.pyplot(fig)

        # Attendance vs Marks (merged)
        if (not attendance.empty) and (not results.empty):
            merged = attendance.groupby('StudentID')['Attendance'].mean().reset_index().rename(columns={'Attendance':'AvgAttendance'})
            merged = merged.merge(results.groupby('StudentID')['Marks'].mean().reset_index().rename(columns={'Marks':'AvgMarks'}), on='StudentID', how='inner')
            st.subheader('Attendance vs Average Marks')
            fig, ax = plt.subplots()
            sns.scatterplot(data=merged, x='AvgAttendance', y='AvgMarks', ax=ax)
            ax.set_xlabel('Average Attendance (0-1)')
            st.pyplot(fig)

# -----------------------------
# 2) Student Profile
# -----------------------------
with tabs[1]:
    st.header("👤 Student Profile — Deep Dive")
    if attendance.empty and results.empty:
        st.info("Upload data to explore profiles")
    else:
        students = sorted(set(list(attendance.get('StudentID',[])) + list(results.get('StudentID',[]))))
        sel = st.selectbox('Select a student', students)
        if sel:
            st.subheader(f"Overview for Student {sel}")
            st.markdown("**Summary**")
            att_s = attendance[attendance['StudentID']==sel]
            res_s = results[results['StudentID']==sel]

            # Basic numbers
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Attendance %", f"{(att_s['Attendance'].mean()*100):.2f}%" if not att_s.empty and not np.isnan(att_s['Attendance'].mean()) else 'N/A')
            with col2:
                st.metric("Avg Marks", f"{res_s['Marks'].mean():.2f}" if not res_s.empty and not np.isnan(res_s['Marks'].mean()) else 'N/A')
            with col3:
                st.metric("Total Records (attendance)", len(att_s))

            # Attendance timeline
            if 'Date' in att_s.columns and att_s['Date'].notna().any():
                att_time = att_s.set_index('Date').sort_index()
                st.subheader('Attendance over time')
                st.line_chart(att_time['Attendance'].resample('D').mean().fillna(0))

            # Results timeline
            if not res_s.empty:
                st.subheader('Marks over exams')
                tmp = res_s.copy()
                if 'Exam' not in tmp.columns:
                    tmp['Exam'] = range(len(tmp))
                tmp = tmp.sort_values('Exam')
                st.line_chart(tmp.set_index('Exam')['Marks'])

            # Detailed tables
            with st.expander('Attendance records'):
                st.dataframe(att_s.sort_values('Date').reset_index(drop=True))
            with st.expander('Result records'):
                st.dataframe(res_s.reset_index(drop=True))

# -----------------------------
# 3) Attendance Heatmap
# -----------------------------
with tabs[2]:
    st.header('📅 Attendance Heatmap (calendar-like)')
    if attendance.empty:
        st.info('Upload attendance data to view heatmaps')
    else:
        students = attendance['StudentID'].unique()
        student = st.selectbox('Select Student for heatmap', students)
        att_s = attendance[attendance['StudentID']==student].copy()
        if att_s.empty or att_s['Date'].isna().all():
            st.warning('No date-attendance records for this student')
        else:
            att_s['Date'] = pd.to_datetime(att_s['Date'])
            att_s['val'] = att_s['Attendance'].fillna(0)

            # build pivot of year-month-day average attendance
            att_s['year'] = att_s['Date'].dt.year
            att_s['month'] = att_s['Date'].dt.month
            att_s['day'] = att_s['Date'].dt.day

            # allow choosing year-month
            years = sorted(att_s['year'].unique())
            y = st.selectbox('Year', years, index=len(years)-1)
            months = sorted(att_s[att_s['year']==y]['month'].unique())
            m = st.selectbox('Month', months)

            month_df = att_s[(att_s['year']==y)&(att_s['month']==m)]
            heat = month_df.pivot_table(index='day', columns='month', values='val', aggfunc='mean').fillna(0)

            # Simple vertical heatmap (days vs month)
            fig, ax = plt.subplots(figsize=(6,8))
            sns.heatmap(heat, annot=True, fmt='.2f', ax=ax, cbar=True)
            ax.set_ylabel('Day of month')
            st.pyplot(fig)

# -----------------------------
# 4) Comparison
# -----------------------------
with tabs[3]:
    st.header('⚖️ Compare two students')
    if attendance.empty and results.empty:
        st.info('Upload data to use comparison tools')
    else:
        students = sorted(set(list(attendance.get('StudentID',[])) + list(results.get('StudentID',[]))))
        st.write('Select up to two students to compare')
        sel = st.multiselect('Students', students, default=students[:2])
        if len(sel) >= 1:
            s1 = sel[0]
            s2 = sel[1] if len(sel) > 1 else None

            colA, colB = st.columns(2)
            with colA:
                st.subheader(f'Student {s1}')
                a1 = attendance[attendance['StudentID']==s1]
                r1 = results[results['StudentID']==s1]
                st.metric('Avg Attendance', f"{a1['Attendance'].mean()*100:.2f}%" if not a1.empty else 'N/A')
                st.metric('Avg Marks', f"{r1['Marks'].mean():.2f}" if not r1.empty else 'N/A')
            with colB:
                if s2:
                    st.subheader(f'Student {s2}')
                    a2 = attendance[attendance['StudentID']==s2]
                    r2 = results[results['StudentID']==s2]
                    st.metric('Avg Attendance', f"{a2['Attendance'].mean()*100:.2f}%" if not a2.empty else 'N/A')
                    st.metric('Avg Marks', f"{r2['Marks'].mean():.2f}" if not r2.empty else 'N/A')

            # Side-by-side plots
            st.subheader('Side-by-side attendance timelines')
            fig, ax = plt.subplots()
            if not attendance.empty:
                for s, label in zip(sel[:2], ['Left','Right']):
                    tmp = attendance[attendance['StudentID']==s].copy()
                    if 'Date' in tmp.columns:
                        tmp['Date'] = pd.to_datetime(tmp['Date'])
                        t = tmp.set_index('Date')['Attendance'].resample('D').mean().fillna(0)
                        t.plot(label=s, ax=ax)
                ax.legend()
                st.pyplot(fig)

# -----------------------------
# 5) Results Analysis
# -----------------------------
with tabs[4]:
    st.header('📑 Results Analysis')
    if results.empty:
        st.info('Upload results data to analyze marks')
    else:
        # Exam filter
        exams = results.get('Exam', pd.Series(dtype=object)).unique().tolist()
        exam = st.selectbox('Select exam/term (All shows combined)', ['All'] + exams)
        df = results.copy()
        if exam != 'All':
            df = df[df['Exam']==exam]

        st.subheader('Marks Summary')
        st.write(df['Marks'].describe())

        st.subheader('Top students')
        top = df.groupby('StudentID')['Marks'].mean().sort_values(ascending=False).head(20)
        st.dataframe(top.reset_index().rename(columns={'Marks':'AvgMarks'}))

# -----------------------------
# 6) Alerts & Simple Predictions
# -----------------------------
with tabs[5]:
    st.header('🚨 Alerts & Simple Predictions')
    if attendance.empty and results.empty:
        st.info('Upload data to generate alerts')
    else:
        # At-risk by attendance threshold
        thresh = st.slider('Attendance warning threshold (%)', 0, 100, 75)
        merged_att = attendance.groupby('StudentID')['Attendance'].mean().reset_index().rename(columns={'Attendance':'AvgAttendance'})
        at_risk = merged_att[merged_att['AvgAttendance'] < (thresh/100.0)]
        st.subheader('Students below attendance threshold')
        st.dataframe(at_risk)

        # At-risk by marks threshold
        mthresh = st.slider('Marks warning threshold', 0, 100, 40)
        merged_marks = results.groupby('StudentID')['Marks'].mean().reset_index().rename(columns={'Marks':'AvgMarks'})
        marks_risk = merged_marks[merged_marks['AvgMarks'] < mthresh]
        st.subheader('Students below marks threshold')
        st.dataframe(marks_risk)

        # Simple linear trend prediction for marks (if multiple exams)
        st.subheader('Simple marks trend (linear)')
        sid = st.selectbox('Select student to predict (if multiple exam records exist)', merged_marks['StudentID'].unique().tolist())
        if sid:
            stud = results[results['StudentID']==sid].copy()
            if len(stud) >= 3 and pd.api.types.is_numeric_dtype(stud['Marks']):
                # map exam to numeric index
                stud = stud.reset_index(drop=True)
                x = np.arange(len(stud))
                y = stud['Marks'].to_numpy()
                coeff = np.polyfit(x,y,1)
                pred_next = coeff[0]* (len(stud)) + coeff[1]
                st.write(f"Predicted next marks (linear fit): {pred_next:.2f}")
            else:
                st.write('Not enough exam records to predict (need >=3).')

# -----------------------------
# Footer: Export & Download
# -----------------------------
st.markdown('---')
if not attendance.empty:
    if st.button('Download cleaned attendance CSV'):
        csv = attendance.to_csv(index=False)
        st.download_button('Click to download attendance CSV', data=csv, file_name='cleaned_attendance_export.csv')
if not results.empty:
    if st.button('Download cleaned results CSV'):
        csv2 = results.to_csv(index=False)
        st.download_button('Click to download results CSV', data=csv2, file_name='cleaned_results_export.csv')

st.markdown('\n---\nBuilt with ❤️ — tell me what additional insights or visuals you want next!')
