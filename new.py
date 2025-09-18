# superduper_streamlit_app.py
# Streamlit App: SuperDuper Student Insights (Rewritten for your CSVs)
# This version is tailored for the two CSV schemas you provided:
#  - cleanest_marks.csv: long-format marks (columns: ID, Roll, Name, Subject, ExamType, ExamNumber, FullMarks, Marks, WasAbsent)
#  - combined_attendance.csv: daily attendance (columns: ID, Roll, Name, Gender, Date, Status)
# The app supports file upload (optional). If no files are uploaded it uses the sample files at /mnt/data/...

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date

st.set_page_config(page_title='SuperDuper Student Insights', layout='wide', initial_sidebar_state='expanded')

# ----------------------
# Utility functions
# ----------------------
@st.cache_data
def safe_read_csv(path_or_buffer):
    try:
        return pd.read_csv(path_or_buffer)
    except Exception as e:
        return pd.DataFrame()


def parse_attendance_dates(df, date_col='Date'):
    df = df.copy()
    if date_col in df.columns:
        # Try multiple common formats; dayfirst True to handle dd/mm/yyyy like your data
        try:
            df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors='coerce')
        except Exception:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    return df


def standardize_attendance_status(df, status_col='Status'):
    df = df.copy()
    if status_col in df.columns:
        df['_present_flag_'] = df[status_col].astype(str).str.upper().map({'P':1,'PRESENT':1,'1':1,'A':0,'ABSENT':0,'0':0})
        # if mapping produced NaNs, try to infer from single characters
        df['_present_flag_'] = df['_present_flag_'].fillna(df[status_col].astype(str).str[0].map({'P':1,'A':0}))
    else:
        df['_present_flag_'] = np.nan
    return df


def subject_summary(marks_df):
    # returns subject-level aggregates
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

st.markdown('<div class="h1">📊 SuperDuper Student Insights</div>', unsafe_allow_html=True)
st.markdown('<div class="small">Sleek, interactive visualizations for marks & attendance. Toggle explanations with the sidebar option.</div>', unsafe_allow_html=True)
st.write('---')

with st.sidebar.expander('Upload CSVs (optional)'):
    uploaded_marks = st.file_uploader('Upload `cleanest_marks.csv` (marks, long format)', type=['csv'], key='marks')
    uploaded_att = st.file_uploader('Upload `combined_attendance.csv` (attendance)', type=['csv'], key='att')

# fallback to provided sample files
marks_path_fallback = "/mnt/data/cleanest_marks.csv"
att_path_fallback = "/mnt/data/combined_attendance.csv"

marks_df = safe_read_csv(uploaded_marks if uploaded_marks is not None else marks_path_fallback)
att_df = safe_read_csv(uploaded_att if uploaded_att is not None else att_path_fallback)

if marks_df.empty and att_df.empty:
    st.error('No data available. Please upload at least one of the CSVs.')
    st.stop()

# Normalize column names to expected casing
marks_df.columns = [c.strip() for c in marks_df.columns]
att_df.columns = [c.strip() for c in att_df.columns]

# Ensure expected columns exist, rename if there are slight variations
# For marks: ID, Roll, Name, Subject, ExamType, ExamNumber, FullMarks, Marks, WasAbsent
marks_df.rename(columns={
    'fullname':'Name', 'Student':'Name', 'student_name':'Name'
}, inplace=True)

att_df.rename(columns={
    'fullname':'Name', 'Student':'Name', 'student_name':'Name'
}, inplace=True)

# ----------------------
# Clean & derive fields
# ----------------------
# Marks: ensure numeric Marks and FullMarks
if 'Marks' in marks_df.columns:
    marks_df['Marks'] = pd.to_numeric(marks_df['Marks'], errors='coerce')
else:
    marks_df['Marks'] = np.nan

if 'FullMarks' in marks_df.columns:
    marks_df['FullMarks'] = pd.to_numeric(marks_df['FullMarks'], errors='coerce')
else:
    marks_df['FullMarks'] = np.nan

# WasAbsent normalization
if 'WasAbsent' in marks_df.columns:
    marks_df['WasAbsent'] = marks_df['WasAbsent'].astype(str)
else:
    marks_df['WasAbsent'] = 'False'

# Attendance: parse dates and status
if not att_df.empty:
    att_df = parse_attendance_dates(att_df, date_col='Date')
    att_df = standardize_attendance_status(att_df, status_col='Status')

# attendance summary per student
if not att_df.empty and 'ID' in att_df.columns:
    att_summary = att_df.groupby(['ID','Roll','Name']).agg(
        total_days=('Date','nunique'),
        present_count=('_present_flag_','sum')
    ).reset_index()
    att_summary['attendance_rate'] = att_summary['present_count']/att_summary['total_days']
else:
    att_summary = pd.DataFrame()

# Subject summary and student summary
subj_summary = subject_summary(marks_df)
student_mark_summary = student_overview_marks(marks_df)

# Merge marks + attendance for convenience
if not marks_df.empty and not att_summary.empty:
    # merge on ID if present
    if 'ID' in marks_df.columns:
        marks_with_att = marks_df.merge(att_summary[['ID','attendance_rate']], on='ID', how='left')
    else:
        marks_with_att = marks_df.copy()
else:
    marks_with_att = marks_df.copy()

# ----------------------
# Sidebar controls
# ----------------------
st.sidebar.header('Controls & Filters')
with st.sidebar.form('filters_form'):
    score_min = st.number_input('Minimum score (for filtering visualizations)', value=0, min_value=0, max_value=100)
    score_max = st.number_input('Maximum score (for filtering visualizations)', value=100, min_value=0, max_value=100)
    min_att_pct = st.slider('Minimum attendance rate (%)', min_value=0, max_value=100, value=0)
    show_explanations = st.checkbox('Show explanations / guidance', value=False)
    st.form_submit_button('Apply filters')

# Build student list
student_list = []
if 'Name' in marks_df.columns and not marks_df['Name'].isna().all():
    student_list = marks_df['Name'].astype(str).unique().tolist()
elif 'Name' in att_df.columns and not att_df['Name'].isna().all():
    student_list = att_df['Name'].astype(str).unique().tolist()

# ----------------------
# Main layout - Tabs
# ----------------------
tabs = st.tabs(['Class overview','Single student','Compare students','Attendance explorer','Insights & Export'])

# ----------------------
# Tab: Class overview
# ----------------------
with tabs[0]:
    st.header('Class overview: simple + advanced visuals')
    if show_explanations:
        with st.expander('What you will see (explanation)', expanded=False):
            st.write('This page summarizes the whole class. Simple visuals (histogram, summary numbers) are for quick understanding. Advanced visuals (correlation heatmap, subject trends) help teachers discover patterns across subjects and exams.')

    c1, c2 = st.columns([3,2])
    with c1:
        st.subheader('Overall score distribution (all subjects & exams) — Simple')
        if not marks_df.empty:
            hist_df = marks_df[(marks_df['Marks']>=score_min) & (marks_df['Marks']<=score_max)]
            fig = px.histogram(hist_df, x='Marks', nbins=30, title='Score distribution', marginal='box', labels={'Marks':'Score'})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info('No marks data available.')

    with c2:
        st.subheader('Quick summary numbers')
        avg_score = marks_df['Marks'].mean() if 'Marks' in marks_df.columns else np.nan
        total_students = marks_df['ID'].nunique() if 'ID' in marks_df.columns else 'Unknown'
        avg_att = att_summary['attendance_rate'].mean() if not att_summary.empty else np.nan
        st.metric('Average score (all entries)', f"{avg_score:.2f}" if not np.isnan(avg_score) else 'N/A')
        st.metric('Unique students in marks', total_students)
        st.metric('Average attendance rate (class)', f"{avg_att:.1%}" if not np.isnan(avg_att) else 'N/A')

    st.markdown('---')
    st.subheader('Subject-level insights (basic + complex)')
    col1, col2 = st.columns(2)
    with col1:
        if not subj_summary.empty:
            st.markdown('**Subject averages**')
            st.table(subj_summary[['Subject','avg_score','exams','entries']].sort_values('avg_score', ascending=False).rename(columns={'avg_score':'Avg score','exams':'#Exams','entries':'#Records'}).set_index('Subject'))
        else:
            st.info('No subject summary available.')
    with col2:
        # radar of top subjects by average (complex but intuitive)
        if not subj_summary.empty:
            top_subj = subj_summary.sort_values('avg_score', ascending=False).head(8)
            fig = go.Figure()
            fig.add_trace(go.Bar(x=top_subj['Subject'], y=top_subj['avg_score'], name='Avg score'))
            fig.update_layout(title='Top subject average scores')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info('No subject data to plot.')

    st.subheader('Correlation across subjects (complex)')
    # prepare pivot table: each student as row, subject averages as columns
    if not marks_df.empty:
        pivot = marks_df.groupby(['ID','Name','Subject'])['Marks'].mean().reset_index()
        wide = pivot.pivot_table(index=['ID','Name'], columns='Subject', values='Marks')
        if wide.shape[1] >= 2:
            corr = wide.corr()
            fig = px.imshow(corr, text_auto=True, title='Correlation matrix between subjects')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info('Not enough distinct subjects to compute correlation matrix.')
    else:
        st.info('No marks data')

# ----------------------
# Tab: Single Student
# ----------------------
with tabs[1]:
    st.header('Single student: profile, timeline & simple explanations')
    if show_explanations:
        with st.expander('How to interpret this page', expanded=False):
            st.write('Pick a student. You will see their profile, marks timeline per subject and attendance timeline. Hover charts for tooltips. Use the simple visuals to quickly spot drops or spikes in performance.')

    student = st.selectbox('Select student', options=sorted(student_list)) if student_list else st.text_input('Enter student name')

    if student:
        # student profile
        st.subheader('Profile & summary')
        if not marks_df.empty and 'Name' in marks_df.columns:
            s_marks = marks_df[marks_df['Name'].astype(str)==student]
        else:
            s_marks = pd.DataFrame()
        if not att_df.empty and 'Name' in att_df.columns:
            s_att = att_df[att_df['Name'].astype(str)==student]
        else:
            s_att = pd.DataFrame()

        col1, col2 = st.columns([2,3])
        with col1:
            if not s_marks.empty:
                sid = s_marks['ID'].iloc[0] if 'ID' in s_marks.columns else 'N/A'
                roll = s_marks['Roll'].iloc[0] if 'Roll' in s_marks.columns else 'N/A'
                st.markdown(f"**Name:** {student}")
                st.markdown(f"**ID:** {sid}")
                st.markdown(f"**Roll:** {roll}")
                st.markdown(f"**Avg score:** {s_marks['Marks'].mean():.2f}")
            else:
                st.info('No marks records found for this student.')
        with col2:
            if not s_att.empty:
                present = s_att['_present_flag_'].sum()
                total = s_att.shape[0]
                rate = present/total if total>0 else np.nan
                st.metric('Attendance rate', f"{rate:.1%}" if not np.isnan(rate) else 'N/A', delta=f"{int(present)} present out of {int(total)}")
            else:
                st.info('No attendance records for this student.')

        st.markdown('---')
        st.subheader('Marks timeline (simple)')
        if not s_marks.empty:
            # show average per exam or per examnumber
            tm = s_marks.groupby(['ExamNumber','ExamType']).agg(avg=('Marks','mean')).reset_index()
            fig = px.line(tm.sort_values('ExamNumber'), x='ExamNumber', y='avg', markers=True, title='Average score across exams')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info('No marks to plot.')

        st.subheader('Marks by subject (complex)')
        if not s_marks.empty:
            fig = px.box(s_marks, x='Subject', y='Marks', points='all', title='Score distribution per subject (this student)')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info('No marks to plot.')

        st.subheader('Attendance timeline (simple)')
        if not s_att.empty and 'Date' in s_att.columns:
            s_att_sorted = s_att.sort_values('Date')
            # Convert dates to simple date objects for plotting widget compatibility
            st.plotly_chart(px.scatter(s_att_sorted, x='Date', y='_present_flag_', title='Present (1) / Absent (0) over time').update_yaxes(tickmode='array', tickvals=[0,1]), use_container_width=True)
        else:
            st.info('No attendance dates to plot.')

# ----------------------
# Tab: Compare students
# ----------------------
with tabs[2]:
    st.header('Compare students (simple and complex)')
    if show_explanations:
        with st.expander('Tips', expanded=False):
            st.write('Select up to 6 students to compare. Simple charts show averages; complex charts like radar normalize scores to make different subjects comparable.')

    sel = st.multiselect('Select up to 6 students', options=sorted(student_list), max_selections=6)
    if sel and not marks_df.empty:
        comp = marks_df[marks_df['Name'].isin(sel)]
        # simple grouped bar for average per student
        avg_by_student = comp.groupby('Name')['Marks'].mean().reset_index()
        st.subheader('Average score (simple)')
        st.plotly_chart(px.bar(avg_by_student, x='Name', y='Marks', title='Average score per selected student').update_layout(xaxis={'categoryorder':'total descending'}), use_container_width=True)

        st.subheader('Radar chart (complex)')
        # prepare normalized subject profile
        pivot = comp.groupby(['Name','Subject'])['Marks'].mean().reset_index()
        wide = pivot.pivot_table(index='Name', columns='Subject', values='Marks').fillna(0)
        if wide.shape[1] >= 3:
            categories = wide.columns.tolist()
            fig = go.Figure()
            for idx, r in wide.iterrows():
                vals = r.values.tolist()
                max_val = np.nanmax(wide.values) if wide.values.size else 1
                if max_val==0:
                    max_val = 1
                norm = [v/max_val for v in vals]
                fig.add_trace(go.Scatterpolar(r=norm, theta=categories, fill='toself', name=str(idx)))
            fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1])), showlegend=True, title='Normalized subject profile (radar)')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info('Not enough distinct subjects for radar. Try selecting different students or more of them.')

        # attendance vs average scatter
        if not att_summary.empty and 'ID' in marks_df.columns:
            merged = marks_df.groupby(['ID','Name']).agg(avg_score=('Marks','mean')).reset_index().merge(att_summary[['ID','attendance_rate']], on='ID', how='left')
            merged_sel = merged[merged['Name'].isin(sel)]
            if not merged_sel.empty:
                st.subheader('Attendance vs average score (simple scatter)')
                st.plotly_chart(px.scatter(merged_sel, x='attendance_rate', y='avg_score', text='Name', size='avg_score', title='Attendance vs Average Score').update_layout(xaxis_tickformat='.0%'), use_container_width=True)
    else:
        st.info('Select students above to compare.')

# ----------------------
# Tab: Attendance explorer
# ----------------------
with tabs[3]:
    st.header('Attendance explorer — heatmaps & trends')
    if show_explanations:
        with st.expander('What this page shows', expanded=False):
            st.write('Heatmap: students × dates showing presence. Trends: class attendance over time. Use the date range selector to zoom.')

    if not att_df.empty:
        # date range inputs must be Python date objects
        min_date = att_df['Date'].min().date() if 'Date' in att_df.columns and not att_df['Date'].isna().all() else date.today()
        max_date = att_df['Date'].max().date() if 'Date' in att_df.columns and not att_df['Date'].isna().all() else date.today()
        start_d, end_d = st.date_input('Select date range', value=(min_date, max_date))
        if isinstance(start_d, tuple) or isinstance(start_d, list):
            start_d, end_d = start_d[0], start_d[1]
        # filter
        mask = (att_df['Date'].dt.date >= start_d) & (att_df['Date'].dt.date <= end_d)
        att_filtered = att_df[mask]

        st.subheader('Class attendance over time (simple)')
        att_over_time = att_filtered.groupby(att_filtered['Date'].dt.date)['_present_flag_'].mean().reset_index().rename(columns={'Date':'date','_present_flag_':'attendance_rate'})
        st.plotly_chart(px.line(att_over_time, x='Date', y='attendance_rate', title='Class average attendance over time').update_yaxes(tickformat='.0%'), use_container_width=True)

        st.subheader('Attendance heatmap (complex)')
        try:
            heat = att_filtered.pivot_table(index='Name', columns=att_filtered['Date'].dt.date, values='_present_flag_', aggfunc='mean').fillna(0)
            # reduce to top N students by records to keep heatmap readable
            top_n = st.slider('Number of students to show in heatmap', min_value=10, max_value=200, value=50)
            top_students = heat.mean(axis=1).sort_values(ascending=False).head(top_n).index
            heat_small = heat.loc[top_students]
            fig = px.imshow(heat_small.values, x=[str(d) for d in heat_small.columns], y=heat_small.index, aspect='auto', title=f'Attendance heatmap ({len(heat_small)} students × {len(heat_small.columns)} dates)')
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.info('Could not build heatmap. Ensure attendance has Name and Date columns.')
    else:
        st.info('No attendance data loaded.')

# ----------------------
# Tab: Insights & Export
# ----------------------
with tabs[4]:
    st.header('Automated insights & export')
    if show_explanations:
        with st.expander('What automated insights do', expanded=False):
            st.write('We flag students who have below-threshold attendance OR below-threshold average marks. You can download cleaned CSVs for further analysis or reporting.')

    # flags: low attendance or low avg score
    low_att_thresh = min_att_pct/100.0
    low_score_thresh = score_min

    # prepare merged student-level summary
    student_level = student_mark_summary.merge(att_summary[['ID','attendance_rate']], on='ID', how='left') if (not student_mark_summary.empty and not att_summary.empty) else student_mark_summary.copy()

    if not student_level.empty:
        student_level['flag_low_attendance'] = student_level['attendance_rate'].fillna(1) < low_att_thresh
        student_level['flag_low_score'] = student_level['avg_score'] < low_score_thresh
        flagged = student_level[student_level['flag_low_attendance'] | student_level['flag_low_score']]
        st.markdown(f"**Flagged students:** {len(flagged)}")
        if not flagged.empty:
            st.dataframe(flagged[['ID','Roll','Name','avg_score','attendance_rate','flag_low_attendance','flag_low_score']].rename(columns={'avg_score':'Avg score','attendance_rate':'Attendance rate'}))
        else:
            st.success('No flagged students with current thresholds.')
    else:
        st.info('Not enough student-level data to generate flags.')

    st.markdown('---')
    st.subheader('Download cleaned/processed datasets')
    if not marks_df.empty:
        st.download_button('Download cleaned marks CSV', marks_df.to_csv(index=False).encode('utf-8'), file_name='cleaned_marks_processed.csv')
    if not att_df.empty:
        st.download_button('Download cleaned attendance CSV', att_df.to_csv(index=False).encode('utf-8'), file_name='cleaned_attendance_processed.csv')

    st.markdown('You can copy this app code and extend it with school-specific grading rules, custom dashboards per class/section, or PDF/Excel report exports.')

# ----------------------
# Footer
# ----------------------
st.write('---')
st.caption('Built for your provided CSVs: cleanest_marks.csv and combined_attendance.csv — contact me for custom tweaks (grading scales, term grouping, smart alerts).')
