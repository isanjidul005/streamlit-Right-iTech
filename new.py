# right_itech_app.py
# Streamlit App: Right iTech Student Insights
# Full-featured version: all tabs populated, unified palette, per-chart explanations,
# and export (Excel + PDF) for single-student reports and flagged lists.

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date
import io
import tempfile
import base64

# PDF/Excel libraries
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4

st.set_page_config(page_title='Right iTech Student Insights', layout='wide', initial_sidebar_state='expanded')

# ---------- Configuration ----------
PALETTE = px.colors.qualitative.Set2
ATT_PRESENT_COLOR = '#2ca02c'  # green
ATT_ABSENT_COLOR = '#d62728'   # red

# Subject-color mapping
SUBJECT_COLORS = {}
def assign_colors(subjects):
    for i, sub in enumerate(sorted(subjects)):
        SUBJECT_COLORS[sub] = PALETTE[i % len(PALETTE)]

# ---------- Utils ----------
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

# ---------- Header + Upload UI ----------
st.markdown("""
<style>
.h1 {font-size:28px; font-weight:700}
.small {color: #666}
.color-box {display:inline-block;width:18px;height:18px;border-radius:4px;margin-right:8px}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="h1">📊 Right iTech Student Insights</div>', unsafe_allow_html=True)
st.markdown('<div class="small">Interactive, teacher-friendly visualizations with exportable student & class reports.</div>', unsafe_allow_html=True)
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

# ---------- Cleaning ----------
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

if 'Subject' in marks_df.columns:
    assign_colors(marks_df['Subject'].dropna().unique())

# ---------- Sidebar controls ----------
st.sidebar.header('Controls & Filters')
with st.sidebar.form('filters_form'):
    score_min = st.number_input('Minimum score', value=0, min_value=0, max_value=100)
    score_max = st.number_input('Maximum score', value=100, min_value=0, max_value=100)
    min_att_pct = st.slider('Minimum attendance rate (%)', min_value=0, max_value=100, value=0)
    show_explanations = st.checkbox('Show explanations (auto-expand)', value=False)
    st.form_submit_button('Apply filters')

if 'Name' in marks_df.columns and not marks_df['Name'].isna().all():
    student_list = sorted(marks_df['Name'].dropna().unique().tolist())
elif 'Name' in att_df.columns and not att_df['Name'].isna().all():
    student_list = sorted(att_df['Name'].dropna().unique().tolist())
else:
    student_list = []

# ---------- Color guide ----------
if SUBJECT_COLORS:
    st.markdown('### 🎨 Color guide')
    cols_html = ''
    for sub, col in SUBJECT_COLORS.items():
        cols_html += f"<div style='display:inline-flex;align-items:center;margin-right:12px'><div class='color-box' style='background:{col}'></div><div>{sub}</div></div>"
    st.markdown(cols_html, unsafe_allow_html=True)
    st.write('---')

# ---------- Tabs ----------
tabs = st.tabs(['Class overview','Single student','Compare students','Attendance explorer','Insights & Export'])

# ---- Class overview ----
with tabs[0]:
    st.header('Class overview')

    if not marks_df.empty:
        st.subheader('Overall score distribution')
        hist_df = marks_df[(marks_df['Marks']>=score_min) & (marks_df['Marks']<=score_max)]
        fig = px.histogram(hist_df, x='Marks', nbins=30, color_discrete_sequence=PALETTE)
        st.plotly_chart(fig, use_container_width=True)
        with st.expander('Explanation', expanded=show_explanations):
            st.write('This histogram shows how student scores are distributed across all subjects and exams. Boxplot gives a quick sense of median and spread.')

    if not subj_summary.empty:
        st.subheader('Subject averages (simple)')
        fig = px.bar(subj_summary, x='Subject', y='avg_score', color='Subject', color_discrete_map=SUBJECT_COLORS)
        st.plotly_chart(fig, use_container_width=True)
        with st.expander('Explanation', expanded=show_explanations):
            st.write('Each bar is the average score for that subject across all students and exams. Use this to spot subjects where the class is excelling or struggling.')

    if not marks_df.empty:
        st.subheader('Correlation between subjects (complex)')
        pivot = marks_df.groupby(['ID','Name','Subject'])['Marks'].mean().reset_index()
        wide = pivot.pivot_table(index=['ID','Name'], columns='Subject', values='Marks')
        if wide.shape[1] >= 2:
            corr = wide.corr()
            fig = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu', title='Correlation matrix')
            st.plotly_chart(fig, use_container_width=True)
            with st.expander('Explanation', expanded=show_explanations):
                st.write('A positive correlation (blue) means students who do well in one subject tend to do well in another. Negative correlation (red) indicates opposite trends.')

# ---- Single student ----
with tabs[1]:
    st.header('Single student overview')
    if not student_list:
        st.info('No students detected. Upload data with Name column.')
    else:
        student = st.selectbox('Select student', student_list)

        if student:
            s_marks = marks_df[marks_df['Name']==student]
            s_att = att_df[att_df['Name']==student] if not att_df.empty else pd.DataFrame()

            # Profile
            st.subheader('Profile & quick metrics')
            col1, col2 = st.columns(2)
            with col1:
                if not s_marks.empty:
                    sid = s_marks['ID'].iloc[0] if 'ID' in s_marks.columns else 'N/A'
                    roll = s_marks['Roll'].iloc[0] if 'Roll' in s_marks.columns else 'N/A'
                    st.markdown(f'**Name:** {student}')
                    st.markdown(f'**ID:** {sid}  
**Roll:** {roll}')
                    st.markdown(f'**Avg score:** {s_marks["Marks"].mean():.2f}')
                else:
                    st.info('No marks for this student.')
            with col2:
                if not s_att.empty:
                    present = int(s_att['_present_flag_'].sum())
                    total = int(s_att.shape[0])
                    rate = present/total if total>0 else np.nan
                    st.metric('Attendance rate', f"{rate:.1%}" if not np.isnan(rate) else 'N/A', delta=f"{present}/{total} present")
                else:
                    st.info('No attendance records for this student.')

            st.markdown('---')
            # Simple charts
            st.subheader('Simple charts')
            c1, c2 = st.columns(2)
            with c1:
                st.markdown('**Subject-wise average (bar)**')
                if not s_marks.empty:
                    subj_avg = s_marks.groupby('Subject')['Marks'].mean().reset_index()
                    fig = px.bar(subj_avg, x='Subject', y='Marks', color='Subject', color_discrete_map=SUBJECT_COLORS)
                    st.plotly_chart(fig, use_container_width=True)
                    with st.expander('Explanation', expanded=show_explanations):
                        st.write('Shows the student\'s average score per subject. Helps identify strengths and weaknesses.')
                else:
                    st.info('No marks to show.')
            with c2:
                st.markdown('**Attendance breakdown (pie)**')
                if not s_att.empty:
                    pres = int(s_att['_present_flag_'].sum())
                    absn = int((s_att['_present_flag_']==0).sum())
                    pie_df = pd.DataFrame({'status':['Present','Absent'],'count':[pres,absn]})
                    fig = px.pie(pie_df, values='count', names='status', color='status', color_discrete_map={'Present':ATT_PRESENT_COLOR,'Absent':ATT_ABSENT_COLOR})
                    st.plotly_chart(fig, use_container_width=True)
                    with st.expander('Explanation', expanded=show_explanations):
                        st.write('Pie chart shows proportion of days present vs absent.')
                else:
                    st.info('No attendance data.')

            st.markdown('---')
            # Complex charts
            st.subheader('Complex charts')
            st.markdown('**Marks trend across exams (line)**')
            if not s_marks.empty:
                tm = s_marks.groupby(['ExamNumber','ExamType']).agg(avg=('Marks','mean')).reset_index()
                fig = px.line(tm.sort_values('ExamNumber'), x='ExamNumber', y='avg', markers=True, title='Average across exams', color_discrete_sequence=PALETTE)
                st.plotly_chart(fig, use_container_width=True)
                with st.expander('Explanation', expanded=show_explanations):
                    st.write('Line shows how the student\'s average across exams changed over time. Use this to see improvement or decline.')
            else:
                st.info('No marks to plot.')

            st.markdown('**Attendance timeline (dots)**')
            if not s_att.empty:
                s_att_sorted = s_att.sort_values('Date')
                fig = px.scatter(s_att_sorted, x='Date', y='_present_flag_', title='Attendance timeline', color_discrete_sequence=[ATT_PRESENT_COLOR])
                fig.update_yaxes(tickmode='array', tickvals=[0,1], ticktext=['Absent','Present'])
                st.plotly_chart(fig, use_container_width=True)
                with st.expander('Explanation', expanded=show_explanations):
                    st.write('Each dot represents attendance on a date (Present=1, Absent=0). Patterns reveal streaks of absence.')
            else:
                st.info('No attendance dates to plot.')

            # Export buttons
            st.markdown('---')
            st.subheader('Export student report')
            colx, coly = st.columns(2)
            with colx:
                if st.button('Download Excel report'):
                    buffer = generate_student_excel_report(student, marks_df, att_df)
                    st.download_button('Click to download Excel', data=buffer.getvalue(), file_name=f'{student}_report.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
            with coly:
                if st.button('Download PDF report'):
                    pdf_bytes = generate_student_pdf_report(student, marks_df, att_df)
                    st.download_button('Click to download PDF', data=pdf_bytes, file_name=f'{student}_report.pdf', mime='application/pdf')

# ---- Compare students ----
with tabs[2]:
    st.header('Compare students')
    sel = st.multiselect('Select up to 6 students', options=student_list, max_selections=6)

    if sel and not marks_df.empty:
        comp = marks_df[marks_df['Name'].isin(sel)]
        # Simple
        st.subheader('Simple: Average score per selected student')
        avg_by_student = comp.groupby('Name')['Marks'].mean().reset_index()
        fig = px.bar(avg_by_student, x='Name', y='Marks', color='Name', color_discrete_sequence=PALETTE)
        st.plotly_chart(fig, use_container_width=True)
        with st.expander('Explanation', expanded=show_explanations):
            st.write('Compare average performance between selected students.')

        # Complex - Radar
        st.subheader('Complex: Radar chart (normalized)')
        pivot = comp.groupby(['Name','Subject'])['Marks'].mean().reset_index()
        wide = pivot.pivot_table(index='Name', columns='Subject', values='Marks').fillna(0)
        if wide.shape[1] >= 3:
            categories = wide.columns.tolist()
            fig = go.Figure()
            max_val = np.nanmax(wide.values)
            for idx, r in wide.iterrows():
                norm = [v/max_val if max_val>0 else 0 for v in r.values.tolist()]
                fig.add_trace(go.Scatterpolar(r=norm, theta=categories, fill='toself', name=str(idx)))
            fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1])), showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
            with st.expander('Explanation', expanded=show_explanations):
                st.write('Radar normalizes scores so you can compare subject strengths across students.')

        # Scatter attendance vs marks
        if not att_summary.empty and 'ID' in marks_df.columns:
            merged = marks_df.groupby(['ID','Name']).agg(avg_score=('Marks','mean')).reset_index().merge(att_summary[['ID','attendance_rate']], on='ID', how='left')
            merged_sel = merged[merged['Name'].isin(sel)]
            if not merged_sel.empty:
                st.subheader('Attendance vs Average score (scatter)')
                fig = px.scatter(merged_sel, x='attendance_rate', y='avg_score', text='Name', size='avg_score')
                st.plotly_chart(fig, use_container_width=True)
                with st.expander('Explanation', expanded=show_explanations):
                    st.write('Each point is a student. x=attendance, y=average score. Helps see if attendance and performance correlate.')

# ---- Attendance explorer ----
with tabs[3]:
    st.header('Attendance explorer')
    if att_df.empty:
        st.info('No attendance data')
    else:
        min_date = att_df['Date'].min().date()
        max_date = att_df['Date'].max().date()
        start_d, end_d = st.date_input('Select date range', value=(min_date, max_date))
        if isinstance(start_d, tuple) or isinstance(start_d, list):
            start_d, end_d = start_d[0], start_d[1]
        mask = (att_df['Date'].dt.date >= start_d) & (att_df['Date'].dt.date <= end_d)
        att_filtered = att_df[mask]

        st.subheader('Class attendance over time (simple)')
        att_over_time = att_filtered.groupby(att_filtered['Date'].dt.date)['_present_flag_'].mean().reset_index()
        att_over_time.rename(columns={'Date':'Date','_present_flag_':'attendance_rate'}, inplace=True)
        fig = px.line(att_over_time, x='Date', y='attendance_rate', markers=True, color_discrete_sequence=[ATT_PRESENT_COLOR])
        fig.update_yaxes(tickformat='.0%')
        st.plotly_chart(fig, use_container_width=True)
        with st.expander('Explanation', expanded=show_explanations):
            st.write('Shows class average attendance per day in the selected range.')

        st.subheader('Monthly attendance (simple)')
        att_filtered['month'] = att_filtered['Date'].dt.to_period('M').astype(str)
        monthly = att_filtered.groupby('month')['_present_flag_'].mean().reset_index()
        fig = px.bar(monthly, x='month', y='_present_flag_', labels={'_present_flag_':'attendance_rate'})
        fig.update_yaxes(tickformat='.0%')
        st.plotly_chart(fig, use_container_width=True)
        with st.expander('Explanation', expanded=show_explanations):
            st.write('Average attendance per month — helpful to spot month-level trends.')

        st.subheader('Attendance heatmap (complex)')
        try:
            heat = att_filtered.pivot_table(index='Name', columns=att_filtered['Date'].dt.date, values='_present_flag_', aggfunc='mean').fillna(0)
            top_n = st.slider('Number of students to show', min_value=10, max_value=min(500, heat.shape[0]), value=min(50, heat.shape[0]))
            top_students = heat.mean(axis=1).sort_values(ascending=False).head(top_n).index
            heat_small = heat.loc[top_students]
            fig = px.imshow(heat_small, color_continuous_scale='RdYlGn', aspect='auto')
            st.plotly_chart(fig, use_container_width=True)
            with st.expander('Explanation', expanded=show_explanations):
                st.write('Heatmap: rows=students, columns=dates. Green=present, red=absent. Use to spot persistent absentees.')
        except Exception:
            st.info('Could not render heatmap — try adjusting the date range or number of students.')

# ---- Insights & Export ----
with tabs[4]:
    st.header('Insights & Export')
    low_att_thresh = min_att_pct/100.0
    low_score_thresh = score_min

    if not student_mark_summary.empty:
        student_level = student_mark_summary.copy()
        if not att_summary.empty:
            student_level = student_level.merge(att_summary[['ID','attendance_rate']], on='ID', how='left')
        student_level['flag_low_attendance'] = student_level['attendance_rate'].fillna(1) < low_att_thresh
        student_level['flag_low_score'] = student_level['avg_score'] < low_score_thresh
        flagged = student_level[student_level['flag_low_attendance'] | student_level['flag_low_score']]

        st.subheader('Flagged students (table)')
        st.dataframe(flagged[['ID','Roll','Name','avg_score','attendance_rate','flag_low_attendance','flag_low_score']])
        with st.expander('Explanation', expanded=show_explanations):
            st.write('Students flagged for low attendance or low average scores. Use this list to prioritize interventions.')

        st.markdown('**Export flagged list**')
        buf_csv = io.StringIO()
        flagged.to_csv(buf_csv, index=False)
        st.download_button('Download CSV', data=buf_csv.getvalue().encode('utf-8'), file_name='flagged_students.csv', mime='text/csv')

        # Excel export
        buf_xl = io.BytesIO()
        with pd.ExcelWriter(buf_xl, engine='openpyxl') as writer:
            flagged.to_excel(writer, index=False, sheet_name='Flagged')
        st.download_button('Download Excel', data=buf_xl.getvalue(), file_name='flagged_students.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    else:
        st.info('Not enough student-level data to compute insights.')

# ---------- Report generation functions ----------
def fig_to_image_bytes(fig, format='png', scale=2):
    # Try to export plotly fig to image bytes
    try:
        img_bytes = fig.to_image(format=format, scale=scale)
        return img_bytes
    except Exception:
        try:
            # alternative
            return fig.to_image(format=format)
        except Exception:
            return None


def generate_student_excel_report(student_name, marks_df, att_df):
    # Create Excel in-memory
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine='openpyxl') as writer:
        s_marks = marks_df[marks_df['Name']==student_name]
        s_att = att_df[att_df['Name']==student_name] if not att_df.empty else pd.DataFrame()
        s_summary = s_marks.groupby('Subject')['Marks'].agg(['mean','count']).reset_index().rename(columns={'mean':'avg_score','count':'entries'})
        s_marks.to_excel(writer, sheet_name='Marks', index=False)
        s_att.to_excel(writer, sheet_name='Attendance', index=False)
        s_summary.to_excel(writer, sheet_name='Summary', index=False)
    out.seek(0)
    return out


def generate_student_pdf_report(student_name, marks_df, att_df):
    # Create a simple PDF with text and chart images
    s_marks = marks_df[marks_df['Name']==student_name]
    s_att = att_df[att_df['Name']==student_name] if not att_df.empty else pd.DataFrame()

    tmpf = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
    doc = SimpleDocTemplate(tmpf.name, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph(f'Student Report — {student_name}', styles['Title']))
    story.append(Spacer(1,12))

    # Basic text summary
    if not s_marks.empty:
        avg_score = s_marks['Marks'].mean()
        story.append(Paragraph(f'Average score: {avg_score:.2f}', styles['Normal']))
    if not s_att.empty:
        present = int(s_att['_present_flag_'].sum())
        total = int(s_att.shape[0])
        rate = present/total if total>0 else 0
        story.append(Paragraph(f'Attendance: {present}/{total} ({rate:.1%})', styles['Normal']))
    story.append(Spacer(1,12))

    # Add charts as images (try to generate)
    imgs = []
    try:
        if not s_marks.empty:
            subj_avg = s_marks.groupby('Subject')['Marks'].mean().reset_index()
            fig = px.bar(subj_avg, x='Subject', y='Marks', color='Subject', color_discrete_map=SUBJECT_COLORS)
            img = fig_to_image_bytes(fig)
            if img:
                fimg = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
                fimg.write(img)
                fimg.flush()
                imgs.append(fimg.name)
        if not s_att.empty:
            s_att_sorted = s_att.sort_values('Date')
            fig2 = px.scatter(s_att_sorted, x='Date', y='_present_flag_')
            img2 = fig_to_image_bytes(fig2)
            if img2:
                fimg2 = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
                fimg2.write(img2)
                fimg2.flush()
                imgs.append(fimg2.name)
    except Exception:
        pass

    for p in imgs:
        try:
            story.append(RLImage(p, width=450, height=200))
            story.append(Spacer(1,12))
        except Exception:
            continue

    doc.build(story)
    with open(tmpf.name, 'rb') as f:
        pdf_bytes = f.read()
    return pdf_bytes

# ---------- End of app ----------

st.caption('Right iTech — interactive student insights. Contact developer to add custom grading rules, notifications or school branding.')
