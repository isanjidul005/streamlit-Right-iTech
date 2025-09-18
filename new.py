# superduper_streamlit_app.py
# Streamlit App: Super Duper Student Insights
# Single-file Streamlit app that supports uploading two CSVs (marks and attendance)
# If no upload provided, example files bundled on server will be used (/mnt/data/cleanest_marks.csv and /mnt/data/combined_attendance.csv)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title='SuperDuper Student Insights', layout='wide', initial_sidebar_state='expanded')

# --- Styling for a modern look ---
st.markdown("""
<style>
.big-font {font-size:28px; font-weight:700}
.small-muted {color: #6c6c6c}
.card {background: linear-gradient(180deg, rgba(255,255,255,0.85), rgba(245,245,250,0.85)); padding: 16px; border-radius: 12px; box-shadow: 0 6px 18px rgba(15,15,30,0.06);}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="big-font">📊 SuperDuper Student Insights</div>', unsafe_allow_html=True)
st.markdown('<div class="small-muted">Upload your marks and attendance CSVs (or use the sample files). Interactive filters, many visualizations and downloadable reports.</div>', unsafe_allow_html=True)
st.write('---')

# --- File upload section ---
with st.expander('Upload datasets (optional)'):
    col1, col2 = st.columns(2)
    with col1:
        marks_file = st.file_uploader('Upload marks CSV', type=['csv'], key='marks_upload')
    with col2:
        att_file = st.file_uploader('Upload attendance CSV', type=['csv'], key='att_upload')

# --- Data loader with fallback to bundled files ---
@st.cache_data
def load_data(marks_file, att_file):
    # Load marks
    if marks_file is not None:
        marks = pd.read_csv(marks_file)
    else:
        try:
            marks = pd.read_csv('/mnt/data/cleanest_marks.csv')
        except Exception:
            marks = pd.DataFrame()
    # Load attendance
    if att_file is not None:
        att = pd.read_csv(att_file)
    else:
        try:
            att = pd.read_csv('/mnt/data/combined_attendance.csv')
        except Exception:
            att = pd.DataFrame()
    return marks, att

marks_df, att_df = load_data(marks_file, att_file)

# quick sanity checks
if marks_df.empty and att_df.empty:
    st.error('No data available. Please upload at least one CSV (marks or attendance).')
    st.stop()

# --- Try to unify column names a bit (common patterns) ---
def normalize_cols(df):
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    return df

marks_df = normalize_cols(marks_df)
att_df = normalize_cols(att_df)

# Infer common fields
# We'll try to find student id and name columns
possible_id_cols = ['student_id','id','roll','roll_no','rollno','roll number']
possible_name_cols = ['student_name','name','student','fullname','full_name']

def find_col(df, candidates):
    for c in df.columns:
        if c.lower().replace(' ','_') in candidates:
            return c
    # fallback: substring match
    for cand in candidates:
        for c in df.columns:
            if cand in c.lower():
                return c
    return None

marks_id = find_col(marks_df, [c.lower() for c in possible_id_cols])
marks_name = find_col(marks_df, [c.lower() for c in possible_name_cols])
att_id = find_col(att_df, [c.lower() for c in possible_id_cols])
att_name = find_col(att_df, [c.lower() for c in possible_name_cols])

# best-guess names for UI
student_id_col = marks_id or att_id
student_name_col = marks_name or att_name

# If still missing, try common heuristics
if student_name_col is None:
    # pick any text column with "name" in it
    for c in list(marks_df.columns) + list(att_df.columns):
        if 'name' in c.lower():
            student_name_col = c
            break

# Convert date columns if present in attendance
for c in att_df.columns:
    if 'date' in c.lower():
        try:
            att_df[c] = pd.to_datetime(att_df[c])
        except Exception:
            pass

# Create derived attendance metrics
if not att_df.empty:
    # Expect columns: student, date, status/present
    # heuristics
    present_col = None
    for c in att_df.columns:
        if any(k in c.lower() for k in ['present','status','is_present','attend']):
            present_col = c
            break
    if present_col is None:
        # if there are only 2 columns aside from date and id, assume third is present
        possible = [c for c in att_df.columns if c not in [student_id_col, student_name_col]]
        if len(possible) == 1:
            present_col = possible[0]

    # Standardize present to 1/0
    if present_col is not None:
        def to_present(x):
            if pd.isna(x):
                return np.nan
            s = str(x).strip().lower()
            if s in ['1','yes','y','present','p','true','t']:
                return 1
            if s in ['0','no','n','absent','a','false','f']:
                return 0
            try:
                return int(float(s))
            except Exception:
                return np.nan
        att_df['_present_flag_'] = att_df[present_col].apply(to_present)
    else:
        att_df['_present_flag_'] = np.nan

    # group by student to compute attendance rate
    if student_id_col in att_df.columns:
        att_summary = att_df.groupby(student_id_col)['_present_flag_'].agg(['count','sum']).reset_index().rename(columns={'count':'total_sessions','sum':'present_count'})
        att_summary['attendance_rate'] = att_summary['present_count']/att_summary['total_sessions']
    else:
        att_summary = pd.DataFrame()
else:
    att_summary = pd.DataFrame()

# Marks processing - try to melt wide exam columns into tidy long format
marks_long = None
if not marks_df.empty:
    # find numeric columns (scores) and exam/date columns heuristics
    numeric_cols = marks_df.select_dtypes(include=[np.number]).columns.tolist()
    # if numeric columns include student id, remove
    if student_id_col in numeric_cols:
        numeric_cols.remove(student_id_col)
    # treat all numeric columns except maybe attendance-like as subject/exam marks
    if len(numeric_cols) >= 1:
        # if there is 'total' or 'percentage' prefer those for quick overviews
        marks_long = marks_df.copy()
        # If there are subject columns, melt them
        subject_like = [c for c in marks_df.columns if c not in [student_id_col, student_name_col] and marks_df[c].dtype in [np.float64, np.int64]]
        if len(subject_like) > 1:
            marks_long = marks_df.melt(id_vars=[c for c in [student_id_col, student_name_col] if c in marks_df.columns], value_vars=subject_like, var_name='assessment', value_name='score')
        else:
            # single numeric column or many non-numeric - keep as-is
            marks_long = marks_df.copy()

# Merge marks and attendance summary for class-level insights
if not marks_df.empty and not att_summary.empty and student_id_col is not None:
    merged = marks_df.merge(att_summary[[student_id_col,'attendance_rate']], on=student_id_col, how='left')
else:
    merged = marks_df.copy() if not marks_df.empty else pd.DataFrame()

# Sidebar filters
st.sidebar.header('Filters & Controls')
with st.sidebar.expander('Filter options'):
    # Class/section filter if exists
    class_col = None
    for c in marks_df.columns:
        if any(x in c.lower() for x in ['class','grade','section','group']):
            class_col = c
            break
    selected_class = None
    if class_col and class_col in marks_df.columns:
        selected_class = st.sidebar.multiselect('Select class/section', options=sorted(marks_df[class_col].dropna().unique().tolist()), default=sorted(marks_df[class_col].dropna().unique().tolist()))
    # Student selector (single and multi) - build list from name or id
    student_options = []
    if student_name_col and student_name_col in marks_df.columns:
        student_options = marks_df[student_name_col].astype(str).tolist()
    elif student_id_col and student_id_col in marks_df.columns:
        student_options = marks_df[student_id_col].astype(str).tolist()
    selected_students = st.sidebar.multiselect('Select students (for comparison)', options=sorted(set(student_options)), max_selections=10)

    # Slider for minimum attendance filter
    min_att = st.sidebar.slider('Minimum attendance rate (%)', min_value=0, max_value=100, value=0)

    # Score range slider if score column exists
    score_min, score_max = 0, 100
    if 'score' in marks_long.columns if marks_long is not None else False:
        try:
            score_min = int(np.floor(marks_long['score'].min()))
            score_max = int(np.ceil(marks_long['score'].max()))
        except Exception:
            score_min, score_max = 0, 100
    selected_score_range = st.sidebar.slider('Score range', min_value=score_min, max_value=score_max, value=(score_min, score_max))

# --- Layout: Tabs for different overviews ---
tabs = st.tabs(['Class overview','Single student','Compare students','Attendance explorer','Advanced insights & exports'])

# --- Tab 1: Class overview ---
with tabs[0]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header('Class overview')
    col1, col2, col3 = st.columns([3,2,2])
    with col1:
        st.subheader('Marks distribution')
        if marks_long is not None and 'score' in marks_long.columns:
            fig = px.histogram(marks_long, x='score', nbins=25, marginal='box', title='Score distribution (all assessments)')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info('No numeric marks found to plot distribution.')
    with col2:
        st.subheader('Attendance snapshot')
        if not att_summary.empty:
            top = att_summary.sort_values('attendance_rate', ascending=False).head(5)
            st.table(top.style.format({'attendance_rate': '{:.1%}'}))
        else:
            st.info('No attendance data available.')
    with col3:
        st.subheader('Class summary stats')
        if not merged.empty:
            mean_score = merged.select_dtypes(include=[np.number]).mean(numeric_only=True).median()
            st.metric('Median of numeric fields (quick glance)', f"{mean_score:.2f}")
        else:
            st.info('No merged data for quick stats.')
    st.markdown('</div>', unsafe_allow_html=True)

    # Correlation heatmap for marks (if subjects exist)
    st.subheader('Correlation across assessments/subjects')
    if marks_df.select_dtypes(include=[np.number]).shape[1] > 1:
        num_df = marks_df.select_dtypes(include=[np.number])
        corr = num_df.corr()
        fig = px.imshow(corr, text_auto=True, title='Correlation matrix')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info('Not enough numeric columns to compute correlation matrix.')

    # Top/Bottom performers
    st.subheader('Top & bottom performers (by total / average)')
    if not marks_df.empty:
        numeric_cols = marks_df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            marks_df['_avg_'] = marks_df[numeric_cols].mean(axis=1)
            top5 = marks_df.sort_values('_avg_', ascending=False).head(5)
            bot5 = marks_df.sort_values('_avg_', ascending=True).head(5)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown('**Top 5 (by average score)**')
                st.table(top5[[student_name_col, '_avg_']].rename(columns={student_name_col:'Student','_avg_':'Average Score'}).set_index('Student'))
            with c2:
                st.markdown('**Bottom 5 (by average score)**')
                st.table(bot5[[student_name_col, '_avg_']].rename(columns={student_name_col:'Student','_avg_':'Average Score'}).set_index('Student'))
        else:
            st.info('No numeric marks to compute averages.')

# --- Tab 2: Single student overview ---
with tabs[1]:
    st.header('Single Student Overview')
    st.markdown('Choose a student from the dropdown and explore their marks and attendance timeline.')
    # single student selection
    single_student = None
    if student_options:
        single_student = st.selectbox('Select student for single overview', options=sorted(set(student_options)))
    else:
        st.info('No student names detected in marks dataset; try uploading a marks file with a name column.')

    if single_student:
        # find student row(s)
        # match by name first, else by id
        s_rows = marks_df[marks_df[student_name_col].astype(str) == single_student] if student_name_col in marks_df.columns else marks_df[marks_df[student_id_col].astype(str) == single_student]
        st.subheader(f'Profile: {single_student}')
        st.write(s_rows.dropna(axis=1, how='all').T)

        # student marks timeline (if assessment/date present)
        st.subheader('Marks timeline')
        if marks_long is not None and student_name_col in marks_long.columns:
            student_marks = marks_long[marks_long[student_name_col].astype(str) == single_student]
            if 'assessment' in student_marks.columns:
                fig = px.line(student_marks, x='assessment', y='score', markers=True, title='Scores across assessments')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info('Marks data is not in assessment-level format.')
        else:
            st.info('No detailed marks long-format data available.')

        # attendance for student
        st.subheader('Attendance timeline')
        if not att_df.empty and student_name_col in att_df.columns:
            s_att = att_df[att_df[student_name_col].astype(str) == single_student]
            date_cols = [c for c in s_att.columns if 'date' in c.lower() or np.issubdtype(s_att[c].dtype, np.datetime64)]
            if date_cols:
                date_col = date_cols[0]
                fig = px.scatter(s_att, x=date_col, y='_present_flag_', title='Attendance over time', labels={'_present_flag_':'Present (1) / Absent (0)'})
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info('No date field detected in attendance data for timeline plot.')
        else:
            st.info('No attendance available for this student.')

# --- Tab 3: Compare students ---
with tabs[2]:
    st.header('Compare Students')
    st.markdown('Pick up to 10 students to compare. Uses radar, grouped bar, and scatter for multi-dimensional comparison.')
    if selected_students:
        # get data for selected
        def get_student_key(s):
            if student_name_col and student_name_col in marks_df.columns and s in marks_df[student_name_col].astype(str).tolist():
                return marks_df[marks_df[student_name_col].astype(str) == s]
            return marks_df
        comp_df = marks_df[marks_df[student_name_col].astype(str).isin(selected_students)] if student_name_col in marks_df.columns else pd.DataFrame()
        if not comp_df.empty:
            # build aggregate metrics
            numeric_cols = comp_df.select_dtypes(include=[np.number]).columns.tolist()
            comp_metrics = comp_df[[student_name_col] + numeric_cols].copy()
            comp_metrics['_avg_'] = comp_metrics[numeric_cols].mean(axis=1)
            st.subheader('Grouped bar: average by assessment (selected students)')
            avg_df = comp_df.set_index(student_name_col)[numeric_cols].T
            if not avg_df.empty:
                fig = go.Figure()
                for s in selected_students:
                    if s in comp_df[student_name_col].astype(str).tolist():
                        row = comp_df[comp_df[student_name_col].astype(str)==s]
                        vals = row[numeric_cols].mean(axis=0).values
                        fig.add_trace(go.Bar(name=s, x=numeric_cols, y=vals))
                fig.update_layout(barmode='group', title='Average scores across numeric assessments')
                st.plotly_chart(fig, use_container_width=True)

            # radar chart (normalize)
            st.subheader('Radar: normalized profile')
            try:
                categories = numeric_cols[:8]
                fig = go.Figure()
                for i, r in comp_df.iterrows():
                    vals = r[categories].fillna(0).values.tolist()
                    # normalize
                    mx = np.nanmax(comp_df[categories].values) if comp_df[categories].values.size else 1
                    if mx==0: mx=1
                    vals = [v/mx for v in vals]
                    fig.add_trace(go.Scatterpolar(r=vals, theta=categories, fill='toself', name=str(r[student_name_col])))
                fig.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=True, title='Normalized skill/profile radar')
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                st.info('Not enough numeric columns for radar.')

            # scatter: attendance vs avg score
            if not att_summary.empty and student_id_col in att_summary.columns:
                # build a mapping for names to ids if possible
                merged_for_comp = comp_df.merge(att_summary[[student_id_col,'attendance_rate']], on=student_id_col, how='left') if student_id_col in comp_df.columns else pd.DataFrame()
                if not merged_for_comp.empty:
                    merged_for_comp['_avg_'] = merged_for_comp.select_dtypes(include=[np.number]).mean(axis=1)
                    fig = px.scatter(merged_for_comp, x='attendance_rate', y='_avg_', text=student_name_col, size='_avg_', title='Attendance vs average score')
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info('Cannot find selected students in marks dataset. Ensure names match exactly.')
    else:
        st.info('Select students in the sidebar to compare.')

# --- Tab 4: Attendance explorer ---
with tabs[3]:
    st.header('Attendance Explorer')
    if not att_df.empty:
        # allow date range selection if date column exists
        date_cols = [c for c in att_df.columns if 'date' in c.lower() or np.issubdtype(att_df[c].dtype, np.datetime64)]
        if date_cols:
            date_col = date_cols[0]
            min_date = att_df[date_col].min()
            max_date = att_df[date_col].max()
            date_range = st.date_input('Select date range', value=(min_date, max_date))
            if len(date_range) == 2:
                sdate, edate = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
                mask = (att_df[date_col] >= sdate) & (att_df[date_col] <= edate)
                filtered = att_df[mask]
            else:
                filtered = att_df.copy()
        else:
            filtered = att_df.copy()

        # heatmap of attendance by student vs date
        st.subheader('Attendance heatmap (students × dates)')
        try:
            heat = filtered.pivot_table(index=student_name_col, columns=date_col, values='_present_flag_', aggfunc='mean')
            fig = px.imshow(heat.fillna(0).values, x=heat.columns.astype(str), y=heat.index.astype(str), aspect='auto', title='Attendance heatmap (mean present flag)')
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            st.info('Could not construct attendance heatmap. Ensure attendance has student name/id and date columns.')

        st.subheader('Attendance trends')
        # overall attendance rate over time
        try:
            att_over_time = filtered.groupby(date_col)['_present_flag_'].mean().reset_index()
            fig = px.line(att_over_time, x=date_col, y='_present_flag_', title='Class average attendance over time')
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            st.info('No suitable date/present columns to plot trends.')
    else:
        st.info('No attendance data uploaded.')

# --- Tab 5: Advanced insights & exports ---
with tabs[4]:
    st.header('Advanced insights & exports')
    st.subheader('Auto insights (flagging and recommendations)')
    # flag low attendance low performers
    if not merged.empty:
        merged['_avg_'] = merged.select_dtypes(include=[np.number]).mean(axis=1)
        low_att_mask = merged['attendance_rate'] < (min_att/100.0) if 'attendance_rate' in merged.columns else pd.Series([False]*len(merged))
        low_score_mask = merged['_avg_'] < (selected_score_range[0] if selected_score_range else merged['_avg_'].min())
        flagged = merged[low_att_mask | low_score_mask]
        if not flagged.empty:
            st.markdown(f"Found **{len(flagged)}** students flagged for attention (low attendance or low scores).")
            st.dataframe(flagged[[student_name_col,'attendance_rate','_avg_']].rename(columns={student_name_col:'Student'}))
        else:
            st.success('No students flagged based on current filters.')

    # Export buttons
    st.subheader('Download datasets & reports')
    if not marks_df.empty:
        st.download_button('Download cleaned marks (csv)', marks_df.to_csv(index=False).encode('utf-8'), file_name='cleaned_marks.csv')
    if not att_df.empty:
        st.download_button('Download cleaned attendance (csv)', att_df.to_csv(index=False).encode('utf-8'), file_name='cleaned_attendance.csv')

    st.markdown('---')
    st.markdown('**Customization tips:** Use this app as a starting point. You can add more domain-specific rules (grading scales, pass/fail thresholds, term grouping) easily by editing the file.')

# --- Footer / quick help ---
st.markdown('---')
st.markdown('<div class="small-muted">Tip: If any visualization says "not enough data", try uploading a marks CSV with clear name/id columns and numeric subject columns, and an attendance CSV with student, date and present/absent status.</div>', unsafe_allow_html=True)

# End of file
