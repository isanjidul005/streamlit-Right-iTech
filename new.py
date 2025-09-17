# Student Dashboard — Robust Interactive Streamlit App
# File: student_dashboard_streamlit.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO

st.set_page_config(page_title="Student Dashboard — Robust", layout="wide")

# ----------------------------
# Utilities: robust readers & cleaning
# ----------------------------
@st.cache_data
def read_file_guess_header(uploaded_file):
    """Read a CSV/XLSX and try to detect header row automatically.
    Returns DataFrame with header set to detected header row if possible.
    """
    if uploaded_file is None:
        return None

    name = uploaded_file.name.lower()
    try:
        # read as raw with no header to inspect top rows
        if name.endswith(('.xls', '.xlsx')):
            raw = pd.read_excel(uploaded_file, header=None, engine='openpyxl')
        else:
            # try different encodings when necessary
            try:
                raw = pd.read_csv(uploaded_file, header=None, encoding='utf-8')
            except Exception:
                uploaded_file.seek(0)
                raw = pd.read_csv(uploaded_file, header=None, encoding='latin1')
    except Exception as e:
        st.error(f"Error reading file {uploaded_file.name}: {e}")
        return None

    # Heuristic: find a row that contains likely header labels (id/name/roll)
    header_row = None
    for r in range(min(8, len(raw))):
        row_vals = raw.iloc[r].astype(str).str.lower().tolist()
        joined = ' '.join(row_vals)
        if any(k in joined for k in ['id', 'name', 'roll', 'student', 'serial']):
            header_row = r
            break

    # If we found header_row, re-read the file with that row as header (or set columns manually)
    try:
        uploaded_file.seek(0)
        if name.endswith(('.xls', '.xlsx')):
            if header_row is not None:
                df = pd.read_excel(uploaded_file, header=header_row, engine='openpyxl')
            else:
                df = pd.read_excel(uploaded_file, header=0, engine='openpyxl')
        else:
            if header_row is not None:
                df = pd.read_csv(uploaded_file, header=header_row, encoding='utf-8')
            else:
                try:
                    df = pd.read_csv(uploaded_file, header=0, encoding='utf-8')
                except Exception:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, header=0, encoding='latin1')
    except Exception as e:
        st.error(f"Failed to parse file {uploaded_file.name}: {e}")
        return None

    # Drop columns that are entirely NaN
    df = df.dropna(axis=1, how='all')
    # Drop rows that are entirely NaN
    df = df.dropna(axis=0, how='all')

    # Strip whitespace from column names
    df.columns = df.columns.astype(str).str.strip()
    return df


def ensure_id_roll_name(df, file_label="attendance"):
    """Ensure a DataFrame has ID, Roll, Name columns. Try to find closest matches and rename.
    Returns df (or None if inadequate).
    """
    if df is None:
        return None
    cols = list(df.columns)
    cols_lower = {c.lower(): c for c in cols}

    # Possible matches
    id_col = None
    for k in ['id', 'student id', 'sid', 'student_no', 'student id']:
        if k in cols_lower:
            id_col = cols_lower[k]
            break

    name_col = None
    for k in ['name', 'student name', 'full name']:
        if k in cols_lower:
            name_col = cols_lower[k]
            break

    roll_col = None
    for k in ['roll', 'roll no', 'rollno', 'roll_number', 'serial']:
        if k in cols_lower:
            roll_col = cols_lower[k]
            break

    # If not found, attempt heuristics: first three columns
    if id_col is None and len(cols) >= 1:
        id_col = cols[0]
    if roll_col is None and len(cols) >= 2:
        roll_col = cols[1]
    if name_col is None and len(cols) >= 3:
        name_col = cols[2]

    if id_col is None or name_col is None:
        st.warning(f"Could not detect ID/Name columns cleanly for {file_label}. We need at least ID and Name.")
        return None

    # Rename into canonical columns
    rename_map = {id_col: 'ID', roll_col: 'Roll', name_col: 'Name'}
    df = df.rename(columns=rename_map)
    df['Name'] = df['Name'].astype(str).str.strip()
    return df


def standardize_attendance_marks(s):
    """Convert various attendance mark types to 1/0/np.nan"""
    if pd.isna(s):
        return np.nan
    s_str = str(s).strip()
    if s_str == '':
        return np.nan
    s_up = s_str.upper()
    if s_up in ['P', 'PRESENT', 'YES', '1', '✔', 'V'] or s_str.startswith('✔'):
        return 1
    if s_up in ['A', 'ABSENT', 'NO', '0']:
        return 0
    # If numeric-like
    try:
        v = float(s_str)
        return 1 if v > 0 else 0
    except Exception:
        # fallback: check for presence of letters
        if any(ch.isalpha() for ch in s_str):
            if 'P' in s_up:
                return 1
            if 'A' in s_up:
                return 0
        return np.nan


def melt_attendance_df(df):
    # find id_vars automatically (ID, Roll, Name, Gender if present)
    id_vars = [c for c in ['ID', 'Roll', 'Name', 'Gender'] if c in df.columns]
    value_vars = [c for c in df.columns if c not in id_vars]
    if not value_vars:
        st.error('No attendance columns detected. Make sure the file contains session/date columns after ID, Roll, Name.')
        return None

    long = df.melt(id_vars=id_vars, value_vars=value_vars, var_name='Session', value_name='RawStatus')
    # parse Session to date when possible
    long['Session_parsed'] = pd.to_datetime(long['Session'], dayfirst=True, errors='coerce')
    long['Session_label'] = long['Session'].astype(str)
    long['Present'] = long['RawStatus'].apply(standardize_attendance_marks)
    return long


def detect_score_columns(df):
    # find columns containing WMT or Test or Exam or numeric suffixes
    candidates = [c for c in df.columns if any(k in c.lower() for k in ['wmt', 'test', 'exam', 'score', 'marks'])]
    # exclude ID/Roll/Name
    candidates = [c for c in candidates if c not in ['ID','Roll','Name']]
    # If nothing found, fallback to all numeric-like columns beyond first 3
    if not candidates:
        numeric_cols = []
        for c in df.columns:
            if c in ['ID','Roll','Name']:
                continue
            # sample values
            sample = df[c].dropna().astype(str).head(10).tolist()
            n_numeric = sum(1 for v in sample if any(ch.isdigit() for ch in v))
            if n_numeric >= max(1, len(sample)//2):
                numeric_cols.append(c)
        candidates = numeric_cols
    return candidates


def parse_scores_long(df, chosen_cols):
    id_vars = [c for c in ['ID','Roll','Name'] if c in df.columns]
    if not id_vars:
        st.error('Score file lacks ID/Name/Roll columns. Cannot proceed.')
        return None
    long = df.melt(id_vars=id_vars, value_vars=chosen_cols, var_name='Assessment', value_name='RawScore')
    # extract numeric from strings (handles 'ab', '95/100', etc.)
    long['Score'] = pd.to_numeric(long['RawScore'].astype(str).str.extract(r'(\d+\.?\d*)')[0], errors='coerce')
    long['Score'] = long['Score'].fillna(0)
    return long


# ----------------------------
# Analytics helpers
# ----------------------------

def compute_student_aggregates(score_long, attendance_summary):
    # avg score per student
    score_ag = score_long.groupby(['ID','Name','Roll']).Score.agg(['mean','median','std','count']).reset_index().rename(columns={'mean':'AvgScore','median':'MedianScore','std':'ScoreStd','count':'ScoreRecords'})
    # merge attendance
    merge_keys = [k for k in ['ID','Name','Roll'] if k in attendance_summary.columns]
    merged = pd.merge(score_ag, attendance_summary, how='left', on=merge_keys)
    if 'AttendanceRate' in merged.columns:
        merged['AttendanceRate'] = merged['AttendanceRate'].fillna(0)
    else:
        merged['AttendanceRate'] = 0
    # risk rule
    merged['LowAttendanceFlag'] = merged['AttendanceRate'] < 0.6
    merged['LowScoreFlag'] = merged['AvgScore'] < 40
    merged['AtRisk'] = merged['LowAttendanceFlag'] | merged['LowScoreFlag']
    # normalized score (0-1) assuming 100 max; handle bigger scales gracefully
    merged['NormScore'] = merged['AvgScore'] / merged['AvgScore'].max() if merged['AvgScore'].max() > 0 else 0
    # risk score (0-1): higher means more risk
    merged['RiskScore'] = (1 - merged['AttendanceRate'])*0.6 + (1 - merged['NormScore'])*0.4
    merged = merged.sort_values(by='RiskScore', ascending=False)
    return merged


def add_poly_trend(fig, df, x_col, y_col, group_col=None, color=None):
    # Add simple OLS trend lines using numpy.polyfit so we don't rely on statsmodels
    if group_col and group_col in df.columns:
        groups = df[group_col].dropna().unique()
        for g in groups:
            sub = df[df[group_col]==g]
            x = sub[x_col].astype(float).dropna()
            y = sub[y_col].astype(float).dropna()
            # align indices
            sub = sub.loc[x.index.intersection(y.index)]
            if len(sub) < 2:
                continue
            coeffs = np.polyfit(sub[x_col].astype(float), sub[y_col].astype(float), 1)
            xs = np.linspace(sub[x_col].min(), sub[x_col].max(), 50)
            ys = np.polyval(coeffs, xs)
            fig.add_trace(go.Scatter(x=xs, y=ys, mode='lines', name=f'Trend — {g}', line=dict(dash='dash')))
    else:
        sub = df.dropna(subset=[x_col, y_col])
        if len(sub) >= 2:
            coeffs = np.polyfit(sub[x_col].astype(float), sub[y_col].astype(float), 1)
            xs = np.linspace(sub[x_col].min(), sub[x_col].max(), 50)
            ys = np.polyval(coeffs, xs)
            fig.add_trace(go.Scatter(x=xs, y=ys, mode='lines', name='Trend', line=dict(dash='dash')))
    return fig


# ----------------------------
# App
# ----------------------------

def main():
    st.title("🎯 Student Dashboard — Robust & Interactive")

    # Sidebar: uploads + settings
    st.sidebar.header('Uploads')
    att_boys = st.sidebar.file_uploader('Boys Attendance (xlsx/csv)', type=['xlsx','csv'], key='boys')
    att_girls = st.sidebar.file_uploader('Girls Attendance (xlsx/csv)', type=['xlsx','csv'], key='girls')
    scores_file = st.sidebar.file_uploader('Scores (xlsx/csv)', type=['xlsx','csv'], key='scores')

    st.sidebar.markdown('---')
    st.sidebar.header('Cleaning options')
    normalize_names = st.sidebar.checkbox('Normalize student names (strip & title)', value=True)
    drop_blank_sessions = st.sidebar.checkbox('Drop attendance sessions with all-missing values', value=True)
    attendance_date_filtering = st.sidebar.checkbox('Enable attendance date filtering (if dates parsed)', value=True)

    if not (att_boys and att_girls and scores_file):
        st.info('Upload three files (boys attendance, girls attendance, scores) to begin. Use the sidebar.')
        return

    # Read files robustly
    df_boys_raw = read_file_guess_header(att_boys)
    df_girls_raw = read_file_guess_header(att_girls)
    df_scores_raw = read_file_guess_header(scores_file)

    # ensure we have ID/Roll/Name
    df_boys = ensure_id_roll_name(df_boys_raw, 'boys attendance')
    df_girls = ensure_id_roll_name(df_girls_raw, 'girls attendance')
    df_scores = ensure_id_roll_name(df_scores_raw, 'score file')

    if df_boys is None or df_girls is None or df_scores is None:
        st.error('One or more files could not be parsed with required columns (ID/Name). Check your files or adjust headers.')
        return

    # tag gender
    df_boys['Gender'] = 'Boy'
    df_girls['Gender'] = 'Girl'

    # Optional name normalization
    if normalize_names:
        for d in [df_boys, df_girls, df_scores]:
            d['Name'] = d['Name'].astype(str).str.strip().str.title()

    # Combine attendance and drop obviously empty session columns
    attendance_all = pd.concat([df_boys, df_girls], ignore_index=True, sort=False)
    # Identify session columns
    id_cols = [c for c in ['ID','Roll','Name','Gender'] if c in attendance_all.columns]
    session_cols = [c for c in attendance_all.columns if c not in id_cols]
    if drop_blank_sessions:
        # drop session cols that are all null/blank
        non_empty_sessions = [c for c in session_cols if not attendance_all[c].dropna().empty]
        session_cols = non_empty_sessions
        attendance_all = attendance_all[id_cols + session_cols]

    # Melt attendance
    attendance_long = melt_attendance_df(attendance_all)
    if attendance_long is None:
        st.error('Attendance could not be melted into long format')
        return

    # Attendance summary per student
    group_keys = [c for c in ['ID','Name','Roll','Gender'] if c in attendance_long.columns]
    attendance_summary = attendance_long.groupby(group_keys).Present.agg(['mean','count']).reset_index().rename(columns={'mean':'AttendanceRate','count':'SessionsRecorded'})

    # Prepare scores
    score_candidate_cols = detect_score_columns(df_scores)
    st.sidebar.markdown('---')
    st.sidebar.header('Score columns')
    chosen_score_cols = st.sidebar.multiselect('Select score columns to include (auto-detected suggestions shown)', options=score_candidate_cols, default=score_candidate_cols[:5])
    if not chosen_score_cols:
        st.error('Please select at least one score column to analyze.')
        return

    score_long = parse_scores_long(df_scores, chosen_score_cols)
    if score_long is None:
        st.error('Scores could not be parsed correctly.')
        return

    # Merge scores with attendance summary using best available keys
    merge_on = [k for k in ['ID','Roll','Name'] if k in score_long.columns and k in attendance_summary.columns]
    if not merge_on:
        st.error('No common merge keys found between scores and attendance. Ensure both files contain compatible ID/Roll/Name columns.')
        return

    combined = pd.merge(score_long, attendance_summary, how='left', on=merge_on)
    combined['AttendanceRate'] = combined['AttendanceRate'].fillna(0)

    # Student aggregates
    student_aggs = compute_student_aggregates(score_long, attendance_summary)

    # Sidebar filters for visuals
    st.sidebar.markdown('---')
    st.sidebar.header('Filters')
    genders_available = combined['Gender'].dropna().unique().tolist() if 'Gender' in combined.columns else []
    sel_genders = st.sidebar.multiselect('Gender', options=genders_available, default=genders_available)

    assessments_available = combined['Assessment'].unique().tolist() if 'Assessment' in combined.columns else combined['WMT'].unique().tolist() if 'WMT' in combined.columns else combined['Assessment'].unique().tolist() if 'Assessment' in combined.columns else chosen_score_cols
    sel_assessments = st.sidebar.multiselect('Assessments', options=assessments_available, default=assessments_available)

    score_min, score_max = st.sidebar.slider('Score range', 0.0, float(score_long['Score'].max() if not score_long['Score'].empty else 100.0), (0.0, float(score_long['Score'].max() if not score_long['Score'].empty else 100.0)))
    att_min, att_max = st.sidebar.slider('Attendance range (0-1)', 0.0, 1.0, (0.0,1.0))

    # Apply filters
    filt = combined.copy()
    if sel_genders:
        if 'Gender' in filt.columns:
            filt = filt[filt['Gender'].isin(sel_genders)]
    if sel_assessments:
        if 'Assessment' in filt.columns:
            filt = filt[filt['Assessment'].isin(sel_assessments)]
        elif 'WMT' in filt.columns:
            filt = filt[filt['WMT'].isin(sel_assessments)]
    filt = filt[(filt['Score'] >= score_min) & (filt['Score'] <= score_max) & (filt['AttendanceRate'] >= att_min) & (filt['AttendanceRate'] <= att_max)]

    # Top-level KPIs
    st.header('Overview')
    total_students = student_aggs['ID'].nunique()
    avg_score = student_aggs['AvgScore'].mean() if not student_aggs['AvgScore'].empty else 0
    avg_att = student_aggs['AttendanceRate'].mean() if 'AttendanceRate' in student_aggs.columns else 0
    at_risk_count = student_aggs['AtRisk'].sum()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric('Students', total_students)
    k2.metric('Avg Score', f"{avg_score:.2f}")
    k3.metric('Avg Attendance', f"{avg_att*100:.1f}%")
    k4.metric('At-risk', int(at_risk_count))

    # Tabs for different views
    tabs = st.tabs(['Scores & Distribution','Attendance Trends','Groups & Correlations','Student Explorer','Data & Export'])

    # --- Scores & Distribution ---
    with tabs[0]:
        st.subheader('Score distribution & breakdown')
        col1, col2 = st.columns([2,1])
        with col1:
            fig_h = px.histogram(filt, x='Score', nbins=30, marginal='box', title='Score distribution')
            st.plotly_chart(fig_h, use_container_width=True)
        with col2:
            if 'Gender' in filt.columns:
                fig_v = px.violin(filt, x='Gender', y='Score', box=True, points='all', title='By gender')
            else:
                fig_v = px.box(filt, y='Score', title='Score distribution')
            st.plotly_chart(fig_v, use_container_width=True)

        st.subheader('Attendance vs Score (interactive)')
        fig_sc = px.scatter(filt, x='AttendanceRate', y='Score', color='Gender' if 'Gender' in filt.columns else None, hover_data=['Name','ID','Roll','Assessment'], title='Attendance vs Score')
        # add trendlines by gender (or overall) using numpy polyfit
        fig_sc = add_poly_trend(fig_sc, filt, 'AttendanceRate', 'Score', group_col='Gender' if 'Gender' in filt.columns else None)
        fig_sc.update_layout(xaxis_title='Attendance rate (0-1)', yaxis_title='Score')
        st.plotly_chart(fig_sc, use_container_width=True)

        st.markdown('**Outliers**: high attendance but low score, or low attendance but high score')
        # find some outliers
        outliers = student_aggs[(student_aggs['AttendanceRate']>0.85) & (student_aggs['AvgScore']<40)].append(student_aggs[(student_aggs['AttendanceRate']<0.5) & (student_aggs['AvgScore']>75)])
        if not outliers.empty:
            st.dataframe(outliers[['ID','Name','Roll','Gender','AvgScore','AttendanceRate']].head(20).style.format({'AttendanceRate':'{:.1%}','AvgScore':'{:.1f}'}))
        else:
            st.info('No notable outliers found for current filters.')

    # --- Attendance Trends ---
    with tabs[1]:
        st.subheader('Attendance trends over sessions')
        if attendance_long['Session_parsed'].notna().any():
            # allow date range selection
            min_d = attendance_long['Session_parsed'].min()
            max_d = attendance_long['Session_parsed'].max()
            dr = st.slider('Date range', min_value=min_d.date(), max_value=max_d.date(), value=(min_d.date(), max_d.date())) if attendance_date_filtering else (min_d.date(), max_d.date())
            start_d, end_d = dr
            att_f = attendance_long[(attendance_long['Session_parsed']>=pd.to_datetime(start_d)) & (attendance_long['Session_parsed']<=pd.to_datetime(end_d))]
            trend = att_f.groupby('Session_parsed').Present.mean().reset_index().sort_values('Session_parsed')
            fig_att = px.line(trend, x='Session_parsed', y='Present', markers=True, title='Attendance rate over time')
            fig_att.update_layout(yaxis_title='Attendance rate (0-1)', xaxis_title='Date')
            st.plotly_chart(fig_att, use_container_width=True)

            # heatmap: students (rows) x dates (columns) for selected students
            st.subheader('Attendance heatmap (sample of students)')
            sample_students = st.multiselect('Pick students (IDs) to show heatmap', options=attendance_long['ID'].unique().tolist(), default=attendance_long['ID'].unique().tolist()[:20])
            if sample_students:
                hm = attendance_long[attendance_long['ID'].isin(sample_students)].copy()
                hm['date_only'] = hm['Session_parsed'].dt.date
                pivot = hm.pivot_table(index='ID', columns='date_only', values='Present', aggfunc='mean')
                fig_hm = px.imshow(pivot.fillna(0), aspect='auto', title='Attendance heatmap (rows=ID, cols=date)')
                st.plotly_chart(fig_hm, use_container_width=True)
        else:
            st.info('Session columns could not be parsed as dates. Showing session-label trend.')
            trend = attendance_long.groupby('Session_label').Present.mean().reset_index()
            fig_att = px.bar(trend, x='Session_label', y='Present', title='Attendance by session label')
            st.plotly_chart(fig_att, use_container_width=True)

    # --- Groups & Correlations ---
    with tabs[2]:
        st.subheader('Group comparisons & correlations')
        if 'Gender' in combined.columns:
            gender_summary = combined.groupby('Gender').agg(AvgScore=('Score','mean'), AvgAttendance=('AttendanceRate','mean'), Count=('ID','nunique')).reset_index()
            st.dataframe(gender_summary.style.format({'AvgScore':'{:.1f}','AvgAttendance':'{:.1%}'}))
            fig_g = px.bar(gender_summary, x='Gender', y=['AvgScore','AvgAttendance'], barmode='group', title='Gender summary')
            st.plotly_chart(fig_g, use_container_width=True)

        st.subheader('Correlation matrix (assessments & attendance)')
        # make wide table of assessments
        wide = filt.pivot_table(index=['ID','Name','Roll'], columns='Assessment', values='Score', aggfunc='mean').reset_index()
        numeric = wide.select_dtypes(include=[np.number]).fillna(0)
        if numeric.shape[1] > 1:
            corr = numeric.corr()
            fig_corr = px.imshow(corr, text_auto=True, aspect='auto', title='Correlation matrix')
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info('Not enough numeric assessment columns to compute correlation matrix.')

    # --- Student Explorer ---
    with tabs[3]:
        st.subheader('Student explorer')
        # allow search by ID or name
        chooser = st.selectbox('Choose student by ID - Name', options=student_aggs.apply(lambda r: f"{r.ID} - {r.Name}", axis=1).tolist())
        if chooser:
            sel_id = chooser.split(' - ')[0]
            student_records = combined[combined['ID'].astype(str) == str(sel_id)].copy()
            st.markdown(f"**{student_records['Name'].iloc[0]}** — ID: {student_records['ID'].iloc[0]} — Roll: {student_records['Roll'].iloc[0]} — Gender: {student_records['Gender'].iloc[0] if 'Gender' in student_records.columns else 'N/A'}")
            st.dataframe(student_records[['Assessment','Score','AttendanceRate']].sort_values('Assessment'))
            fig_s = px.bar(student_records, x='Assessment', y='Score', title='Student scores by assessment', text='Score')
            st.plotly_chart(fig_s, use_container_width=True)
            # attendance mini-trend for that student
            st.subheader('Attendance timeline for this student')
            s_att = attendance_long[attendance_long['ID'].astype(str) == str(sel_id)].dropna(subset=['Session_parsed'])
            if not s_att.empty:
                s_tr = s_att.groupby('Session_parsed').Present.mean().reset_index()
                fig_st = px.line(s_tr, x='Session_parsed', y='Present', markers=True, title='Attendance over time')
                st.plotly_chart(fig_st, use_container_width=True)
            else:
                st.info('No parsed session dates available for this student; check session label data.')

    # --- Data & Export ---
    with tabs[4]:
        st.subheader('Data snapshots')
        with st.expander('Preview: Attendance long (first 200 rows)'):
            st.dataframe(attendance_long.head(200))
        with st.expander('Preview: Scores long (first 200 rows)'):
            st.dataframe(score_long.head(200))
        with st.expander('Student aggregates (at-risk sorted)'):
            st.dataframe(student_aggs.head(200).style.format({'AttendanceRate':'{:.1%}','AvgScore':'{:.1f}','RiskScore':'{:.3f}'}))

        # CSV export
        export_df = combined.copy()
        csv_bytes = export_df.to_csv(index=False).encode('utf-8')
        st.download_button('Download filtered combined CSV', data=csv_bytes, file_name='student_combined_filtered.csv', mime='text/csv')

        # Excel export of key sheets
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            student_aggs.to_excel(writer, sheet_name='StudentAggregates', index=False)
            attendance_summary.to_excel(writer, sheet_name='AttendanceSummary', index=False)
            score_long.to_excel(writer, sheet_name='ScoresLong', index=False)
        st.download_button('Download workbook (xlsx)', data=output.getvalue(), file_name='student_dashboard_export.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

    st.success('Dashboard ready. Use the filters and tabs to explore. Ask me to add class-level rollups, predictive flags, or prettier layout.')


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        st.exception(e)
