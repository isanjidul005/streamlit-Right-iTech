import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date
import io
import re

# Set page configuration
st.set_page_config(
    page_title="Student Performance Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
def load_css():
    st.markdown("""
    <style>
    .main-header {font-size: 24px; color: #1f77b4; font-weight: bold;}
    .metric-card {background-color: #f0f2f6; padding: 15px; border-radius: 10px;}
    .stMetric {background-color: white; padding: 10px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);}
    </style>
    """, unsafe_allow_html=True)

load_css()

# ---------- Helper functions ----------

def safe_read_csv(file):
    # Reset pointer then read
    try:
        file.seek(0)
    except Exception:
        pass
    return pd.read_csv(file)

def safe_read_excel(file):
    try:
        file.seek(0)
    except Exception:
        pass
    return pd.read_excel(file)

def read_data_file(file):
    """Read CSV or Excel files and return DataFrame. More flexible header handling."""
    try:
        # file might be UploadedFile or path-like object with .name
        fname = getattr(file, "name", str(file))
        ext = fname.split('.')[-1].lower()
        # Some files may have header on first row; avoid forcing skiprows=1 unconditionally.
        if ext == 'csv':
            try:
                df = safe_read_csv(file)
            except Exception as e:
                # try with different encodings / engine fallback
                df = pd.read_csv(file, engine='python', error_bad_lines=False)
            return df
        elif ext in ['xlsx', 'xls']:
            df = safe_read_excel(file)
            return df
        else:
            st.error(f"Unsupported file format: {fname}")
            return None
    except Exception as e:
        st.error(f"Error reading file {getattr(file,'name',str(file))}: {str(e)}")
        return None

def is_date_like(value):
    """Return True if value (string/object) can be parsed as a date."""
    try:
        # Pandas is quite flexible
        parsed = pd.to_datetime(value, errors='coerce')
        return not pd.isna(parsed)
    except Exception:
        return False

def natural_wmt_key(wmt):
    """Extract numeric part from WMT like 'WMT W1 [30]' or 'W1' to allow better ordering."""
    if pd.isna(wmt):
        return 0
    s = str(wmt)
    m = re.search(r'(\d+)', s)
    return int(m.group(1)) if m else 0

def standardize_status(raw):
    """Normalize a variety of attendance status representations into 'Present'/'Absent'/np.nan"""
    if pd.isna(raw):
        return np.nan
    s = str(raw).strip()
    # common symbols
    if '✔' in s or s.lower().startswith('p') or re.match(r'^[1]\b', s):
        return 'Present'
    if '✘' in s or s.lower().startswith('a') or re.match(r'^[0]\b', s):
        return 'Absent'
    # words
    if re.search(r'present', s, re.IGNORECASE):
        return 'Present'
    if re.search(r'absent', s, re.IGNORECASE):
        return 'Absent'
    # fallback: if contains non-empty cell and not clearly absent, treat as Present
    if s != '' and s.lower() not in ['nan', 'none', '']:
        return 'Present'
    return np.nan

# ---------- Data processing (cached) ----------

@st.cache_data
def process_uploaded_files(attendance_files, score_file):
    # attendance_files: list of UploadedFile
    attendance_dfs = []
    
    for file in attendance_files:
        df = read_data_file(file)
        if df is None:
            continue

        # Ensure columns are strings
        df.columns = df.columns.astype(str)

        # If ID/Name are missing, try rename heuristically
        if 'ID' not in df.columns or 'Name' not in df.columns:
            st.warning(f"Could not find ID/Name columns in {file.name}. Trying to infer.")
            if len(df.columns) >= 3:
                df = df.rename(columns={df.columns[0]: 'ID', df.columns[1]: 'Roll', df.columns[2]: 'Name'})
            elif len(df.columns) >= 2:
                df = df.rename(columns={df.columns[0]: 'ID', df.columns[1]: 'Name'})
            else:
                st.error(f"File {file.name} doesn't have enough columns.")
                continue

        # Ensure 'Roll' column exists for consistency (may be absent)
        if 'Roll' not in df.columns:
            df['Roll'] = np.nan

        # Detect date-like columns:
        non_date_candidates = {'ID', 'Name', 'Gender', 'Roll'}
        # Treat columns as date columns if the column name itself can be parsed as date
        date_columns = []
        for col in df.columns:
            if col in non_date_candidates:
                continue
            if is_date_like(col):
                date_columns.append(col)
                continue
            # if header isn't a date-like string, check first few values for checkmarks / P/A patterns
            sample_vals = df[col].dropna().astype(str).head(10).tolist()
            # if sample contains checkmarks or P/A or Present/Absent text -> treat as date column
            if any(re.search(r'✔|✘|\bP\b|\bA\b|Present|Absent', v, re.IGNORECASE) for v in sample_vals):
                date_columns.append(col)

        # If still nothing, try assume all columns after Name/ID are dates
        if not date_columns:
            possible = [c for c in df.columns if c not in non_date_candidates]
            if possible:
                date_columns = possible

        # Try to set Gender by filename if possible
        fname = file.name.lower()
        if 'boy' in fname or 'male' in fname:
            df['Gender'] = 'Boy'
        elif 'girl' in fname or 'female' in fname:
            df['Gender'] = 'Girl'
        elif 'b' in fname and 'g' not in fname and 'boy' in fname:  # fallback
            df['Gender'] = 'Boy'
        else:
            # keep existing Gender if available, otherwise Unknown
            if 'Gender' not in df.columns:
                df['Gender'] = 'Unknown'
            else:
                df['Gender'] = df['Gender'].fillna('Unknown')

        # Keep only the relevant columns (ID, Roll, Name, Gender + date columns)
        keep_cols = ['ID', 'Roll', 'Name', 'Gender'] + date_columns
        df = df[[c for c in keep_cols if c in df.columns]]

        attendance_dfs.append(df)

    if not attendance_dfs:
        return None, None, None

    attendance = pd.concat(attendance_dfs, ignore_index=True)

    # Identify non-date columns and date columns again for combined df
    non_date_columns = ['ID', 'Roll', 'Name', 'Gender']
    candidate_date_columns = [c for c in attendance.columns if c not in non_date_columns]

    if not candidate_date_columns:
        return None, None, None

    # Melt attendance into long format
    attendance_long = attendance.melt(
        id_vars=['ID', 'Roll', 'Name', 'Gender'],
        value_vars=candidate_date_columns,
        var_name='Date',
        value_name='RawStatus'
    )

    # Standardize status values
    attendance_long['Status'] = attendance_long['RawStatus'].apply(standardize_status)

    # Try parse Date column robustly:
    # If Date column values are strings like 'Jan 1 Wed' or real datetimes, try flexible parsing
    # First try parse column headings (they were used as var_name)
    attendance_long['Date_parsed'] = pd.to_datetime(attendance_long['Date'], errors='coerce', dayfirst=False)

    # For any rows still NaT, try to parse using more flexible strategies (e.g., if Date is e.g., 'Jan 01 (Wed)')
    mask_na = attendance_long['Date_parsed'].isna()
    if mask_na.any():
        # try to clean string and parse again
        def try_parse_date_str(s):
            try:
                s = str(s)
                # remove bracketed parts
                s = re.sub(r'[\(\)\[\]]', ' ', s)
                s = re.sub(r'\s+', ' ', s).strip()
                return pd.to_datetime(s, errors='coerce', dayfirst=False)
            except Exception:
                return pd.NaT
        attendance_long.loc[mask_na, 'Date_parsed'] = attendance_long.loc[mask_na, 'Date'].apply(try_parse_date_str)

    # Drop rows without parsed dates or without status
    attendance_long = attendance_long.dropna(subset=['Date_parsed', 'Status'])

    # Rename date column
    attendance_long = attendance_long.rename(columns={'Date_parsed': 'Date'}).drop(columns=['RawStatus', 'Date'])

    # Normalize column types
    attendance_long['ID'] = attendance_long['ID'].astype(str).str.strip()
    attendance_long['Name'] = attendance_long['Name'].astype(str).str.strip()

    # ---------------- Score file ----------------
    score_df = read_data_file(score_file)
    if score_df is None:
        return None, None, None

    # Convert columns to string names
    score_df.columns = score_df.columns.astype(str)

    # If ID/Name missing attempt heuristics
    if 'ID' not in score_df.columns or 'Name' not in score_df.columns:
        st.warning(f"Could not find ID/Name columns in score file. Trying to infer.")
        if len(score_df.columns) >= 3:
            score_df = score_df.rename(columns={score_df.columns[0]: 'ID', score_df.columns[2]: 'Name'})
        elif len(score_df.columns) >= 2:
            score_df = score_df.rename(columns={score_df.columns[0]: 'ID', score_df.columns[1]: 'Name'})
        else:
            return None, None, None

    score_df['ID'] = score_df['ID'].astype(str).str.strip()
    score_df['Name'] = score_df['Name'].astype(str).str.strip()

    # Drop irrelevant columns (Total, Merit)
    score_df = score_df.drop(columns=[col for col in score_df.columns if re.search(r'\bTotal\b|\bMerit\b', col, re.IGNORECASE)], errors='ignore')

    # Determine score columns
    score_columns = [c for c in score_df.columns if c not in ['ID', 'Name', 'Roll']]

    if not score_columns:
        return None, None, None

    # Clean score columns -> numeric
    for col in score_columns:
        # Extract the first numeric token (allow decimals)
        score_df[col] = pd.to_numeric(score_df[col].astype(str).str.extract(r'(-?\d+\.?\d*)', expand=False).fillna(0), errors='coerce').fillna(0)

    # Melt scores to long format
    wmt_long = score_df.melt(id_vars=['ID', 'Name'], value_vars=score_columns, var_name='WMT', value_name='Score')

    # Merge attendance and scores on ID + Name
    merged_df = pd.merge(wmt_long, attendance_long, on=['ID', 'Name'], how='outer')

    # Ensure Date column is dtype datetime
    merged_df['Date'] = pd.to_datetime(merged_df['Date'], errors='coerce')

    return merged_df, score_df, attendance_long

# ---------- Main App UI ----------

def main():
    st.title("📊 Student Performance Dashboard")

    st.sidebar.header("Upload Files")

    attendance_files = st.sidebar.file_uploader(
        "Upload Attendance Files",
        type=['csv', 'xlsx', 'xls'],
        accept_multiple_files=True,
        help="Upload one or more attendance files (CSV or Excel)"
    )

    score_file = st.sidebar.file_uploader(
        "Upload WMT Scores File",
        type=['csv', 'xlsx', 'xls'],
        help="Upload the WMT scores file (CSV or Excel)"
    )

    # If the user previously uploaded files via the developer console (local paths), they aren't accessible here.
    if not attendance_files:
        st.info("👆 Please upload attendance files to begin")
        st.stop()

    if not score_file:
        st.info("👆 Please upload WMT scores file to begin")
        st.stop()

    with st.spinner("Processing your files..."):
        try:
            df, wmt_scores, attendance_long = process_uploaded_files(attendance_files, score_file)
            if df is None:
                st.error("Failed to process files. Please check your file formats and headers.")
                st.stop()
        except Exception as e:
            st.error(f"Error processing files: {str(e)}")
            st.stop()

    st.success(f"✅ Successfully processed {len(attendance_files)} attendance files and 1 score file")

    with st.expander("View Raw Data Preview"):
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Attendance Data Preview:**")
            st.dataframe(attendance_long.head())
        with col2:
            st.write("**Score Data Preview:**")
            st.dataframe(wmt_scores.head())

    st.sidebar.header("Dashboard Controls")
    section = st.sidebar.radio("Select Section", ["Class Overview", "Student Comparison", "Individual Student Dashboard"])

    # Global filters
    st.sidebar.subheader("Global Filters")
    gender_options = ['All'] + sorted(df['Gender'].dropna().unique().astype(str).tolist())
    gender_filter = st.sidebar.selectbox("Select Gender", gender_options)

    # Date range safe handling
    try:
        min_date = df['Date'].min()
        max_date = df['Date'].max()
        if pd.isna(min_date) or pd.isna(max_date):
            min_date = date.today()
            max_date = date.today()
        else:
            min_date = min_date.date()
            max_date = max_date.date()
    except Exception:
        min_date = date.today()
        max_date = date.today()

    date_range = st.sidebar.date_input("Select Date Range", [min_date, max_date])

    # Filter data
    filtered_df = df.copy()
    if gender_filter != 'All':
        filtered_df = filtered_df[filtered_df['Gender'] == gender_filter]
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start, end = date_range
        # ensure Date column is datetime
        filtered_df = filtered_df[(filtered_df['Date'].dt.date >= start) & (filtered_df['Date'].dt.date <= end)]

    # ----------------- Class Overview -----------------
    if section == "Class Overview":
        st.header("Class Overview Dashboard")

        wmt_options = [w for w in sorted(filtered_df['WMT'].dropna().unique(), key=natural_wmt_key)]
        if not wmt_options:
            st.error("No WMT columns found in the score data.")
            st.stop()

        selected_wmt = st.selectbox("Select WMT", wmt_options)

        wmt_data = filtered_df[filtered_df['WMT'] == selected_wmt]
        if wmt_data.empty:
            avg_score = 0
            pass_rate = 0
            st.warning(f"No data available for {selected_wmt}")
        else:
            avg_score = wmt_data['Score'].mean()
            pass_count = wmt_data[wmt_data['Score'] >= 50]['Score'].count()
            pass_rate = (pass_count / len(wmt_data)) * 100 if len(wmt_data) > 0 else 0

        attendance_rate = (filtered_df[filtered_df['Status'] == 'Present'].shape[0] / filtered_df.shape[0]) * 100 if filtered_df.shape[0] > 0 else 0

        col1, col2, col3 = st.columns(3)
        col1.metric("Average Score", f"{avg_score:.2f}")
        col2.metric("Pass Rate", f"{pass_rate:.2f}%")
        col3.metric("Attendance Rate", f"{attendance_rate:.2f}%")

        st.subheader("Score Distribution")
        if not wmt_data.empty:
            fig_hist = px.histogram(wmt_data, x="Score", nbins=20, title=f"Score Distribution for {selected_wmt}")
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.info("No score data available for the selected WMT")

        st.subheader("Top and Bottom Performers")
        if not wmt_data.empty:
            wmt_data_unique = wmt_data.drop_duplicates(subset=['Name'])
            top_10 = wmt_data_unique.nlargest(min(10, len(wmt_data_unique)), 'Score')
            bottom_10 = wmt_data_unique.nsmallest(min(10, len(wmt_data_unique)), 'Score')

            fig_top = px.bar(top_10, x='Name', y='Score', title="Top Performers")
            fig_bottom = px.bar(bottom_10, x='Name', y='Score', title="Bottom Performers")

            col1, col2 = st.columns(2)
            col1.plotly_chart(fig_top, use_container_width=True)
            col2.plotly_chart(fig_bottom, use_container_width=True)
        else:
            st.info("No score data available for performance comparison")

        st.subheader("Attendance Rate Over Time")
        if not filtered_df.empty:
            daily_attendance = filtered_df.groupby(filtered_df['Date'].dt.date)['Status'].apply(lambda x: (x == 'Present').sum() / x.count() * 100 if x.count() > 0 else 0).reset_index(name='Attendance Rate')
            daily_attendance['Date'] = pd.to_datetime(daily_attendance['Date'])
            fig_att = px.line(daily_attendance, x='Date', y='Attendance Rate', title="Daily Attendance Rate")
            st.plotly_chart(fig_att, use_container_width=True)
        else:
            st.info("No attendance data available")

        st.subheader("Students Needing Attention")
        if not filtered_df.empty:
            # compute per-student averages and attendance
            student_stats = filtered_df.groupby(['Name', 'WMT']).agg({
                'Score': 'mean',
                'Status': lambda x: (x == 'Present').mean() * 100
            }).reset_index()

            # declining score detection (simple heuristic based on WMT numeric ordering)
            declining_students = []
            for name in student_stats['Name'].unique():
                sd = student_stats[student_stats['Name'] == name].copy()
                sd = sd.sort_values(by='WMT', key=lambda col: col.map(natural_wmt_key))
                if len(sd) > 2:
                    # check if scores strictly decrease across WMTs (simple)
                    if sd['Score'].is_monotonic_decreasing:
                        declining_students.append(name)

            low_attendance = student_stats[student_stats['Status'] < 75]['Name'].unique()

            if declining_students:
                st.write("**Students with declining scores:**", ", ".join(declining_students[:10]))
            else:
                st.write("No students with declining scores found.")

            if len(low_attendance) > 0:
                st.write("**Students with low attendance:**", ", ".join(low_attendance[:10]))
            else:
                st.write("No students with low attendance found.")
        else:
            st.info("No data available for student analysis")

    # ----------------- Student Comparison -----------------
    elif section == "Student Comparison":
        st.header("Student Comparison Dashboard")

        student_list = sorted(filtered_df['Name'].dropna().unique())
        if not student_list:
            st.error("No students found in the filtered data")
            st.stop()

        col1, col2 = st.columns(2)
        student1 = col1.selectbox("Select Student 1", student_list)
        student2 = col2.selectbox("Select Student 2", student_list, index=1 if len(student_list) > 1 else 0)

        student1_data = filtered_df[filtered_df['Name'] == student1]
        student2_data = filtered_df[filtered_df['Name'] == student2]

        st.subheader("Score Trend Comparison")
        if not student1_data.empty or not student2_data.empty:
            fig_trend = go.Figure()
            if not student1_data.empty:
                s1 = student1_data.groupby('WMT')['Score'].mean().reset_index().sort_values(by='WMT', key=lambda col: col.map(natural_wmt_key))
                fig_trend.add_trace(go.Scatter(x=s1['WMT'], y=s1['Score'], name=student1))
            if not student2_data.empty:
                s2 = student2_data.groupby('WMT')['Score'].mean().reset_index().sort_values(by='WMT', key=lambda col: col.map(natural_wmt_key))
                fig_trend.add_trace(go.Scatter(x=s2['WMT'], y=s2['Score'], name=student2))
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.info("No score data available for comparison")

        st.subheader("Attendance vs Performance")
        if not filtered_df.empty:
            scatter_data = filtered_df.groupby('Name').agg({
                'Score': 'mean',
                'Status': lambda x: (x == 'Present').mean() * 100
            }).reset_index()
            fig_scatter = px.scatter(scatter_data, x='Status', y='Score', hover_data=['Name'], title="Attendance % vs Avg Score")

            # Highlight selected students (add bigger markers)
            for stud, color in [(student1, 'red'), (student2, 'blue')]:
                if stud in scatter_data['Name'].values:
                    s = scatter_data[scatter_data['Name'] == stud]
                    fig_scatter.add_trace(go.Scatter(
                        x=s['Status'], y=s['Score'],
                        mode='markers',
                        marker=dict(size=14, line=dict(width=2)),
                        name=stud,
                        hoverinfo='name+x+y'
                    ))

            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.info("No data available for scatter plot")

        st.subheader("Performance Comparison")
        comparison_data = {
            'Metric': ['Average Score', 'Median Score', 'Attendance Rate'],
            student1: [
                student1_data['Score'].mean() if not student1_data.empty else 0,
                student1_data['Score'].median() if not student1_data.empty else 0,
                (student1_data['Status'] == 'Present').mean() * 100 if not student1_data.empty else 0
            ],
            student2: [
                student2_data['Score'].mean() if not student2_data.empty else 0,
                student2_data['Score'].median() if not student2_data.empty else 0,
                (student2_data['Status'] == 'Present').mean() * 100 if not student2_data.empty else 0
            ]
        }
        st.table(pd.DataFrame(comparison_data))

    # ----------------- Individual Student Dashboard -----------------
    else:
        st.header("Individual Student Dashboard")

        student_list = sorted(filtered_df['Name'].dropna().unique())
        if not student_list:
            st.error("No students found in the filtered data")
            st.stop()

        selected_student = st.selectbox("Select Student", student_list)
        student_data = filtered_df[filtered_df['Name'] == selected_student]

        if student_data.empty:
            st.warning(f"No data available for {selected_student}")
            st.stop()

        avg_score = student_data['Score'].mean()
        attendance_rate = (student_data['Status'] == 'Present').mean() * 100
        total_score = student_data['Score'].sum()

        col1, col2, col3 = st.columns(3)
        col1.metric("Average Score", f"{avg_score:.2f}")
        col2.metric("Attendance Rate", f"{attendance_rate:.2f}%")
        col3.metric("Total Score", f"{total_score:.2f}")

        st.subheader("Score Trend Over Time")
        score_trend = student_data.groupby('WMT')['Score'].mean().reset_index().sort_values(by='WMT', key=lambda col: col.map(natural_wmt_key))
        if not score_trend.empty:
            fig_score = px.line(score_trend, x='WMT', y='Score', title="WMT Score Trend")
            st.plotly_chart(fig_score, use_container_width=True)
        else:
            st.info("No score trend data available")

        st.subheader("Attendance Heatmap")
        student_att = student_data.copy()
        # ensure Date exists
        student_att = student_att.dropna(subset=['Date'])
        if not student_att.empty:
            student_att['Day'] = student_att['Date'].dt.day
            student_att['Month'] = student_att['Date'].dt.month
            student_att['Year'] = student_att['Date'].dt.year
            student_att['Status_num'] = student_att['Status'].map({'Present': 1, 'Absent': 0})

            heatmap_data = student_att.pivot_table(values='Status_num', index='Day', columns='Month', aggfunc='mean')
            if heatmap_data.empty:
                st.info("Not enough attendance data for heatmap")
            else:
                fig_heatmap = px.imshow(heatmap_data, title="Monthly Attendance Pattern")
                st.plotly_chart(fig_heatmap, use_container_width=True)
        else:
            st.info("No attendance data available for heatmap")

        st.subheader("Subject-wise Performance")
        subject_data = student_data.copy()
        subject_data['Subject'] = subject_data['WMT'].astype(str).str.extract(r'(W\d+)', expand=False)
        subject_avg = subject_data.groupby('Subject')['Score'].mean().reset_index()
        if not subject_avg.empty:
            fig_subject = px.bar(subject_avg, x='Subject', y='Score', title="Average Score by Subject")
            st.plotly_chart(fig_subject, use_container_width=True)
        else:
            st.info("No subject-wise data available")

if __name__ == "__main__":
    main()
