# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import re
import os
from datetime import date

st.set_page_config(page_title="Student Performance Dashboard", page_icon="📊", layout="wide")

# -------------------------
# Helpers
# -------------------------
def read_data_file(uploaded):
    """Read uploaded file (Streamlit UploadedFile or path string). Return DataFrame or None."""
    try:
        # uploaded may be a path string or UploadedFile
        if isinstance(uploaded, str):
            path = uploaded
            ext = path.split('.')[-1].lower()
            if ext == 'csv':
                return pd.read_csv(path)
            elif ext in ('xls', 'xlsx'):
                return pd.read_excel(path)
            else:
                return None
        else:
            # Streamlit UploadedFile
            name = getattr(uploaded, "name", "")
            ext = name.split('.')[-1].lower() if name else ''
            uploaded.seek(0)
            if ext == 'csv':
                return pd.read_csv(uploaded)
            elif ext in ('xls', 'xlsx'):
                return pd.read_excel(uploaded)
            else:
                # try both
                try:
                    uploaded.seek(0)
                    return pd.read_csv(uploaded)
                except Exception:
                    uploaded.seek(0)
                    return pd.read_excel(uploaded)
    except Exception as e:
        st.warning(f"Could not read {getattr(uploaded,'name',str(uploaded))}: {e}")
        return None

def normalize_colnames(df):
    """Normalize column names to lowercase stripped strings for detection but preserve original mapping."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df

def standardize_columns(df, file_label="file"):
    """Try to detect ID, Name, Roll columns and rename them to standard names."""
    df = df.copy()
    df = normalize_colnames(df)
    cols_lower = [c.lower() for c in df.columns]

    rename_map = {}
    for i, c in enumerate(cols_lower):
        if ('id' == c) or ('student id' in c) or (c.endswith('id')) or ('reg' in c and 'no' in c):
            rename_map[df.columns[i]] = 'ID'
        elif 'name' in c and 'student' in c or c == 'name' or 'full name' in c:
            rename_map[df.columns[i]] = 'Name'
        elif 'roll' in c or 'roll no' in c or 'roll_no' in c:
            rename_map[df.columns[i]] = 'Roll'
    df.rename(columns=rename_map, inplace=True)

    # If ID or Name still missing, try heuristics (first column->ID, second or third ->Name)
    if 'ID' not in df.columns or 'Name' not in df.columns:
        # show header to user for debugging
        st.info(f"Detected columns for {file_label}: {list(df.columns)}")
        # try naive fallback
        if len(df.columns) >= 2:
            if 'ID' not in df.columns:
                df.rename(columns={df.columns[0]: 'ID'}, inplace=True)
            if 'Name' not in df.columns:
                # prefer 2nd or 3rd
                candidate = df.columns[1] if len(df.columns) >= 2 else df.columns[0]
                if candidate != 'ID':
                    df.rename(columns={candidate: 'Name'}, inplace=True)
        # if still not present, leave as-is and caller will handle
    return df

def detect_date_columns(df):
    """Return a list of columns that appear to be attendance date columns.
       Strategy:
         - exclude known non-date columns (ID, Name, Gender, Roll)
         - treat a column as date if its header parses to a date OR
           a sample of its values look like attendance marks (✔,✘,P,A,Present,Absent,1/0)
    """
    non_date = {'ID', 'Name', 'Gender', 'Roll'}
    candidates = [c for c in df.columns if c not in non_date]
    date_cols = []
    for c in candidates:
        # if column name looks like a date
        try:
            parsed = pd.to_datetime(c, errors='coerce')
            if not pd.isna(parsed):
                date_cols.append(c)
                continue
        except Exception:
            pass

        # sample values
        sample = df[c].dropna().astype(str).head(10).tolist()
        if not sample:
            # empty column - treat as non-date
            continue
        # check for checkmarks, P/A, Present/Absent, 1/0
        pattern = re.compile(r'✔|✘|\bP\b|\bA\b|Present|Absent|present|absent|^[01]$')
        if any(pattern.search(s) for s in sample):
            date_cols.append(c)
            continue

        # if many non-numeric values and short header, might be a date header like 'Jan 1'
        if len(c) <= 20:
            # try flexible parse
            t = pd.to_datetime(c, errors='coerce', dayfirst=False)
            if not pd.isna(t):
                date_cols.append(c)
                continue
    return date_cols

def normalize_status(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if s == '':
        return np.nan
    # direct symbols
    if '✔' in s:
        return 'Present'
    if '✘' in s:
        return 'Absent'
    # words
    if re.search(r'present', s, re.IGNORECASE):
        return 'Present'
    if re.search(r'absent', s, re.IGNORECASE):
        return 'Absent'
    # single-letter
    if re.fullmatch(r'P', s, flags=re.IGNORECASE):
        return 'Present'
    if re.fullmatch(r'A', s, flags=re.IGNORECASE):
        return 'Absent'
    # numeric 1/0
    if re.fullmatch(r'1', s):
        return 'Present'
    if re.fullmatch(r'0', s):
        return 'Absent'
    # fallback: non-empty treat as Present
    return 'Present'

def extract_numeric_score(val):
    if pd.isna(val):
        return np.nan
    s = str(val)
    m = re.search(r'(-?\d+\.?\d*)', s)
    return float(m.group(1)) if m else np.nan

# -------------------------
# Processing pipeline
# -------------------------
@st.cache_data
def process_files(attendance_uploads, score_upload):
    # attendance_uploads: list of UploadedFile or file paths
    attendance_dfs = []
    for uploaded in attendance_uploads:
        df = read_data_file(uploaded)
        if df is None:
            continue
        df = standardize_columns(df, getattr(uploaded, "name", str(uploaded)))
        # try gender detection from filename (if available)
        fname = getattr(uploaded, "name", "") if not isinstance(uploaded, str) else os.path.basename(uploaded)
        fname_l = fname.lower()
        if 'boy' in fname_l or 'male' in fname_l:
            df['Gender'] = df.get('Gender', 'Boy')
        elif 'girl' in fname_l or 'female' in fname_l:
            df['Gender'] = df.get('Gender', 'Girl')
        else:
            df['Gender'] = df.get('Gender', 'Unknown')
        attendance_dfs.append(df)

    if not attendance_dfs:
        return None, None, None

    attendance = pd.concat(attendance_dfs, ignore_index=True)

    # detect date columns
    date_cols = detect_date_columns(attendance)
    if not date_cols:
        # As last resort, treat every column except ID/Name/Roll/Gender as date columns
        date_cols = [c for c in attendance.columns if c not in {'ID', 'Name', 'Roll', 'Gender'}]
        if not date_cols:
            return None, None, None

    # melt into long format
    attendance_long = attendance.melt(id_vars=['ID', 'Roll', 'Name', 'Gender'], value_vars=date_cols,
                                     var_name='Date', value_name='RawStatus')

    # normalize Status
    attendance_long['Status'] = attendance_long['RawStatus'].apply(normalize_status)

    # try parse Date column values
    # Some Date values may be column headers like "Jan 01 (Wed)" so parse flexibly
    def parse_date_string(s):
        try:
            return pd.to_datetime(s, errors='coerce', dayfirst=False)
        except Exception:
            return pd.NaT

    attendance_long['Date_parsed'] = attendance_long['Date'].apply(parse_date_string)
    # If parsing headers failed, attempt to infer date using patterns (remove brackets etc.)
    mask_na = attendance_long['Date_parsed'].isna()
    if mask_na.any():
        attendance_long.loc[mask_na, 'Date_parsed'] = attendance_long.loc[mask_na, 'Date'].astype(str).str.replace(r'[\[\]\(\)]', ' ', regex=True).apply(parse_date_string)

    attendance_long = attendance_long.dropna(subset=['Date_parsed', 'Status']).copy()
    attendance_long = attendance_long.rename(columns={'Date_parsed': 'Date'}).drop(columns=['RawStatus', 'Date'])

    # Ensure string types for join keys
    attendance_long['ID'] = attendance_long['ID'].astype(str).str.strip()
    attendance_long['Name'] = attendance_long['Name'].astype(str).str.strip()

    # Process score file
    score_df = read_data_file(score_upload)
    if score_df is None:
        return None, None, None
    score_df = standardize_columns(score_df, getattr(score_upload, "name", str(score_upload)))
    # drop "total"/"merit" columns heuristically
    drop_cols = [c for c in score_df.columns if re.search(r'\btotal\b|\bmerit\b', c, re.IGNORECASE)]
    score_df = score_df.drop(columns=drop_cols, errors='ignore')

    score_cols = [c for c in score_df.columns if c not in {'ID', 'Name', 'Roll'}]
    if not score_cols:
        return None, None, None

    # convert score columns to numeric
    for c in score_cols:
        score_df[c] = score_df[c].apply(extract_numeric_score)
        score_df[c] = pd.to_numeric(score_df[c], errors='coerce').fillna(0)

    # melt scores
    wmt_long = score_df.melt(id_vars=['ID', 'Name'], value_vars=score_cols, var_name='WMT', value_name='Score')

    # normalize keys
    wmt_long['ID'] = wmt_long['ID'].astype(str).str.strip()
    wmt_long['Name'] = wmt_long['Name'].astype(str).str.strip()

    # merge
    merged = pd.merge(wmt_long, attendance_long, on=['ID', 'Name'], how='outer')

    # ensure Date column present and typed
    if 'Date' in merged.columns:
        merged['Date'] = pd.to_datetime(merged['Date'], errors='coerce')
    return merged, score_df, attendance_long

# -------------------------
# Auto-detect sample files under /mnt/data (optional)
# -------------------------
def find_sample_files():
    sample_paths = []
    base = "/mnt/data"
    if os.path.isdir(base):
        for name in os.listdir(base):
            if name.lower().endswith(('.csv', '.xls', '.xlsx')):
                sample_paths.append(os.path.join(base, name))
    return sample_paths

# -------------------------
# Main UI
# -------------------------
def main():
    st.title("📊 Student Performance Dashboard (Stable Build)")

    st.sidebar.header("1) Upload Files")
    attendance_uploads = st.sidebar.file_uploader("Upload one or more attendance files (CSV/XLSX)", type=['csv', 'xls', 'xlsx'], accept_multiple_files=True)
    score_upload = st.sidebar.file_uploader("Upload WMT scores file (CSV/XLSX)", type=['csv', 'xls', 'xlsx'])

    # If user didn't upload anything, offer to auto-load sample files from /mnt/data
    if not attendance_uploads:
        samples = find_sample_files()
        if samples:
            st.sidebar.write("No uploads detected — sample files found on the server. You can use them for testing.")
            use_samples = st.sidebar.checkbox("Use sample files from /mnt/data", value=True)
            if use_samples:
                attendance_uploads = [p for p in samples if 'result' not in p.lower()]  # heuristic
                # pick a result file for score if exists
                score_candidates = [p for p in samples if 'result' in p.lower()]
                score_upload = score_candidates[0] if score_candidates else (samples[0] if samples else None)
        else:
            st.sidebar.write("No sample files found on server. Please upload files to proceed.")

    # show what will be processed
    st.sidebar.markdown("**Files to be processed:**")
    if attendance_uploads:
        if isinstance(attendance_uploads, list):
            for a in attendance_uploads:
                st.sidebar.write(f"- {getattr(a,'name',a)}")
        else:
            st.sidebar.write(f"- {getattr(attendance_uploads,'name',attendance_uploads)}")
    else:
        st.sidebar.write("- (none)")

    st.sidebar.write(f"Score file: {getattr(score_upload,'name',score_upload) if score_upload else '(none)'}")

    # require files
    if not attendance_uploads or not score_upload:
        st.info("Please upload attendance files and a score file (or enable sample files).")
        return

    with st.spinner("Processing files..."):
        merged_df, score_df, attendance_long = process_files(attendance_uploads, score_upload)

    if merged_df is None:
        st.error("Failed to process files. Check file headers and content. Use the debug preview below to inspect files.")
        # show previews of raw files to help debug
        st.markdown("### Raw File Previews (debug)")
        for uploaded in attendance_uploads:
            df = read_data_file(uploaded)
            if df is not None:
                st.write(f"**Preview — {getattr(uploaded,'name',uploaded)}**")
                st.dataframe(df.head())
        if score_upload:
            df_s = read_data_file(score_upload)
            if df_s is not None:
                st.write(f"**Preview — {getattr(score_upload,'name',score_upload)}**")
                st.dataframe(df_s.head())
        return

    st.success("✅ Files processed successfully")

    # show small merged preview and attendance preview
    st.header("Data Preview")
    st.markdown("Merged (scores + attendance) — first rows")
    st.dataframe(merged_df.head())

    st.markdown("Attendance long-format preview")
    st.dataframe(attendance_long.head())

    # -------------------------
    # Dashboard controls
    # -------------------------
    st.sidebar.header("Dashboard Controls")
    section = st.sidebar.radio("Select section", ["Class Overview", "Student Comparison", "Individual Student Dashboard"])

    # global filters
    # ensure Gender column exists
    if 'Gender' not in merged_df.columns:
        merged_df['Gender'] = 'Unknown'
    gender_options = ['All'] + sorted(merged_df['Gender'].dropna().unique().astype(str).tolist())
    gender_filter = st.sidebar.selectbox("Filter by Gender", gender_options)

    # ensure Date exists
    if 'Date' not in merged_df.columns or merged_df['Date'].isna().all():
        st.warning("Attendance dates missing or could not be parsed. Date-based filters and charts will be disabled.")
        min_date = date.today()
        max_date = date.today()
    else:
        try:
            min_date = merged_df['Date'].min().date()
            max_date = merged_df['Date'].max().date()
        except Exception:
            min_date = date.today()
            max_date = date.today()

    date_range = st.sidebar.date_input("Date range", [min_date, max_date])

    # apply filters
    df = merged_df.copy()
    if gender_filter != 'All':
        df = df[df['Gender'] == gender_filter]
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2 and 'Date' in df.columns:
        df = df[(df['Date'].dt.date >= date_range[0]) & (df['Date'].dt.date <= date_range[1])]

    # -------------------------
    # Class Overview
    # -------------------------
    if section == "Class Overview":
        st.header("Class Overview")

        wmt_vals = sorted(df['WMT'].dropna().unique().tolist(), key=lambda s: str(s))
        if not wmt_vals:
            st.info("No WMT score columns found.")
            return
        selected_wmt = st.selectbox("Select WMT", wmt_vals)

        wmt_df = df[df['WMT'] == selected_wmt]
        avg_score = wmt_df['Score'].mean() if not wmt_df.empty else 0
        pass_rate = (wmt_df['Score'] >= 50).mean() * 100 if not wmt_df.empty else 0
        attendance_rate = (df['Status'] == 'Present').mean() * 100 if len(df) > 0 else 0

        c1, c2, c3 = st.columns(3)
        c1.metric("Average Score", f"{avg_score:.2f}")
        c2.metric("Pass Rate", f"{pass_rate:.2f}%")
        c3.metric("Attendance Rate", f"{attendance_rate:.2f}%")

        st.subheader("Score distribution")
        if not wmt_df.empty:
            fig = px.histogram(wmt_df, x='Score', nbins=20, title=f"Distribution — {selected_wmt}")
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Top / Bottom performers")
        if not wmt_df.empty:
            uniq = wmt_df.drop_duplicates(subset=['Name'])
            top = uniq.nlargest(10, 'Score')
            bottom = uniq.nsmallest(10, 'Score')
            colA, colB = st.columns(2)
            colA.write("Top performers")
            colA.dataframe(top[['Name','Score']])
            colB.write("Bottom performers")
            colB.dataframe(bottom[['Name','Score']])

        st.subheader("Attendance over time")
        if 'Date' in df.columns and not df['Date'].isna().all():
            att = df.groupby(df['Date'].dt.date)['Status'].apply(lambda s: (s == 'Present').sum() / s.count() * 100 if s.count() > 0 else 0).reset_index(name='Attendance%')
            att['Date'] = pd.to_datetime(att['Date'])
            fig2 = px.line(att, x='Date', y='Attendance%', title='Daily Attendance %')
            st.plotly_chart(fig2, use_container_width=True)

    # -------------------------
    # Student Comparison
    # -------------------------
    elif section == "Student Comparison":
        st.header("Student Comparison")
        students = sorted(df['Name'].dropna().unique().tolist())
        if not students:
            st.info("No students found.")
            return
        s1, s2 = st.columns(2)
        student1 = s1.selectbox("Student 1", students, index=0)
        student2 = s2.selectbox("Student 2", students, index=min(1, max(0, len(students)-1)))

        d1 = df[df['Name'] == student1]
        d2 = df[df['Name'] == student2]

        st.subheader("Score trends")
        fig = go.Figure()
        if not d1.empty:
            t1 = d1.groupby('WMT')['Score'].mean().reset_index()
            fig.add_trace(go.Scatter(x=t1['WMT'], y=t1['Score'], name=student1))
        if not d2.empty:
            t2 = d2.groupby('WMT')['Score'].mean().reset_index()
            fig.add_trace(go.Scatter(x=t2['WMT'], y=t2['Score'], name=student2))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Attendance vs Score (all students)")
        scatter = df.groupby('Name').agg({'Score':'mean', 'Status': lambda s: (s=='Present').mean()*100}).reset_index()
        scatter = scatter.rename(columns={'Status':'Attendance%','Score':'AvgScore'})
        fig_sc = px.scatter(scatter, x='Attendance%', y='AvgScore', hover_data=['Name'])
        st.plotly_chart(fig_sc, use_container_width=True)

        st.subheader("Summary table")
        def student_summary(d):
            return {
                'Average Score': d['Score'].mean() if not d.empty else 0,
                'Median Score': d['Score'].median() if not d.empty else 0,
                'Attendance%': (d['Status'] == 'Present').mean()*100 if not d.empty else 0
            }
        summ = pd.DataFrame({
            'Metric': ['Average Score','Median Score','Attendance%'],
            student1: list(student_summary(d1).values()),
            student2: list(student_summary(d2).values())
        })
        st.table(summ)

    # -------------------------
    # Individual Student
    # -------------------------
    else:
        st.header("Individual Student Dashboard")
        students = sorted(df['Name'].dropna().unique().tolist())
        if not students:
            st.info("No students found.")
            return
        student = st.selectbox("Select student", students)
        sd = df[df['Name'] == student]
        if sd.empty:
            st.info("No data for this student.")
            return

        avg = sd['Score'].mean()
        att_rate = (sd['Status'] == 'Present').mean()*100
        total = sd['Score'].sum()

        a,b,c = st.columns(3)
        a.metric("Average Score", f"{avg:.2f}")
        b.metric("Attendance %", f"{att_rate:.2f}%")
        c.metric("Total Score", f"{total:.2f}")

        st.subheader("Score by WMT")
        sb = sd.groupby('WMT')['Score'].mean().reset_index()
        st.bar_chart(sb.set_index('WMT'))

        st.subheader("Attendance heatmap (Day x Month)")
        att = sd.dropna(subset=['Date']).copy()
        if not att.empty:
            att['Day'] = att['Date'].dt.day
            att['Month'] = att['Date'].dt.month
            att['PresentNum'] = att['Status'].map({'Present':1,'Absent':0})
            heat = att.pivot_table(values='PresentNum', index='Day', columns='Month', aggfunc='mean')
            fig_h = px.imshow(heat, labels={'x':'Month','y':'Day','color':'Present %'}, aspect='auto')
            st.plotly_chart(fig_h, use_container_width=True)
        else:
            st.info("No attendance date entries to show heatmap.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Unexpected error: {e}")
