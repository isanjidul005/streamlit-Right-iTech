import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="Student Dashboard — Interactive", layout="wide")

# ----------------------------
# Helpers & Readers
# ----------------------------
@st.cache_data
def read_data_file(file, header=None):
    """Read CSV or Excel files robustly, handling encoding issues."""
    file_type = getattr(file, "type", "")
    file.seek(0)
    try:
        if "excel" in file_type or file.name.lower().endswith((".xls", ".xlsx")):
            return pd.read_excel(file, header=header)
        else:
            try:
                return pd.read_csv(file, header=header, encoding="utf-8")
            except UnicodeDecodeError:
                file.seek(0)
                return pd.read_csv(file, header=header, encoding="latin1")
    except Exception as e:
        st.error(f"Failed to read {file.name}: {e}")
        return None


def auto_rename_attendance_columns(df):
    """Try to coerce attendance-style files into a consistent schema.
    We expect first 3 columns to be ID, Roll, Name and the rest dates/session columns.
    """
    if df is None:
        return None
    # Strip whitespace and convert to strings
    df.columns = df.columns.astype(str).str.strip()
    # Ensure at least 3 columns
    if df.shape[1] < 3:
        return None
    new_cols = list(df.columns)
    new_cols[0] = "ID"
    new_cols[1] = "Roll"
    new_cols[2] = "Name"
    df.columns = new_cols
    # Convert name column to string
    df["Name"] = df["Name"].astype(str).str.strip()
    return df


def read_attendance_file(file, gender):
    df = read_data_file(file, header=1)  # try a messy header line
    if df is None:
        return None
    df = auto_rename_attendance_columns(df)
    if df is None:
        return None
    df["Gender"] = gender

    # If date-like columns are not parsed, try to coerce them (keep original col names as strings)
    # We'll leave the column labels as-is; they will be parsed later if possible
    return df


def read_score_file(file):
    df = read_data_file(file, header=0)
    if df is None:
        return None
    # Standardize columns
    df.columns = df.columns.astype(str).str.strip()
    # Try to find ID/Roll/Name columns (case-insensitive)
    cols_lower = {c.lower(): c for c in df.columns}
    id_col = cols_lower.get("id", cols_lower.get("student id", None))
    roll_col = cols_lower.get("roll", cols_lower.get("roll no", None))
    name_col = cols_lower.get("name", cols_lower.get("student name", None))

    if not id_col or not roll_col or not name_col:
        st.error("Score file must contain columns named ID, Roll and Name (or similar).")
        return None
    # Rename primary columns for internal consistency
    df = df.rename(columns={id_col: "ID", roll_col: "Roll", name_col: "Name"})
    return df


# ----------------------------
# Cleaning & Transformations
# ----------------------------

def melt_attendance(att_df):
    # Melt all columns after the first 4 (ID, Roll, Name, Gender)
    id_vars = [c for c in att_df.columns[:4]] if "Gender" in att_df.columns else [c for c in att_df.columns[:3]]
    value_vars = [c for c in att_df.columns if c not in id_vars]

    long = att_df.melt(id_vars=id_vars, value_vars=value_vars, var_name="Date", value_name="Status")
    # Try parse Date to datetime if possible
    try:
        long["Date_parsed"] = pd.to_datetime(long["Date"], dayfirst=True, errors="coerce")
    except Exception:
        long["Date_parsed"] = pd.to_datetime(long["Date"], errors="coerce")
    long["RawDateLabel"] = long["Date"].astype(str)

    # Normalize status to P/A
    def status_to_present(x):
        s = str(x).strip().upper()
        if s == "" or s == "NAN":
            return np.nan
        if s.startswith("✔") or s.startswith("P") or s.startswith("1") or s in ["PRESENT", "YES"]:
            return 1
        if s.startswith("A") or s in ["ABSENT", "0", "NO"]:
            return 0
        # try numeric
        try:
            v = float(s)
            return 1 if v > 0 else 0
        except Exception:
            return np.nan

    long["Present"] = long["Status"].apply(status_to_present)
    return long


def extract_wmt_columns(score_df):
    # WMT columns may be labelled 'WMT1', 'wmt 1', 'WMT-1' etc. We'll find columns containing 'WMT' or similar.
    wmt_cols = [c for c in score_df.columns if "wmt" in c.lower()]
    # If none found, heuristically pick columns that look like assessment columns (numbers, 'Test', 'Exam')
    if not wmt_cols:
        wmt_cols = [c for c in score_df.columns if any(k in c.lower() for k in ["test", "assess", "exam"]) ]
    return wmt_cols

# ----------------------------
# UI
# ----------------------------

def main():
    st.title("🚀 Awesome Interactive Student Dashboard")

    st.sidebar.header("Uploads & Settings")
    boys_file = st.sidebar.file_uploader("Boys attendance (xlsx/csv)", type=["xlsx", "csv"] , key="boys")
    girls_file = st.sidebar.file_uploader("Girls attendance (xlsx/csv)", type=["xlsx", "csv"], key="girls")
    score_file = st.sidebar.file_uploader("Score file (xlsx/csv)", type=["xlsx", "csv"], key="scores")

    advanced = st.sidebar.expander("Advanced options")
    with advanced:
        normalize_names = st.checkbox("Try to normalize student names (strip, title)", value=True)
        attendance_date_filter = st.checkbox("Enable date-range filter for attendance visualizations", value=True)

    if not (boys_file and girls_file and score_file):
        st.info("Upload Boys, Girls attendance files and Score file in the sidebar to begin. Example: Excel files exported from your MIS.")
        st.stop()

    # Read files
    boys_df = read_attendance_file(boys_file, "Boy")
    girls_df = read_attendance_file(girls_file, "Girl")
    score_df = read_score_file(score_file)

    if any(x is None for x in [boys_df, girls_df, score_df]):
        st.stop()

    # Optional name normalization
    if normalize_names:
        score_df["Name"] = score_df["Name"].astype(str).str.strip().str.title()
        boys_df["Name"] = boys_df["Name"].astype(str).str.strip().str.title()
        girls_df["Name"] = girls_df["Name"].astype(str).str.strip().str.title()

    # Combine attendance
    attendance_df = pd.concat([boys_df, girls_df], ignore_index=True, sort=False)
    attendance_long = melt_attendance(attendance_df)

    # Attendance summary per student (mean present across available days)
    attendance_summary = (
        attendance_long.groupby(["ID", "Name", "Roll", "Gender"])['Present']
        .agg(['mean', 'count'])
        .reset_index()
        .rename(columns={'mean': 'AttendanceRate', 'count': 'RecordedSessions'})
    )

    # Process scores
    wmt_cols = extract_wmt_columns(score_df)
    if not wmt_cols:
        st.warning("No WMT/test columns detected automatically. You can select which columns are score columns below.")
        chosen = st.multiselect("Pick score columns (if any)", options=list(score_df.columns), default=[c for c in score_df.columns if c not in ['ID','Roll','Name']][:3])
        wmt_cols = chosen

    score_long = score_df.melt(id_vars=[c for c in ['ID','Roll','Name'] if c in score_df.columns], value_vars=wmt_cols, var_name='WMT', value_name='Score')
    # Extract numeric from Score and coerce
    score_long['Score'] = pd.to_numeric(score_long['Score'].astype(str).str.extract(r'(\d+\.?\d*)')[0], errors='coerce').fillna(0)

    # Merge
    combined = pd.merge(score_long, attendance_summary, how='left', on=["ID","Name","Roll"]) 
    # Fill missing AttendanceRate as 0 if no records
    combined['AttendanceRate'] = combined['AttendanceRate'].fillna(0)

    # ---------------- Layout: KPIs ----------------
    st.header("Key performance indicators")
    total_students = combined['ID'].nunique()
    avg_score = combined['Score'].mean()
    avg_att = combined['AttendanceRate'].mean()

    k1, k2, k3, k4 = st.columns([1.2,1.2,1.2,1.2])
    k1.metric("Students", total_students)
    k2.metric("Avg Score", f"{avg_score:.2f}")
    k3.metric("Avg Attendance", f"{avg_att*100:.1f}%")
    # At-risk metric: students with attendance < 60% or avg score < 40
    at_risk = combined.groupby('ID').agg(AvgScore=('Score','mean'), AvgAttendance=('AttendanceRate','mean')).reset_index()
    at_risk_count = at_risk[(at_risk['AvgAttendance']<0.6) | (at_risk['AvgScore']<40)]['ID'].nunique()
    k4.metric("At-risk students", at_risk_count)

    # ---------------- Filters ----------------
    st.sidebar.header("Interactive filters")
    genders = combined['Gender'].dropna().unique().tolist()
    selected_genders = st.sidebar.multiselect("Gender", options=genders, default=genders)

    wmt_options = combined['WMT'].unique().tolist()
    selected_wmts = st.sidebar.multiselect("Assessments (WMT)", options=wmt_options, default=wmt_options)

    # Date filter for attendance
    if attendance_date_filter:
        min_date = attendance_long['Date_parsed'].min()
        max_date = attendance_long['Date_parsed'].max()
        if pd.isna(min_date):
            # fallback: use RawDateLabel unique values
            min_date = None; max_date = None
            st.sidebar.info("Attendance dates could not be parsed automatically — charts will use session labels instead of calendar dates.")
        else:
            dr = st.sidebar.date_input("Attendance date range", value=(min_date, max_date))
            start_d, end_d = dr if isinstance(dr, tuple) else (dr, dr)
            # filter attendance_long for that range
            attendance_filtered = attendance_long[(attendance_long['Date_parsed']>=pd.to_datetime(start_d)) & (attendance_long['Date_parsed']<=pd.to_datetime(end_d))]
    else:
        attendance_filtered = attendance_long.copy()

    # Filter combined data according to sidebar selections
    filtered = combined[combined['Gender'].isin(selected_genders) & combined['WMT'].isin(selected_wmts)].copy()

    # ---------------- Visuals ----------------
    st.header("Visual Insights")

    # 1) Score Distribution
    left, right = st.columns([2,1])
    with left:
        st.subheader("Score distribution (all selected assessments)")
        fig_dist = px.histogram(filtered, x='Score', nbins=25, marginal='box', title='Scores distribution')
        st.plotly_chart(fig_dist, use_container_width=True)

    with right:
        st.subheader("Score by Gender")
        fig_violin = px.violin(filtered, x='Gender', y='Score', box=True, points='all', title='Score distribution by gender')
        st.plotly_chart(fig_violin, use_container_width=True)

    # 2) Attendance vs Score scatter with trendline
    st.subheader("Attendance vs Score (per record)")
    fig_scatter = px.scatter(filtered, x='AttendanceRate', y='Score', color='Gender', hover_data=['Name','Roll','WMT'], trendline='ols')
    fig_scatter.update_layout(yaxis_title='Score', xaxis_title='Attendance rate (0-1)')
    st.plotly_chart(fig_scatter, use_container_width=True)

    # 3) Correlation heatmap (pivot student-level)
    st.subheader("Correlations between assessments & attendance")
    # build wide table: rows=ID, cols=WMT average score
    pivot_scores = filtered.pivot_table(index=['ID','Name','Roll','Gender'], columns='WMT', values='Score', aggfunc='mean').reset_index()
    corr_df = pivot_scores.select_dtypes(include=[np.number]).corr()
    if corr_df.shape[0] > 0:
        fig_corr = px.imshow(corr_df, text_auto=True, aspect='auto', title='Correlation matrix (numeric columns)')
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info('Not enough numeric columns for a correlation matrix.')

    # 4) Attendance trend across sessions
    st.subheader("Attendance trend across sessions/dates")
    # If we have parsed dates use them, else use RawDateLabel
    if attendance_filtered['Date_parsed'].notna().any():
        trend = attendance_filtered.groupby('Date_parsed').Present.mean().reset_index()
        trend = trend.sort_values('Date_parsed')
        fig_trend = px.line(trend, x='Date_parsed', y='Present', markers=True, title='Attendance rate over time')
        fig_trend.update_layout(yaxis_title='Attendance rate (0-1)', xaxis_title='Date')
        st.plotly_chart(fig_trend, use_container_width=True)
    else:
        trend = attendance_filtered.groupby('RawDateLabel').Present.mean().reset_index()
        fig_trend = px.bar(trend, x='RawDateLabel', y='Present', title='Attendance rate across sessions (labels)')
        st.plotly_chart(fig_trend, use_container_width=True)

    # 5) Leaderboard and at-risk list
    st.subheader("Leaderboard & At-risk students")
    leaderboard = filtered.groupby(['ID','Name','Roll','Gender']).agg(AvgScore=('Score','mean'), AttendanceRate=('AttendanceRate','mean')).reset_index()
    leaderboard = leaderboard.sort_values(['AvgScore','AttendanceRate'], ascending=[False, False])

    lb_col1, lb_col2 = st.columns([2,1])
    with lb_col1:
        st.write("Top performers (by Avg score)")
        st.dataframe(leaderboard.head(20).style.format({"AvgScore": "{:.1f}", "AttendanceRate": "{:.1%}"}))
    with lb_col2:
        st.write("At-risk (low attendance or low score)")
        risk_df = leaderboard[(leaderboard['AttendanceRate']<0.6) | (leaderboard['AvgScore']<40)].copy()
        st.dataframe(risk_df.style.format({"AvgScore": "{:.1f}", "AttendanceRate": "{:.1%}"}))

    # 6) Per-student detail explorer
    st.subheader("Student explorer")
    student_selector = st.selectbox("Pick student (ID - Name)", options=leaderboard.apply(lambda r: f"{r.ID} - {r.Name}", axis=1).tolist())
    if student_selector:
        sid = int(student_selector.split(" - ")[0])
        student_records = filtered[filtered['ID']==sid].copy()
        st.markdown(f"**{student_records['Name'].iloc[0]}** — Roll: {student_records['Roll'].iloc[0]} — Gender: {student_records['Gender'].iloc[0]}")
        # show per-assessment scores
        st.dataframe(student_records[['WMT','Score','AttendanceRate']].sort_values('WMT'))
        # small radar or bar chart
        fig_student = px.bar(student_records, x='WMT', y='Score', title='Scores by assessment', text='Score')
        st.plotly_chart(fig_student, use_container_width=True)

    # ---------------- Export ----------------
    st.header("Exports & Quick actions")
    # let user download filtered combined data
    export_df = filtered.copy()
    export_df['AttendanceRate'] = export_df['AttendanceRate'].round(3)
    csv = export_df.to_csv(index=False).encode('utf-8')
    st.download_button(label="Download filtered data as CSV", data=csv, file_name='student_dashboard_export.csv', mime='text/csv')

    st.success("Interactive dashboard generated. Use the filters on the left to explore, and tell me if you want custom analyses (e.g., class-wise comparisons, teacher-level rollup, or predictive flags).")


if __name__ == "__main__":
    main()
