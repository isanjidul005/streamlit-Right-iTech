import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Student Dashboard — Interactive", layout="wide")

# ----------------------------
# Helpers & Readers
# ----------------------------
@st.cache_data
def read_data_file(file, header=None):
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
    if df is None:
        return None
    df.columns = df.columns.astype(str).str.strip()
    if df.shape[1] < 3:
        return None
    new_cols = list(df.columns)
    new_cols[0] = "ID"
    new_cols[1] = "Roll"
    new_cols[2] = "Name"
    df.columns = new_cols
    df["Name"] = df["Name"].astype(str).str.strip()
    return df


def read_attendance_file(file, gender):
    df = read_data_file(file, header=1)
    if df is None:
        return None
    df = auto_rename_attendance_columns(df)
    if df is None:
        return None
    df["Gender"] = gender
    return df


def read_score_file(file):
    df = read_data_file(file, header=0)
    if df is None:
        return None
    df.columns = df.columns.astype(str).str.strip()
    cols_lower = {c.lower(): c for c in df.columns}
    id_col = cols_lower.get("id", cols_lower.get("student id", None))
    roll_col = cols_lower.get("roll", cols_lower.get("roll no", None))
    name_col = cols_lower.get("name", cols_lower.get("student name", None))

    if not id_col or not roll_col or not name_col:
        st.error("Score file must contain columns named ID, Roll and Name (or similar).")
        return None
    df = df.rename(columns={id_col: "ID", roll_col: "Roll", name_col: "Name"})
    return df


# ----------------------------
# Cleaning & Transformations
# ----------------------------

def melt_attendance(att_df):
    id_vars = [c for c in ["ID", "Roll", "Name", "Gender"] if c in att_df.columns]
    value_vars = [c for c in att_df.columns if c not in id_vars]

    long = att_df.melt(id_vars=id_vars, value_vars=value_vars, var_name="Date", value_name="Status")
    try:
        long["Date_parsed"] = pd.to_datetime(long["Date"], dayfirst=True, errors="coerce")
    except Exception:
        long["Date_parsed"] = pd.to_datetime(long["Date"], errors="coerce")
    long["RawDateLabel"] = long["Date"].astype(str)

    def status_to_present(x):
        s = str(x).strip().upper()
        if s == "" or s == "NAN":
            return np.nan
        if s.startswith("✔") or s.startswith("P") or s.startswith("1") or s in ["PRESENT", "YES"]:
            return 1
        if s.startswith("A") or s in ["ABSENT", "0", "NO"]:
            return 0
        try:
            v = float(s)
            return 1 if v > 0 else 0
        except Exception:
            return np.nan

    long["Present"] = long["Status"].apply(status_to_present)
    return long


def extract_wmt_columns(score_df):
    wmt_cols = [c for c in score_df.columns if "wmt" in c.lower()]
    if not wmt_cols:
        wmt_cols = [c for c in score_df.columns if any(k in c.lower() for k in ["test", "assess", "exam"])]
    return wmt_cols

# ----------------------------
# UI
# ----------------------------

def main():
    st.title("🚀 Awesome Interactive Student Dashboard")

    st.sidebar.header("Uploads & Settings")
    boys_file = st.sidebar.file_uploader("Boys attendance (xlsx/csv)", type=["xlsx", "csv"], key="boys")
    girls_file = st.sidebar.file_uploader("Girls attendance (xlsx/csv)", type=["xlsx", "csv"], key="girls")
    score_file = st.sidebar.file_uploader("Score file (xlsx/csv)", type=["xlsx", "csv"], key="scores")

    if not (boys_file and girls_file and score_file):
        st.info("Upload Boys, Girls attendance files and Score file in the sidebar to begin.")
        st.stop()

    boys_df = read_attendance_file(boys_file, "Boy")
    girls_df = read_attendance_file(girls_file, "Girl")
    score_df = read_score_file(score_file)

    if any(x is None for x in [boys_df, girls_df, score_df]):
        st.stop()

    attendance_df = pd.concat([boys_df, girls_df], ignore_index=True, sort=False)
    attendance_long = melt_attendance(attendance_df)

    group_keys = [c for c in ["ID", "Name", "Roll", "Gender"] if c in attendance_long.columns]
    attendance_summary = (
        attendance_long.groupby(group_keys)['Present']
        .agg(['mean', 'count'])
        .reset_index()
        .rename(columns={'mean': 'AttendanceRate', 'count': 'RecordedSessions'})
    )

    wmt_cols = extract_wmt_columns(score_df)
    if not wmt_cols:
        chosen = st.multiselect("Pick score columns", options=list(score_df.columns), default=[c for c in score_df.columns if c not in ['ID','Roll','Name']][:3])
        wmt_cols = chosen

    score_long = score_df.melt(id_vars=[c for c in ['ID','Roll','Name'] if c in score_df.columns], value_vars=wmt_cols, var_name='WMT', value_name='Score')
    score_long['Score'] = pd.to_numeric(score_long['Score'].astype(str).str.extract(r'(\d+\.?\d*)')[0], errors='coerce').fillna(0)

    combined = pd.merge(score_long, attendance_summary, how='left', on=[c for c in ['ID','Name','Roll'] if c in score_long.columns and c in attendance_summary.columns])
    combined['AttendanceRate'] = combined['AttendanceRate'].fillna(0)

    st.header("Key performance indicators")
    total_students = combined['ID'].nunique()
    avg_score = combined['Score'].mean()
    avg_att = combined['AttendanceRate'].mean()

    k1, k2, k3 = st.columns(3)
    k1.metric("Students", total_students)
    k2.metric("Avg Score", f"{avg_score:.2f}")
    k3.metric("Avg Attendance", f"{avg_att*100:.1f}%")

    st.header("Visual Insights")
    st.subheader("Score distribution")
    fig_dist = px.histogram(combined, x='Score', nbins=25, marginal='box')
    st.plotly_chart(fig_dist, use_container_width=True)

    st.subheader("Attendance vs Score")
    if 'Gender' in combined.columns:
        fig_scatter = px.scatter(combined, x='AttendanceRate', y='Score', color='Gender', hover_data=['Name','Roll','WMT'], trendline='ols')
    else:
        fig_scatter = px.scatter(combined, x='AttendanceRate', y='Score', hover_data=['Name','Roll','WMT'], trendline='ols')
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.subheader("Attendance trend")
    if attendance_long['Date_parsed'].notna().any():
        trend = attendance_long.groupby('Date_parsed').Present.mean().reset_index().sort_values('Date_parsed')
        fig_trend = px.line(trend, x='Date_parsed', y='Present', markers=True)
        st.plotly_chart(fig_trend, use_container_width=True)
    else:
        trend = attendance_long.groupby('RawDateLabel').Present.mean().reset_index()
        fig_trend = px.bar(trend, x='RawDateLabel', y='Present')
        st.plotly_chart(fig_trend, use_container_width=True)

    st.subheader("Leaderboard")
    leaderboard = combined.groupby(['ID','Name','Roll']).agg(AvgScore=('Score','mean'), AttendanceRate=('AttendanceRate','mean')).reset_index()
    leaderboard = leaderboard.sort_values(['AvgScore','AttendanceRate'], ascending=[False, False])
    st.dataframe(leaderboard.head(20).style.format({"AvgScore": "{:.1f}", "AttendanceRate": "{:.1%}"}))

    st.header("Download Data")
    csv = combined.to_csv(index=False).encode('utf-8')
    st.download_button(label="Download CSV", data=csv, file_name='student_dashboard_export.csv', mime='text/csv')


if __name__ == "__main__":
    main()
