import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Student Dashboard", layout="wide")

# ----------------------------
# File readers
# ----------------------------
@st.cache_data
def read_data_file(file, header=None):
    """Read CSV or Excel files based on their MIME type, handling encoding errors."""
    file_type = file.type
    
    file.seek(0)
    
    if file_type == 'text/csv':
        try:
            return pd.read_csv(file, header=header, encoding='utf-8')
        except UnicodeDecodeError:
            file.seek(0)
            return pd.read_csv(file, header=header, encoding='latin1')
    elif file_type == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':
        return pd.read_excel(file, header=header)
    else:
        st.error(f"Unsupported file format: {file.name} (MIME type: {file_type})")
        return None

def read_attendance_file(file, gender):
    """Read and clean attendance file, fix messy headers, tag gender"""
    df = read_data_file(file, header=1)
    if df is None:
        return None
    
    # Standardize column names by stripping spaces
    df.columns = df.columns.astype(str).str.strip()
    
    # Rename columns to ensure consistency
    df.rename(columns={df.columns[0]: 'ID', df.columns[1]: 'Roll', df.columns[2]: 'Name'}, inplace=True)
    
    df["Gender"] = gender
    return df

def read_score_file(file):
    """Read and clean score file with single-row header"""
    df = read_data_file(file, header=0)
    if df is None:
        return None
    
    # Standardize column names to remove extra spaces
    df.columns = df.columns.astype(str).str.strip()
    
    # Check for required columns
    required_cols = {'ID', 'Roll', 'Name'}
    if not required_cols.issubset(df.columns):
        st.error(f"The score file does not contain the required columns: {required_cols - set(df.columns)}")
        return None

    return df

# ----------------------------
# Dashboard
# ----------------------------
def main():
    st.title("📊 Student Performance Dashboard")

    st.sidebar.header("Upload Files")
    boys_file = st.sidebar.file_uploader("Upload Boys Attendance", type=["xlsx", "csv"])
    girls_file = st.sidebar.file_uploader("Upload Girls Attendance", type=["xlsx", "csv"])
    result_file = st.sidebar.file_uploader("Upload Score File", type=["xlsx", "csv"])

    if boys_file and girls_file and result_file:
        # Load data
        boys_df = read_attendance_file(boys_file, "Boy")
        girls_df = read_attendance_file(girls_file, "Girl")
        score_df = read_score_file(result_file)

        if boys_df is None or girls_df is None or score_df is None:
            st.stop()
            
        # Combine attendance
        attendance_df = pd.concat([boys_df, girls_df], ignore_index=True)
        
        # Melt attendance into long form
        attendance_long = attendance_df.melt(
            id_vars=["ID", "Roll", "Name", "Gender"],
            var_name="Date",
            value_name="Status"
        )

        # Clean up attendance (P = present, A = absent, etc.)
        attendance_long["Present"] = attendance_long["Status"].apply(
            lambda x: 1 if str(x).strip().upper().startswith("✔") or str(x).strip().upper() == "P" else 0
        )

        # Attendance summary per student
        attendance_summary = (
            attendance_long.groupby(["ID", "Name", "Roll", "Gender"])["Present"]
            .mean()
            .reset_index()
        )
        attendance_summary.rename(columns={"Present": "AttendanceRate"}, inplace=True)

        # Process scores file
        score_columns = [col for col in score_df.columns if 'WMT' in col.upper()]
        
        # Melt scores into long format
        score_long = score_df.melt(
            id_vars=['ID', 'Roll', 'Name'],
            value_vars=score_columns,
            var_name='WMT',
            value_name='Score'
        )
        
        # Clean score data and convert to numeric, handling 'ab'
        score_long['Score'] = pd.to_numeric(
            score_long['Score'].astype(str).str.extract(r'(\d+\.?\d*)').fillna('0'),
            errors='coerce'
        ).fillna(0)
        
        # Merge with scores
        combined = pd.merge(
            score_long,
            attendance_summary,
            how="left",
            on=["ID", "Name", "Roll"]
        )
        
        # ---------------- Overview ----------------
        st.header("📍 Overview")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Students", len(combined['ID'].unique()))
        col2.metric("Avg Score", f"{combined['Score'].mean():.2f}")
        col3.metric("Avg Attendance", f"{combined['AttendanceRate'].mean()*100:.1f}%")

        # ---------------- Gender Comparison ----------------
        st.header("👫 Gender Comparison")
        gender_summary = combined.groupby("Gender").agg(
            AvgScore=("Score", "mean"),
            AvgAttendance=("AttendanceRate", "mean"),
            Count=("ID", "count")
        ).reset_index()

        st.dataframe(gender_summary)

        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        sns.barplot(data=gender_summary, x="Gender", y="AvgScore", ax=ax[0])
        ax[0].set_title("Average Score by Gender")
        sns.barplot(data=gender_summary, x="Gender", y="AvgAttendance", ax=ax[1])
        ax[1].set_title("Average Attendance by Gender")
        st.pyplot(fig)

        # ---------------- Attendance vs Score ----------------
        st.header("📈 Attendance vs Score")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.scatterplot(
            data=combined,
            x="AttendanceRate",
            y="Score",
            hue="Gender",
            ax=ax
        )
        st.pyplot(fig)

        # ---------------- Trends ----------------
        st.header("📅 Attendance Trend Over Time")
        trend = attendance_long.groupby("Date")["Present"].mean().reset_index()
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.lineplot(data=trend, x="Date", y="Present", marker="o", ax=ax)
        ax.set_ylabel("Attendance Rate")
        st.pyplot(fig)

        # ---------------- Data Previews ----------------
        with st.expander("Preview: Attendance (long format)"):
            st.dataframe(attendance_long.head())
        with st.expander("Preview: Score File"):
            st.dataframe(score_df.head())
        with st.expander("Preview: Combined Data"):
            st.dataframe(combined.head())

    else:
        st.info("⬅️ Upload all three files to start.")

# ----------------------------
if __name__ == "__main__":
    main()
