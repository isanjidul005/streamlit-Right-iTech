import streamlit as st
import pandas as pd

st.set_page_config(page_title="Student Dashboard", layout="wide")

# ----------------------------
# File reader
# ----------------------------
def read_attendance_file(file):
    """Read and clean attendance file"""
    df_raw = pd.read_excel(file, header=None)

    # Drop the first row (report title)
    df = df_raw.drop(0).reset_index(drop=True)

    # Use the next row as header
    df.columns = df.iloc[0]
    df = df.drop(0).reset_index(drop=True)

    # If columns still look like Unnamed, rename manually
    cols = list(df.columns)
    if all(str(c).startswith("Unnamed") or pd.isna(c) for c in cols):
        new_cols = ["ID", "Roll", "Name"] + [f"Day{i}" for i in range(1, len(cols)-2+1)]
        df.columns = new_cols
    else:
        # Otherwise enforce first 3 cols
        df.rename(columns={cols[0]: "ID", cols[1]: "Roll", cols[2]: "Name"}, inplace=True)

    return df

def read_score_file(file):
    """Read score file normally"""
    return pd.read_excel(file)

# ----------------------------
# Main
# ----------------------------
def main():
    st.title("📊 Student Performance Dashboard")

    st.sidebar.header("Upload Files")
    boys_file = st.sidebar.file_uploader("Upload Boys Attendance", type=["xlsx"])
    girls_file = st.sidebar.file_uploader("Upload Girls Attendance", type=["xlsx"])
    result_file = st.sidebar.file_uploader("Upload Score File", type=["xlsx"])

    if boys_file and girls_file:
        boys_df = read_attendance_file(boys_file)
        girls_df = read_attendance_file(girls_file)
        st.success("✅ Attendance files processed")
        with st.expander("Preview Attendance (Boys)"):
            st.dataframe(boys_df.head())
        with st.expander("Preview Attendance (Girls)"):
            st.dataframe(girls_df.head())

    if result_file:
        score_df = read_score_file(result_file)
        st.success("✅ Score file processed")
        with st.expander("Preview Scores"):
            st.dataframe(score_df.head())

if __name__ == "__main__":
    main()
