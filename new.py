# new.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Class Dashboard", layout="wide")

# -----------------
# LOAD DATA
# -----------------
@st.cache_data
def load_data():
    attendance = pd.read_csv("clean_attendance.csv")
    marks = pd.read_csv("clean_marks.csv")

    # Ensure clean merge keys
    attendance["ID"] = attendance["ID"].astype(str)
    marks["ID"] = marks["ID"].astype(str)

    return attendance, marks

attendance, marks = load_data()

# -----------------
# ATTENDANCE SUMMARY
# -----------------
attendance_summary = (
    attendance.groupby(["ID", "Name", "Gender"])
    .Status.value_counts(normalize=True)
    .rename("Rate")
    .reset_index()
    .pivot(index=["ID", "Name", "Gender"], columns="Status", values="Rate")
    .fillna(0)
    .reset_index()
)

attendance_summary["Attendance %"] = (attendance_summary["Present"] * 100).round(2)

# -----------------
# MARKS SUMMARY
# -----------------
marks_summary = (
    marks.groupby(["ID", "Name", "Subject"])
    .agg(
        AvgMarks=("Marks", "mean"),
        BestScore=("Marks", "max"),
        ExamsTaken=("Marks", "count"),
        AbsentCount=("WasAbsent", "sum")
    )
    .reset_index()
)

# Student-level avg
student_avg = (
    marks.groupby(["ID", "Name"])
    .agg(AverageMarks=("Marks", "mean"))
    .reset_index()
)

# -----------------
# MERGE DATA
# -----------------
merged = pd.merge(
    attendance_summary,
    student_avg,
    on="ID",
    how="left"
)

# Keep consistent Name column
if "Name_y" in merged.columns:
    merged["Name"] = merged["Name_x"].combine_first(merged["Name_y"])
    merged = merged.drop(columns=["Name_x", "Name_y"])

# -----------------
# DASHBOARD
# -----------------
st.title("📊 Class Dashboard")

tab1, tab2, tab3 = st.tabs(["📅 Attendance", "📝 Marks", "📚 Combined"])

# --- Attendance Tab ---
with tab1:
    st.header("Attendance Overview")
    st.dataframe(attendance_summary)

    fig, ax = plt.subplots()
    attendance_summary["Attendance %"].hist(ax=ax, bins=10)
    ax.set_title("Distribution of Attendance %")
    ax.set_xlabel("Attendance %")
    ax.set_ylabel("Number of Students")
    st.pyplot(fig)

    st.markdown("**Comment:** Students with attendance below 75% may need follow-up.")
    with st.expander("See explanation"):
        st.write("Attendance percentage is calculated as the ratio of 'Present' entries to total entries per student.")

# --- Marks Tab ---
with tab2:
    st.header("Marks Overview")
    st.dataframe(marks_summary)

    avg_marks = student_avg["AverageMarks"].dropna()
    fig, ax = plt.subplots()
    avg_marks.hist(ax=ax, bins=10)
    ax.set_title("Distribution of Average Marks")
    ax.set_xlabel("Average Marks")
    ax.set_ylabel("Number of Students")
    st.pyplot(fig)

    st.markdown("**Comment:** The marks distribution shows performance clustering. Identify low performers for extra support.")
    with st.expander("See explanation"):
        st.write("Marks are averaged across all subjects and exams for each student.")

# --- Combined Tab ---
with tab3:
    st.header("Attendance vs Marks")
    st.dataframe(merged)

    fig, ax = plt.subplots()
    ax.scatter(merged["Attendance %"], merged["AverageMarks"])
    ax.set_title("Attendance vs Average Marks")
    ax.set_xlabel("Attendance %")
    ax.set_ylabel("Average Marks")
    st.pyplot(fig)

    st.markdown("**Comment:** There is often a positive correlation between attendance and marks. Outliers may need special review.")
    with st.expander("See explanation"):
        st.write("This scatterplot compares each student's overall attendance percentage with their average marks across subjects.")
