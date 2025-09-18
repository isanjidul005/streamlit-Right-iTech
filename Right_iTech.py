import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# -----------------------
# App Config
# -----------------------
st.set_page_config(page_title="Right iTech", layout="wide")

# -----------------------
# Title
# -----------------------
st.markdown(
    """
    <h1 style='text-align: center; color: #2E86C1;'>Right iTech</h1>
    <p style='text-align: center; font-size:18px; color: gray;'>Professional, readable analytics for marks & attendance — interactive visuals.</p>
    """,
    unsafe_allow_html=True
)

# -----------------------
# Sidebar File Uploads
# -----------------------
st.sidebar.header("Upload your datasets")
attendance_file = st.sidebar.file_uploader("Upload Attendance CSV", type=["csv"])
marks_file = st.sidebar.file_uploader("Upload Marks CSV", type=["csv"])

attendance_df, marks_df = pd.DataFrame(), pd.DataFrame()

if attendance_file is not None:
    attendance_df = pd.read_csv(attendance_file)
    if "Date" in attendance_df.columns:
        attendance_df["Date"] = pd.to_datetime(attendance_df["Date"], errors="coerce")

if marks_file is not None:
    marks_df = pd.read_csv(marks_file)

# -----------------------
# Tabs
# -----------------------
tabs = st.tabs(["Class Overview", "Student Dashboard", "Compare Students", "Attendance", "Marks", "Insights"])

# -----------------------
# Tab 1: Class Overview
# -----------------------
with tabs[0]:
    st.subheader("Class Overview")

    if not attendance_df.empty and not marks_df.empty:
        total_students = attendance_df["Name"].nunique()

        # Boys/Girls detection (if Gender col exists)
        boys = 0
        girls = 0
        if "Gender" in attendance_df.columns:
            boys = (attendance_df["Gender"].str.lower() == "male").sum()
            girls = (attendance_df["Gender"].str.lower() == "female").sum()

        # Attendance summary
        avg_attendance = attendance_df.groupby("Name")["Status"].apply(lambda x: (x == "Present").mean()).mean()
        avg_present = (attendance_df["Status"] == "Present").mean()
        avg_absent = (attendance_df["Status"] == "Absent").mean()

        cols = st.columns(4)
        cols[0].metric("Total students", total_students)
        cols[1].metric("Boys", boys)
        cols[2].metric("Girls", girls)
        cols[3].metric("Avg attendance", f"{avg_attendance*100:.1f}%")

        st.metric("Avg Present", f"{avg_present*100:.1f}%")
        st.metric("Avg Absent", f"{avg_absent*100:.1f}%")

        # Threshold-based grouped visualization
        st.subheader("Students above / below threshold (average marks)")
        threshold = st.slider("Threshold for average marks", 0, 100, 40)

        marks_summary = marks_df.groupby("Name")["Marks"].mean().reset_index()
        marks_summary["Category"] = marks_summary["Marks"].apply(lambda x: f">= {threshold}" if x >= threshold else f"< {threshold}")

        fig = px.histogram(
            marks_summary,
            x="Category",
            color="Category",
            barmode="group",
            text_auto=True,
            title=f"Distribution of students above/below threshold {threshold}"
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("Please upload both Attendance and Marks CSV files to see the overview.")

# -----------------------
# Tab 2: Student Dashboard
# -----------------------
with tabs[1]:
    st.subheader("Student Dashboard")
    if not attendance_df.empty and not marks_df.empty:
        student_list = attendance_df["Name"].unique().tolist()
        selected_student = st.selectbox("Select a student", student_list)

        st.write(f"### Dashboard for {selected_student}")

        # Attendance over time
        student_attendance = attendance_df[attendance_df["Name"] == selected_student]
        if not student_attendance.empty:
            att_trend = student_attendance.groupby("Date")["Status"].apply(lambda x: (x == "Present").mean()).reset_index(name="Attendance Rate")
            fig = px.line(att_trend, x="Date", y="Attendance Rate", title="Daily Attendance Rate")
            st.plotly_chart(fig, use_container_width=True)

        # Marks across subjects
        student_marks = marks_df[marks_df["Name"] == selected_student]
        if not student_marks.empty:
            fig = px.line(student_marks, x="Exam", y="Marks", color="Subject", markers=True, title="Marks across Exams by Subject")
            st.plotly_chart(fig, use_container_width=True)

# -----------------------
# Tab 3: Compare Students
# -----------------------
with tabs[2]:
    st.subheader("Compare Students")
    if not marks_df.empty:
        selected_students = st.multiselect("Select students to compare", marks_df["Name"].unique())

        if selected_students:
            compare_df = marks_df[marks_df["Name"].isin(selected_students)]
            fig = px.bar(compare_df, x="Subject", y="Marks", color="Name", barmode="group", title="Marks Comparison by Subject")
            st.plotly_chart(fig, use_container_width=True)

# -----------------------
# Tab 4: Attendance
# -----------------------
with tabs[3]:
    st.subheader("Attendance Trends")
    if not attendance_df.empty:
        # Daily overall attendance
        daily_attendance = attendance_df.groupby("Date")["Status"].apply(lambda x: (x == "Present").mean()).reset_index(name="Attendance Rate")
        fig = px.line(daily_attendance, x="Date", y="Attendance Rate", title="Class Attendance Over Time")
        st.plotly_chart(fig, use_container_width=True)

        # Student-wise daily
        selected_students = st.multiselect("Select students for daily attendance", attendance_df["Name"].unique())
        if selected_students:
            daily_by_student = attendance_df[attendance_df["Name"].isin(selected_students)].groupby(["Date", "Name"])["Status"].apply(lambda x: (x == "Present").mean()).reset_index(name="Attendance Rate")
            fig = px.line(daily_by_student, x="Date", y="Attendance Rate", color="Name", title="Daily Attendance by Student")
            st.plotly_chart(fig, use_container_width=True)

# -----------------------
# Tab 5: Marks
# -----------------------
with tabs[4]:
    st.subheader("Marks Overview")
    if not marks_df.empty:
        # Average marks per subject
        subj_avg = marks_df.groupby("Subject")["Marks"].mean().reset_index()
        fig = px.bar(subj_avg, x="Subject", y="Marks", color="Subject", title="Average Marks per Subject")
        st.plotly_chart(fig, use_container_width=True)

        # Daily / exam-wise marks
        exam_trend = marks_df.groupby(["Exam", "Subject"])["Marks"].mean().reset_index()
        fig = px.line(exam_trend, x="Exam", y="Marks", color="Subject", markers=True, title="Marks Across Exams by Subject")
        st.plotly_chart(fig, use_container_width=True)

# -----------------------
# Tab 6: Insights
# -----------------------
with tabs[5]:
    st.subheader("Insights")
    if not attendance_df.empty and not marks_df.empty:
        # Correlation between avg marks & attendance
        student_attendance = attendance_df.groupby("Name")["Status"].apply(lambda x: (x == "Present").mean()).reset_index(name="Attendance Rate")
        student_marks = marks_df.groupby("Name")["Marks"].mean().reset_index(name="Avg Marks")
        merged = pd.merge(student_attendance, student_marks, on="Name")

        fig = px.scatter(merged, x="Attendance Rate", y="Avg Marks", text="Name", trendline="ols", title="Correlation between Attendance and Average Marks")
        st.plotly_chart(fig, use_container_width=True)
