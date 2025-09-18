import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------
# App Configuration
# -----------------------------
st.set_page_config(page_title="Right iTech", layout="wide")
st.title("📊 Right iTech Dashboard")

# -----------------------------
# File Upload Sidebar
# -----------------------------
st.sidebar.header("Upload Your Data")
attendance_file = st.sidebar.file_uploader("Upload Attendance CSV", type=["csv"])
marks_file = st.sidebar.file_uploader("Upload Marks CSV", type=["csv"])

# -----------------------------
# Load Data
# -----------------------------
if attendance_file is not None and marks_file is not None:
    attendance_df = pd.read_csv(attendance_file)
    marks_df = pd.read_csv(marks_file)

    # Ensure Date column is datetime
    if "Date" in attendance_df.columns:
        attendance_df["Date"] = pd.to_datetime(attendance_df["Date"], errors="coerce")

    # Assign subject colors dynamically based on dataset
    unique_subjects = marks_df["Subject"].unique() if "Subject" in marks_df.columns else []
    color_map = {subj: px.colors.qualitative.Set2[i % len(px.colors.qualitative.Set2)]
                 for i, subj in enumerate(unique_subjects)}

    # -----------------------------
    # Tabs Layout
    # -----------------------------
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📌 Class Overview", "📝 Marks Analysis", "📅 Attendance", "⚖️ Comparison"]
    )

    # -----------------------------
    # CLASS OVERVIEW TAB
    # -----------------------------
    with tab1:
        st.subheader("📌 Class Overview")

        # Basic class statistics
        total_students = attendance_df["Student ID"].nunique()
        girls = attendance_df[attendance_df["Gender"].str.lower() == "female"]["Student ID"].nunique() if "Gender" in attendance_df.columns else 0
        boys = attendance_df[attendance_df["Gender"].str.lower() == "male"]["Student ID"].nunique() if "Gender" in attendance_df.columns else 0
        avg_attendance = attendance_df["Status"].eq("Present").mean() * 100
        avg_absent = 100 - avg_attendance

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total Students", total_students)
        col2.metric("Girls", girls)
        col3.metric("Boys", boys)
        col4.metric("Avg Attendance %", f"{avg_attendance:.1f}%")
        col5.metric("Avg Absent %", f"{avg_absent:.1f}%")

        st.markdown("---")

        # Attendance distribution pie
        att_pie = attendance_df["Status"].value_counts(normalize=True).reset_index()
        att_pie.columns = ["Status", "Percentage"]
        fig_pie = px.pie(att_pie, values="Percentage", names="Status",
                         title="Overall Attendance Distribution",
                         color="Status", color_discrete_map={"Present": "green", "Absent": "red"})
        st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown("---")

        # Top performers
        top_students = marks_df.groupby("Student ID")["Marks"].mean().nlargest(10).reset_index()
        fig_top = px.bar(top_students, x="Student ID", y="Marks",
                         title="Top 10 Performers (Avg Marks)",
                         color="Marks", color_continuous_scale="Viridis")
        st.plotly_chart(fig_top, use_container_width=True)

        # Bottom performers
        bottom_students = marks_df.groupby("Student ID")["Marks"].mean().nsmallest(10).reset_index()
        fig_bottom = px.bar(bottom_students, x="Student ID", y="Marks",
                            title="Bottom 10 Performers (Avg Marks)",
                            color="Marks", color_continuous_scale="Reds")
        st.plotly_chart(fig_bottom, use_container_width=True)

        st.markdown("---")

        # Marks trend across exams
        if "Exam" in marks_df.columns:
            marks_trend = marks_df.groupby("Exam")["Marks"].mean().reset_index()
            fig_trend = px.line(marks_trend, x="Exam", y="Marks",
                                title="Average Marks Trend Across Exams",
                                markers=True)
            st.plotly_chart(fig_trend, use_container_width=True)

    # -----------------------------
    # MARKS TAB
    # -----------------------------
    with tab2:
        st.subheader("📝 Marks Analysis")

        if not marks_df.empty:
            avg_subject = marks_df.groupby("Subject")["Marks"].mean().reset_index()
            fig_avg_subject = px.bar(avg_subject, x="Subject", y="Marks",
                                     title="Average Marks by Subject",
                                     color="Subject", color_discrete_map=color_map)
            st.plotly_chart(fig_avg_subject, use_container_width=True)

            if "Exam" in marks_df.columns:
                subject_exam = marks_df.groupby(["Exam", "Subject"])["Marks"].mean().reset_index()
                fig_exam = px.line(subject_exam, x="Exam", y="Marks", color="Subject",
                                   title="Marks Trend by Subject Across Exams",
                                   markers=True, color_discrete_map=color_map)
                st.plotly_chart(fig_exam, use_container_width=True)

    # -----------------------------
    # ATTENDANCE TAB
    # -----------------------------
    with tab3:
        st.subheader("📅 Attendance Analysis")

        include_weekends = st.checkbox("Include Fridays (Weekends)", value=False)
        if not include_weekends and "Date" in attendance_df.columns:
            attendance_df = attendance_df[attendance_df["Date"].dt.day_name() != "Friday"]

        att_over_time = attendance_df.groupby("Date")["Status"].apply(lambda x: (x == "Present").mean()).reset_index()
        att_over_time.columns = ["Date", "Attendance Rate"]

        fig_att_time = px.line(att_over_time, x="Date", y="Attendance Rate",
                               title="Attendance Rate Over Time",
                               markers=True)
        fig_att_time.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig_att_time, use_container_width=True)

        # Daily attendance heatmap
        if "Student ID" in attendance_df.columns and "Date" in attendance_df.columns:
            pivot = attendance_df.pivot_table(index="Student ID", columns="Date", values="Status",
                                              aggfunc=lambda x: (x == "Present").mean(), fill_value=0)
            fig_heatmap = px.imshow(pivot, aspect="auto", color_continuous_scale="Blues",
                                    title="Daily Attendance Heatmap (Students vs Dates)")
            st.plotly_chart(fig_heatmap, use_container_width=True)

    # -----------------------------
    # COMPARISON TAB
    # -----------------------------
    with tab4:
        st.subheader("⚖️ Student Comparison")

        student_options = marks_df["Student ID"].unique()
        selected_students = st.multiselect("Select students to compare", student_options)

        if selected_students:
            compare_df = marks_df[marks_df["Student ID"].isin(selected_students)]
            compare_avg = compare_df.groupby(["Student ID", "Subject"])["Marks"].mean().reset_index()
            fig_compare = px.bar(compare_avg, x="Subject", y="Marks", color="Student ID",
                                 barmode="group",
                                 title="Comparison of Average Marks by Subject")
            st.plotly_chart(fig_compare, use_container_width=True)

            if "Exam" in compare_df.columns:
                compare_exam = compare_df.groupby(["Exam", "Student ID"])["Marks"].mean().reset_index()
                fig_exam_compare = px.line(compare_exam, x="Exam", y="Marks", color="Student ID",
                                           markers=True,
                                           title="Exam-wise Performance Comparison")
                st.plotly_chart(fig_exam_compare, use_container_width=True)

else:
    st.warning("No data detected. Upload CSVs in the sidebar.")
