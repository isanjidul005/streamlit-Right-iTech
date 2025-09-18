import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# -------------------------------
# App Configuration
# -------------------------------
st.set_page_config(page_title="Right iTech", layout="wide")

# App title
st.markdown("<h1 style='text-align: center;'>Right iTech</h1>", unsafe_allow_html=True)

# -------------------------------
# File Upload Section
# -------------------------------
st.sidebar.header("Upload Your Data")
attendance_file = st.sidebar.file_uploader("Upload Attendance CSV", type=["csv"])
marks_file = st.sidebar.file_uploader("Upload Marks CSV", type=["csv"])

if attendance_file and marks_file:
    attendance_df = pd.read_csv(attendance_file)
    marks_df = pd.read_csv(marks_file)

    # Parse date column safely
    if "Date" in attendance_df.columns:
        attendance_df["Date"] = pd.to_datetime(attendance_df["Date"], errors="coerce")

    # -------------------------------
    # Dynamic subject colors
    # -------------------------------
    if "Subject" in marks_df.columns:
        subjects = sorted(marks_df["Subject"].unique())
        color_palette = px.colors.qualitative.Set2  # Nice, professional palette
        subject_colors = {
            subj: color_palette[i % len(color_palette)]
            for i, subj in enumerate(subjects)
        }
    else:
        subject_colors = {}

    # -------------------------------
    # Layout Tabs
    # -------------------------------
    tabs = st.tabs([
        "Class Overview",
        "Student Dashboard",
        "Compare Students",
        "Attendance",
        "Marks",
        "Insights"
    ])

    # ======================================================
    # CLASS OVERVIEW TAB
    # ======================================================
    with tabs[0]:
        st.subheader("Class Overview")

        # Total students
        total_students = attendance_df["Student ID"].nunique()

        # Boys & Girls counts
        if "Gender" in attendance_df.columns:
            boys_count = (attendance_df["Gender"].str.lower() == "male").sum()
            girls_count = (attendance_df["Gender"].str.lower() == "female").sum()
        else:
            boys_count, girls_count = 0, 0

        # Attendance rate
        if "Status" in attendance_df.columns:
            avg_attendance = (attendance_df["Status"].str.lower() == "present").mean() * 100
            avg_absent = 100 - avg_attendance
        else:
            avg_attendance, avg_absent = 0, 0

        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Students", total_students)
        col2.metric("Boys", boys_count)
        col3.metric("Girls", girls_count)
        col4.metric("Avg Attendance", f"{avg_attendance:.1f}%")

        # Average marks by subject
        if not marks_df.empty and "Subject" in marks_df.columns:
            avg_subjects = marks_df.groupby("Subject")["Marks"].mean().reset_index()
            fig = px.bar(
                avg_subjects,
                x="Subject",
                y="Marks",
                color="Subject",
                color_discrete_map=subject_colors,
                title="Average Marks by Subject"
            )
            st.plotly_chart(fig, use_container_width=True)

    # ======================================================
    # STUDENT DASHBOARD TAB
    # ======================================================
    with tabs[1]:
        st.subheader("Student Dashboard")

        student_ids = attendance_df["Student ID"].unique()
        student_choice = st.selectbox("Select a Student", student_ids)

        student_att = attendance_df[attendance_df["Student ID"] == student_choice]
        student_marks = marks_df[marks_df["Student ID"] == student_choice]

        st.markdown(f"**Student ID:** {student_choice}")
        if "Name" in student_att.columns:
            st.markdown(f"**Name:** {student_att['Name'].iloc[0]}")

        # Attendance trend
        if not student_att.empty:
            fig = px.line(
                student_att,
                x="Date",
                y=student_att["Status"].str.lower().eq("present").astype(int),
                title="Attendance Over Time"
            )
            st.plotly_chart(fig, use_container_width=True)

        # Marks trend
        if not student_marks.empty:
            fig = px.bar(
                student_marks,
                x="Subject",
                y="Marks",
                color="Subject",
                color_discrete_map=subject_colors,
                title="Marks by Subject"
            )
            st.plotly_chart(fig, use_container_width=True)

    # ======================================================
    # COMPARE STUDENTS TAB
    # ======================================================
    with tabs[2]:
        st.subheader("Compare Students")

        selected_students = st.multiselect(
            "Select students to compare",
            options=attendance_df["Student ID"].unique(),
            default=attendance_df["Student ID"].unique()[:2]
        )

        if selected_students:
            comp_marks = marks_df[marks_df["Student ID"].isin(selected_students)]

            # Average marks per subject per student
            if not comp_marks.empty:
                fig = px.bar(
                    comp_marks,
                    x="Subject",
                    y="Marks",
                    color="Student ID",
                    barmode="group",
                    title="Student Comparison by Subject"
                )
                st.plotly_chart(fig, use_container_width=True)

            # Add radar chart (marks profile)
            avg_scores = comp_marks.groupby(["Student ID", "Subject"])["Marks"].mean().reset_index()
            fig = px.line_polar(
                avg_scores,
                r="Marks",
                theta="Subject",
                color="Student ID",
                line_close=True,
                title="Marks Radar Comparison"
            )
            st.plotly_chart(fig, use_container_width=True)

    # ======================================================
    # ATTENDANCE TAB
    # ======================================================
    with tabs[3]:
        st.subheader("Attendance Analysis")

        include_weekends = st.toggle("Include Weekends", value=False)

        att = attendance_df.copy()
        if not include_weekends:
            att = att[att["Date"].dt.day_name() != "Friday"]

        # Daily average attendance
        if not att.empty:
            att_daily = att.groupby("Date")["Status"].apply(lambda x: (x.str.lower() == "present").mean()).reset_index()
            att_daily["Status"] = att_daily["Status"] * 100

            fig = px.line(
                att_daily,
                x="Date",
                y="Status",
                title="Daily Average Attendance (%)"
            )
            st.plotly_chart(fig, use_container_width=True)

        # Attendance distribution (Present vs Absent)
        if not att.empty:
            status_counts = att["Status"].value_counts().reset_index()
            fig = px.pie(
                status_counts,
                names="index",
                values="Status",
                title="Overall Attendance Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)

        # Attendance by weekday
        if not att.empty:
            att["Weekday"] = att["Date"].dt.day_name()
            weekday_att = att.groupby("Weekday")["Status"].apply(lambda x: (x.str.lower() == "present").mean() * 100).reset_index()

            fig = px.bar(
                weekday_att,
                x="Weekday",
                y="Status",
                title="Average Attendance by Weekday"
            )
            st.plotly_chart(fig, use_container_width=True)

    # ======================================================
    # MARKS TAB
    # ======================================================
    with tabs[4]:
        st.subheader("Marks Analysis")

        if not marks_df.empty:
            exam_filter = st.selectbox("Select Exam", options=marks_df["Exam"].unique())
            exam_marks = marks_df[marks_df["Exam"] == exam_filter]

            fig = px.box(
                exam_marks,
                x="Subject",
                y="Marks",
                color="Subject",
                color_discrete_map=subject_colors,
                title=f"Marks Distribution for {exam_filter}"
            )
            st.plotly_chart(fig, use_container_width=True)

    # ======================================================
    # INSIGHTS TAB
    # ======================================================
    with tabs[5]:
        st.subheader("Insights")

        if not marks_df.empty:
            subj_perf = marks_df.groupby("Subject")["Marks"].mean().reset_index()
            top_subject = subj_perf.sort_values(by="Marks", ascending=False).iloc[0]
            st.info(f"Best Performing Subject: **{top_subject['Subject']}** with avg {top_subject['Marks']:.2f} marks")

            worst_subject = subj_perf.sort_values(by="Marks", ascending=True).iloc[0]
            st.info(f"Lowest Performing Subject: **{worst_subject['Subject']}** with avg {worst_subject['Marks']:.2f} marks")

else:
    st.warning("Please upload both Attendance and Marks datasets to continue.")
