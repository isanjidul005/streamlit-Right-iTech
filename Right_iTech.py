import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------
# App Configuration
# -----------------------------
st.set_page_config(page_title="Right iTech Student Dashboard", layout="wide")

# App Title (clean, no tagline)
st.markdown("<h1 style='text-align: center;'>📊 Right iTech Student Dashboard</h1>", unsafe_allow_html=True)

# -----------------------------
# File Upload Section
# -----------------------------
with st.sidebar:
    st.header("Upload Data")
    attendance_file = st.file_uploader("Upload Attendance CSV", type=["csv"], key="attendance")
    marks_file = st.file_uploader("Upload Marks CSV", type=["csv"], key="marks")

if attendance_file and marks_file:
    attendance_df = pd.read_csv(attendance_file)
    marks_df = pd.read_csv(marks_file)

    # Clean Attendance Data
    if "Date" in attendance_df.columns:
        attendance_df["Date"] = pd.to_datetime(attendance_df["Date"], errors="coerce")

    if "Gender" not in attendance_df.columns:
        attendance_df["Gender"] = "Unknown"

    # -----------------------------
    # Generate subject color map dynamically
    # -----------------------------
    if "Subject" in marks_df.columns:
        unique_subjects = marks_df["Subject"].unique()
        color_palette = px.colors.qualitative.Set3 + px.colors.qualitative.Bold + px.colors.qualitative.Pastel
        subject_colors = {subj: color_palette[i % len(color_palette)] for i, subj in enumerate(unique_subjects)}
    else:
        subject_colors = {}

    # -----------------------------
    # Sidebar Filters
    # -----------------------------
    st.sidebar.header("Filters")
    threshold = st.sidebar.slider("Set Marks Threshold", 0, 100, 40)
    selected_exam = st.sidebar.selectbox("Select Exam", marks_df["Exam"].unique()) if "Exam" in marks_df.columns else None
    selected_student = st.sidebar.selectbox("Select Student", marks_df["Name"].unique()) if "Name" in marks_df.columns else None
    include_weekends = st.sidebar.radio("Attendance Analysis:", ["Include Weekends", "Exclude Weekends"])

    # -----------------------------
    # Tabs
    # -----------------------------
    tab1, tab2, tab3, tab4 = st.tabs(["📌 Class Overview", "📅 Attendance", "🧑‍🎓 Student Profile", "⚖️ Comparison"])

    # =============================
    # CLASS OVERVIEW
    # =============================
    with tab1:
        st.subheader("📌 Class Overview")

        total_students = attendance_df["Name"].nunique()
        total_boys = attendance_df[attendance_df["Gender"].str.lower() == "male"]["Name"].nunique()
        total_girls = attendance_df[attendance_df["Gender"].str.lower() == "female"]["Name"].nunique()

        avg_attendance = (attendance_df["Status"].eq("Present").mean()) * 100
        avg_absent = 100 - avg_attendance

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Students", total_students)
        col2.metric("Boys", total_boys)
        col3.metric("Girls", total_girls)
        col4.metric("Avg Attendance %", f"{avg_attendance:.1f}%")

        # Threshold-based subject performance
        if "Marks" in marks_df.columns:
            threshold_df = marks_df.groupby("Subject").apply(
                lambda x: (x["Marks"] < threshold).sum()
            ).reset_index(name="Below Threshold")
            fig_threshold = px.bar(
                threshold_df,
                x="Subject",
                y="Below Threshold",
                color="Subject",
                color_discrete_map=subject_colors,
                title=f"Number of Students Below Threshold ({threshold} Marks)"
            )
            st.plotly_chart(fig_threshold, use_container_width=True)

    # =============================
    # ATTENDANCE
    # =============================
    with tab2:
        st.subheader("📅 Attendance Insights")

        # Student-level attendance %
        student_attendance = attendance_df.groupby("Name")["Status"].apply(
            lambda x: (x.eq("Present").mean()) * 100
        ).reset_index(name="Attendance %")
        fig_student_att = px.bar(
            student_attendance,
            x="Name", y="Attendance %",
            color="Attendance %",
            title="Student-wise Attendance Percentage",
            color_continuous_scale="Blues"
        )
        st.plotly_chart(fig_student_att, use_container_width=True)

        # Daily attendance trend
        daily_trend = attendance_df.copy()
        daily_trend["Day"] = daily_trend["Date"].dt.day_name()

        if include_weekends == "Exclude Weekends":
            daily_trend = daily_trend[daily_trend["Day"] != "Friday"]

        daily_trend = daily_trend.groupby("Date")["Status"].apply(
            lambda x: (x.eq("Present").mean()) * 100
        ).reset_index(name="Class Attendance %")

        fig_trend = px.line(
            daily_trend, x="Date", y="Class Attendance %",
            markers=True, title=f"Daily Class Attendance Trend ({include_weekends})"
        )
        st.plotly_chart(fig_trend, use_container_width=True)

        # Attendance distribution histogram
        fig_hist = px.histogram(
            student_attendance,
            x="Attendance %", nbins=20,
            title="Distribution of Attendance % Across Students",
            color_discrete_sequence=["#636EFA"]
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    # =============================
    # STUDENT PROFILE
    # =============================
    with tab3:
        st.subheader("🧑‍🎓 Student Profile")

        if selected_student:
            student_marks = marks_df[marks_df["Name"] == selected_student]
            student_attendance = attendance_df[attendance_df["Name"] == selected_student]

            st.write(f"### {selected_student}")

            # Marks trend by exam
            if not student_marks.empty:
                fig_marks = px.line(
                    student_marks, x="Exam", y="Marks", color="Subject",
                    markers=True, title="Marks Trend Across Exams",
                    color_discrete_map=subject_colors
                )
                st.plotly_chart(fig_marks, use_container_width=True)

            # Attendance summary
            if not student_attendance.empty:
                att_rate = (student_attendance["Status"].eq("Present").mean()) * 100
                st.metric("Attendance %", f"{att_rate:.1f}%")

    # =============================
    # COMPARISON
    # =============================
    with tab4:
        st.subheader("⚖️ Comparison Between Students")

        if "Marks" in marks_df.columns:
            # Compare subject averages per exam
            subj_avg = marks_df.groupby(["Exam", "Subject"])["Marks"].mean().reset_index()
            fig_comp = px.bar(
                subj_avg[subj_avg["Exam"] == selected_exam],
                x="Subject", y="Marks",
                color="Subject", barmode="group",
                color_discrete_map=subject_colors,
                title=f"Subject Averages for {selected_exam}"
            )
            st.plotly_chart(fig_comp, use_container_width=True)

            # Student vs class average
            if selected_student:
                stu_perf = marks_df[marks_df["Name"] == selected_student].groupby("Subject")["Marks"].mean().reset_index()
                class_perf = marks_df.groupby("Subject")["Marks"].mean().reset_index()
                merged = pd.merge(stu_perf, class_perf, on="Subject", suffixes=("_Student", "_Class"))

                fig_compare = go.Figure()
                fig_compare.add_trace(go.Bar(x=merged["Subject"], y=merged["Marks_Student"], name=selected_student,
                                             marker_color="#00CC96"))
                fig_compare.add_trace(go.Bar(x=merged["Subject"], y=merged["Marks_Class"], name="Class Average",
                                             marker_color="#636EFA"))
                fig_compare.update_layout(barmode="group", title=f"{selected_student} vs Class Average")
                st.plotly_chart(fig_compare, use_container_width=True)

else:
    st.warning("⬅️ Please upload both Attendance and Marks CSV files to begin.")
