import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# =========================
# App Title
# =========================
st.set_page_config(page_title="Right iTech", layout="wide")
st.title("Right iTech 📊")

# =========================
# Sidebar File Upload
# =========================
st.sidebar.header("Upload your data")
att_file = st.sidebar.file_uploader("Upload Attendance CSV", type=["csv"])
marks_file = st.sidebar.file_uploader("Upload Marks CSV", type=["csv"])

if att_file is not None and marks_file is not None:
    attendance_df = pd.read_csv(att_file)
    marks_df = pd.read_csv(marks_file)

    # Ensure correct datatypes
    if "Date" in attendance_df.columns:
        attendance_df["Date"] = pd.to_datetime(attendance_df["Date"], errors="coerce")

    # =========================
    # Assign subject colors dynamically
    # =========================
    if "Subject" in marks_df.columns:
        subjects = marks_df["Subject"].unique()
        color_palette = px.colors.qualitative.Set2
        subject_colors = {subj: color_palette[i % len(color_palette)] for i, subj in enumerate(subjects)}
    else:
        subject_colors = {}

    # =========================
    # Tabs
    # =========================
    tabs = st.tabs(["📌 Class Overview", "🗓 Attendance", "🎓 Student Profiles", "📊 Comparison"])

    # =========================
    # CLASS OVERVIEW TAB
    # =========================
    with tabs[0]:
        st.subheader("Class Overview")

        # Basic stats
        total_students = attendance_df["Name"].nunique() if "Name" in attendance_df.columns else 0
        avg_attendance = (attendance_df["Status"].eq("Present").mean() * 100) if "Status" in attendance_df.columns else 0
        avg_absent = 100 - avg_attendance

        col1, col2, col3 = st.columns(3)
        col1.metric("👨‍🎓 Total Students", total_students)
        col2.metric("✅ Avg Attendance", f"{avg_attendance:.1f}%")
        col3.metric("❌ Avg Absence", f"{avg_absent:.1f}%")

        # Threshold-based bar chart
        st.markdown("### Performance by Subject Thresholds")
        threshold = st.slider("Select minimum marks threshold", 0, 100, 40)
        if "Marks" in marks_df.columns and "Subject" in marks_df.columns:
            perf = marks_df.groupby("Subject").apply(lambda x: (x["Marks"] >= threshold).mean() * 100).reset_index(name="Percentage")
            fig = px.bar(perf, x="Subject", y="Percentage", text="Percentage",
                         color="Subject", color_discrete_map=subject_colors)
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            st.plotly_chart(fig, use_container_width=True)

    # =========================
    # ATTENDANCE TAB
    # =========================
    with tabs[1]:
        st.subheader("Attendance Insights")

        # Toggle include/exclude Fridays
        include_friday = st.checkbox("Include Fridays (weekend)", value=False)
        att_plot = attendance_df.copy()
        if not include_friday and "Date" in att_plot.columns:
            att_plot = att_plot[att_plot["Date"].dt.day_name() != "Friday"]

        # Checkbox toggles for visualizations
        st.markdown("#### Select Attendance Visualizations")
        show_dots = st.checkbox("Timeline Dots", value=True)
        show_stack = st.checkbox("Stacked Bar Chart", value=True)
        show_heatmap = st.checkbox("Attendance Heatmap", value=True)

        if "Date" in att_plot.columns and "Name" in att_plot.columns and "Status" in att_plot.columns:
            # Timeline dots
            if show_dots:
                dot_chart = px.scatter(
                    att_plot,
                    x="Date",
                    y="Name",
                    color="Status",
                    title="Attendance Timeline",
                    color_discrete_map={"Present": "green", "Absent": "red"}
                )
                st.plotly_chart(dot_chart, use_container_width=True)

            # Stacked bar chart
            if show_stack:
                daily_att = att_plot.groupby(["Date", "Status"]).size().reset_index(name="Count")
                stack_chart = px.bar(
                    daily_att,
                    x="Date",
                    y="Count",
                    color="Status",
                    title="Daily Attendance Split",
                    color_discrete_map={"Present": "green", "Absent": "red"}
                )
                st.plotly_chart(stack_chart, use_container_width=True)

            # Heatmap
            if show_heatmap:
                pivot_att = att_plot.pivot_table(index="Name", columns="Date", values="Status", aggfunc=lambda x: (x=="Present").mean())
                heatmap = go.Figure(data=go.Heatmap(
                    z=pivot_att.values,
                    x=pivot_att.columns.strftime("%Y-%m-%d"),
                    y=pivot_att.index,
                    colorscale="RdYlGn"
                ))
                heatmap.update_layout(title="Attendance Heatmap", xaxis_title="Date", yaxis_title="Student")
                st.plotly_chart(heatmap, use_container_width=True)

    # =========================
    # STUDENT PROFILES TAB
    # =========================
    with tabs[2]:
        st.subheader("Individual Student Profiles")
        students = marks_df["Name"].unique() if "Name" in marks_df.columns else []
        selected_student = st.selectbox("Select a student", students)

        if selected_student:
            st.markdown(f"### 📌 Profile: {selected_student}")

            # Student marks trend
            stud_marks = marks_df[marks_df["Name"] == selected_student]
            if not stud_marks.empty:
                fig = px.line(stud_marks, x="Exam", y="Marks", color="Subject",
                              markers=True, title="Marks Trend",
                              color_discrete_map=subject_colors)
                st.plotly_chart(fig, use_container_width=True)

            # Attendance timeline
            stud_att = attendance_df[attendance_df["Name"] == selected_student]
            if not stud_att.empty:
                att_chart = px.scatter(stud_att, x="Date", y="Status", color="Status",
                                       title="Attendance Timeline",
                                       color_discrete_map={"Present": "green", "Absent": "red"})
                st.plotly_chart(att_chart, use_container_width=True)

    # =========================
    # COMPARISON TAB
    # =========================
    with tabs[3]:
        st.subheader("Student Comparison")

        students_compare = st.multiselect("Select students to compare", marks_df["Name"].unique())
        if students_compare:
            compare_df = marks_df[marks_df["Name"].isin(students_compare)]

            # Marks comparison
            fig = px.bar(compare_df, x="Subject", y="Marks", color="Name",
                         barmode="group", title="Marks Comparison by Subject")
            st.plotly_chart(fig, use_container_width=True)

            # Average by exam
            avg_exam = compare_df.groupby(["Exam", "Name"])["Marks"].mean().reset_index()
            line_chart = px.line(avg_exam, x="Exam", y="Marks", color="Name",
                                 markers=True, title="Exam-wise Average Marks")
            st.plotly_chart(line_chart, use_container_width=True)

else:
    st.info("👆 Please upload both Attendance and Marks CSV files to continue.")
