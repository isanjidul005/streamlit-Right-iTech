import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# =======================================
# App Configuration
# =======================================
st.set_page_config(page_title="Class Performance Dashboard", layout="wide")

# App Title (tagline removed as requested)
st.title("📊 Class Performance Dashboard")

# =======================================
# File Upload Section
# =======================================
st.sidebar.header("Upload Your Data")

attendance_file = st.sidebar.file_uploader("Upload Attendance CSV", type=["csv"])
marks_file = st.sidebar.file_uploader("Upload Marks CSV", type=["csv"])

# =======================================
# Load Data
# =======================================
if attendance_file and marks_file:
    attendance_df = pd.read_csv(attendance_file)
    marks_df = pd.read_csv(marks_file)

    # Convert date column
    if "Date" in attendance_df.columns:
        attendance_df["Date"] = pd.to_datetime(attendance_df["Date"], errors="coerce")

    # Dynamic subject color mapping after dataset is read
    subject_list = [col for col in marks_df.columns if col not in ["Student ID", "Name", "Exam"]]
    color_map = px.colors.qualitative.Set2
    subject_colors = {sub: color_map[i % len(color_map)] for i, sub in enumerate(subject_list)}

else:
    st.warning("Please upload both Attendance and Marks datasets to proceed.")
    st.stop()

# =======================================
# Sidebar Controls
# =======================================
st.sidebar.header("Filters & Options")

# Toggle for weekends
include_weekends = st.sidebar.radio(
    "Attendance Data:", ["Include Fridays", "Exclude Fridays"], index=0
)

# Toggles for extra visualizations
show_attendance_extras = st.sidebar.checkbox("Show Extra Attendance Visualizations", value=True)
show_comparison_extras = st.sidebar.checkbox("Show Extra Comparison Visualizations", value=True)

# =======================================
# Tabs
# =======================================
tabs = st.tabs(["📌 Overview", "🗓️ Attendance", "📝 Marks", "📊 Comparison"])

# =======================================
# Overview Tab
# =======================================
with tabs[0]:
    st.subheader("Class Overview")

    total_students = attendance_df["Student ID"].nunique()
    total_boys = attendance_df[attendance_df["Gender"].str.lower() == "male"]["Student ID"].nunique()
    total_girls = attendance_df[attendance_df["Gender"].str.lower() == "female"]["Student ID"].nunique()

    avg_present = attendance_df.groupby("Date")["Status"].apply(lambda x: (x == "Present").mean()).mean() * 100
    avg_absent = 100 - avg_present

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("👨‍🎓 Total Students", total_students)
    col2.metric("👦 Boys", total_boys)
    col3.metric("👧 Girls", total_girls)
    col4.metric("📊 Avg Attendance", f"{avg_present:.1f}%")

    # Threshold Chart (students above chosen %)
    st.subheader("Performance by Threshold")
    threshold = st.slider("Select minimum average mark (%)", 0, 100, 40)
    subject_avgs = marks_df.groupby("Name")[subject_list].mean()
    above_threshold = (subject_avgs >= threshold).sum()
    below_threshold = (subject_avgs < threshold).sum()

    thresh_df = pd.DataFrame({
        "Subject": subject_list,
        "Above Threshold": [above_threshold[sub] for sub in subject_list],
        "Below Threshold": [below_threshold[sub] for sub in subject_list]
    })

    fig_thresh = go.Figure()
    fig_thresh.add_bar(x=thresh_df["Subject"], y=thresh_df["Above Threshold"], name="Above", marker_color="green")
    fig_thresh.add_bar(x=thresh_df["Subject"], y=thresh_df["Below Threshold"], name="Below", marker_color="red")
    fig_thresh.update_layout(barmode="stack", xaxis_title="Subject", yaxis_title="Number of Students")
    st.plotly_chart(fig_thresh, use_container_width=True)

# =======================================
# Attendance Tab
# =======================================
with tabs[1]:
    st.subheader("Attendance Insights")

    att_df = attendance_df.copy()
    if include_weekends == "Exclude Fridays":
        att_df = att_df[att_df["Date"].dt.day_name() != "Friday"]

    # Daily Attendance Trend
    daily_att = att_df.groupby("Date")["Status"].apply(lambda x: (x == "Present").mean() * 100).reset_index()
    fig_att = px.line(daily_att, x="Date", y="Status", title="Daily Attendance (%)", markers=True)
    st.plotly_chart(fig_att, use_container_width=True)

    if show_attendance_extras:
        col1, col2 = st.columns(2)

        # Attendance Distribution by Student
        student_att = att_df.groupby("Name")["Status"].apply(lambda x: (x == "Present").mean() * 100).reset_index()
        fig_hist = px.histogram(student_att, x="Status", nbins=20, title="Distribution of Attendance %")
        col1.plotly_chart(fig_hist, use_container_width=True)

        # Average Attendance by Weekday
        weekday_att = att_df.groupby(att_df["Date"].dt.day_name())["Status"].apply(lambda x: (x == "Present").mean() * 100)
        weekday_att = weekday_att.reindex(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"])
        fig_week = px.bar(weekday_att, x=weekday_att.index, y=weekday_att.values, title="Avg Attendance by Weekday")
        col2.plotly_chart(fig_week, use_container_width=True)

# =======================================
# Marks Tab
# =======================================
with tabs[2]:
    st.subheader("Marks Insights")

    # Subject averages across all students
    subj_means = marks_df[subject_list].mean().reset_index()
    subj_means.columns = ["Subject", "Average"]

    fig_marks = px.bar(subj_means, x="Subject", y="Average", color="Subject",
                       color_discrete_map=subject_colors, title="Average Marks by Subject")
    st.plotly_chart(fig_marks, use_container_width=True)

# =======================================
# Comparison Tab
# =======================================
with tabs[3]:
    st.subheader("Comparisons")

    # Compare Exams by subject
    exam_subj = marks_df.groupby("Exam")[subject_list].mean().reset_index()
    fig_exam = go.Figure()
    for subj in subject_list:
        fig_exam.add_trace(go.Scatter(x=exam_subj["Exam"], y=exam_subj[subj],
                                      mode="lines+markers", name=subj,
                                      line=dict(color=subject_colors[subj])))
    fig_exam.update_layout(title="Subject Performance Across Exams", xaxis_title="Exam", yaxis_title="Average Marks")
    st.plotly_chart(fig_exam, use_container_width=True)

    if show_comparison_extras:
        col1, col2 = st.columns(2)

        # Radar Chart for subject averages
        avg_scores = marks_df[subject_list].mean()
        polar_df = pd.DataFrame({"Subject": subject_list, "Average": avg_scores})
        polar_df = pd.concat([polar_df, polar_df.iloc[[0]]])  # close loop

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(r=polar_df["Average"], theta=polar_df["Subject"],
                                            fill="toself", name="Class Average"))
        fig_radar.update_layout(title="Overall Subject Performance (Radar)")
        col1.plotly_chart(fig_radar, use_container_width=True)

        # Variance of marks by subject
        subj_var = marks_df[subject_list].var().reset_index()
        subj_var.columns = ["Subject", "Variance"]
        fig_var = px.bar(subj_var, x="Subject", y="Variance", color="Subject",
                         color_discrete_map=subject_colors, title="Marks Variance by Subject")
        col2.plotly_chart(fig_var, use_container_width=True)
