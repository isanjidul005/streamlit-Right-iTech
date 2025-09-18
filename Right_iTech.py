import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# --------------------------------------------------------
# Utility Functions
# --------------------------------------------------------

# Assign distinct colors dynamically to each subject
def assign_subject_colors(df):
    subjects = df["Subject"].unique().tolist()
    palette = px.colors.qualitative.Set2 + px.colors.qualitative.Pastel
    color_map = {subj: palette[i % len(palette)] for i, subj in enumerate(subjects)}
    return color_map

# Load uploaded CSV safely
def load_csv(uploaded_file):
    try:
        return pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error reading {uploaded_file.name}: {e}")
        return pd.DataFrame()

# --------------------------------------------------------
# App Title
# --------------------------------------------------------
st.markdown("<h1 style='text-align:center;'>Right iTech</h1>", unsafe_allow_html=True)

# --------------------------------------------------------
# Sidebar Upload
# --------------------------------------------------------
st.sidebar.header("Upload Datasets")
attendance_file = st.sidebar.file_uploader("Upload Attendance CSV", type="csv")
marks_file = st.sidebar.file_uploader("Upload Marks CSV", type="csv")

if attendance_file is not None:
    attendance_df = load_csv(attendance_file)
else:
    attendance_df = pd.DataFrame()

if marks_file is not None:
    marks_df = load_csv(marks_file)
else:
    marks_df = pd.DataFrame()

# Exit early if no data
if attendance_df.empty or marks_df.empty:
    st.warning("Please upload both Attendance and Marks CSVs to continue.")
    st.stop()

# --------------------------------------------------------
# Preprocess Data
# --------------------------------------------------------

# Ensure dates are parsed
if "Date" in attendance_df.columns:
    attendance_df["Date"] = pd.to_datetime(attendance_df["Date"], errors="coerce")

# Ensure columns exist
if "Student ID" not in attendance_df.columns or "Status" not in attendance_df.columns:
    st.error("Attendance CSV must include at least: 'Student ID', 'Date', 'Status'")
    st.stop()

if "Student ID" not in marks_df.columns or "Marks" not in marks_df.columns:
    st.error("Marks CSV must include at least: 'Student ID', 'Subject', 'Exam', 'Marks'")
    st.stop()

# Assign subject colors dynamically
subject_colors = assign_subject_colors(marks_df)

# --------------------------------------------------------
# Tabs
# --------------------------------------------------------
tabs = st.tabs(["Class Overview", "Attendance", "Comparison"])

# --------------------------------------------------------
# CLASS OVERVIEW TAB
# --------------------------------------------------------
with tabs[0]:
    st.subheader("Class Overview")

    total_students = attendance_df["Student ID"].nunique()
    total_boys = attendance_df[attendance_df["Gender"] == "Male"]["Student ID"].nunique() if "Gender" in attendance_df.columns else 0
    total_girls = attendance_df[attendance_df["Gender"] == "Female"]["Student ID"].nunique() if "Gender" in attendance_df.columns else 0

    avg_attendance = (attendance_df["Status"] == "Present").mean() * 100
    avg_absent = 100 - avg_attendance

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Students", total_students)
    col2.metric("Boys", total_boys)
    col3.metric("Girls", total_girls)
    col4.metric("Avg Attendance %", f"{avg_attendance:.1f}%")

    # Subject averages (bar chart with threshold filter)
    threshold = st.slider("Select minimum average marks threshold", 0, 100, 40)
    subj_avg = marks_df.groupby("Subject")["Marks"].mean().reset_index()
    subj_avg = subj_avg[subj_avg["Marks"] >= threshold]

    fig = px.bar(
        subj_avg,
        x="Subject",
        y="Marks",
        title=f"Subjects with Avg ≥ {threshold}",
        color="Subject",
        color_discrete_map=subject_colors,
    )
    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------
# ATTENDANCE TAB
# --------------------------------------------------------
with tabs[1]:
    st.subheader("Attendance Insights")

    # Toggle weekends
    include_weekends = st.toggle("Include Fridays in Attendance", value=False)

    att_df = attendance_df.copy()
    if not include_weekends and "Date" in att_df.columns:
        att_df = att_df[att_df["Date"].dt.day_name() != "Friday"]

    # Daily attendance % line
    if not att_df.empty:
        daily = (
            att_df.groupby("Date")["Status"]
            .apply(lambda x: (x == "Present").mean() * 100)
            .reset_index(name="Attendance %")
        )
        fig = px.line(daily, x="Date", y="Attendance %", title="Daily Attendance %")
        st.plotly_chart(fig, use_container_width=True)

    # Weekly average
    if not att_df.empty:
        att_df["Week"] = att_df["Date"].dt.to_period("W").astype(str)
        weekly = (
            att_df.groupby("Week")["Status"]
            .apply(lambda x: (x == "Present").mean() * 100)
            .reset_index(name="Attendance %")
        )
        fig = px.bar(weekly, x="Week", y="Attendance %", title="Weekly Average Attendance %")
        st.plotly_chart(fig, use_container_width=True)

    # Attendance distribution
    if not att_df.empty:
        student_att = (
            att_df.groupby("Student ID")["Status"]
            .apply(lambda x: (x == "Present").mean() * 100)
            .reset_index(name="Attendance %")
        )
        fig = px.histogram(
            student_att,
            x="Attendance %",
            nbins=10,
            title="Distribution of Student Attendance %"
        )
        st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------
# COMPARISON TAB
# --------------------------------------------------------
with tabs[2]:
    st.subheader("Comparison")

    # Subject averages radar chart
    subj_avg = marks_df.groupby("Subject")["Marks"].mean().reset_index()
    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=subj_avg["Marks"],
            theta=subj_avg["Subject"],
            fill="toself",
            name="Average Marks",
            line=dict(color="blue"),
        )
    )
    fig.update_layout(polar=dict(radialaxis=dict(visible=True)), title="Average Marks per Subject")
    st.plotly_chart(fig, use_container_width=True)

    # Box plot of marks per subject
    fig = px.box(
        marks_df,
        x="Subject",
        y="Marks",
        color="Subject",
        title="Marks Distribution per Subject",
        color_discrete_map=subject_colors,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Compare across exams
    if "Exam" in marks_df.columns:
        exam_avg = marks_df.groupby(["Exam", "Subject"])["Marks"].mean().reset_index()
        fig = px.line(
            exam_avg,
            x="Subject",
            y="Marks",
            color="Exam",
            markers=True,
            title="Subject-wise Averages Across Exams",
        )
        st.plotly_chart(fig, use_container_width=True)
