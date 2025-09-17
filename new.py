import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="📊 Class 3 Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------
# FILE UPLOADS
# -----------------
st.sidebar.header("📂 Upload Your Data")
attendance_file = st.sidebar.file_uploader("Upload Attendance CSV", type="csv")
marks_file = st.sidebar.file_uploader("Upload Marks CSV", type="csv")

if attendance_file and marks_file:
    attendance = pd.read_csv(attendance_file)
    marks = pd.read_csv(marks_file)

    # Ensure ID consistency
    attendance["ID"] = attendance["ID"].astype(str)
    marks["ID"] = marks["ID"].astype(str)

    # -----------------
    # PREP DATA
    # -----------------
    attendance_summary = (
        attendance.groupby(["ID", "Name"])
        .agg(AttendanceRate=("Status", lambda x: (x == "Present").mean() * 100))
        .reset_index()
    )

    marks_summary = (
        marks.groupby(["ID", "Name", "Subject"])
        .agg(AverageScore=("Marks", "mean"))
        .reset_index()
    )

    overall_scores = (
        marks.groupby(["ID", "Name"])
        .agg(AverageScore=("Marks", "mean"))
        .reset_index()
    )

    merged = pd.merge(attendance_summary, overall_scores, on=["ID", "Name"], how="left")

    # -----------------
    # TABS
    # -----------------
    tabs = st.tabs([
        "📊 Class Overview", 
        "👩‍🎓 Student Profile", 
        "⚖️ Student Comparison", 
        "🔍 Attendance vs Performance", 
        "📈 Trends Over Time", 
        "📚 Subject Analysis", 
        "🤖 Clustering Insights"
    ])

    # -----------------
    # TAB 1: Class Overview
    # -----------------
    with tabs[0]:
        st.header("📊 Class Overview")
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Average Attendance Rate", f"{attendance_summary['AttendanceRate'].mean():.1f}%")
        with col2:
            st.metric("Average Score", f"{overall_scores['AverageScore'].mean():.1f}")

        fig_attendance = px.histogram(
            attendance_summary, 
            x="AttendanceRate", 
            nbins=20, 
            title="Distribution of Attendance Rates"
        )
        st.plotly_chart(fig_attendance, use_container_width=True)

        fig_scores = px.histogram(
            overall_scores, 
            x="AverageScore", 
            nbins=20, 
            title="Distribution of Average Scores"
        )
        st.plotly_chart(fig_scores, use_container_width=True)

    # -----------------
    # TAB 2: Student Profile
    # -----------------
    with tabs[1]:
        st.header("👩‍🎓 Student Profile")
        student = st.selectbox("Select a student:", overall_scores["Name"].unique())

        profile_att = attendance_summary[attendance_summary["Name"] == student]
        profile_marks = marks_summary[marks_summary["Name"] == student]

        col1, col2 = st.columns(2)
        with col1:
            st.write("**Attendance Rate:**", f"{profile_att['AttendanceRate'].values[0]:.1f}%")
        with col2:
            st.write("**Average Score:**", f"{profile_marks['AverageScore'].mean():.1f}")

        # Radar chart for subjects
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=profile_marks["AverageScore"],
            theta=profile_marks["Subject"],
            fill="toself",
            name=student
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=False,
            title=f"Performance by Subject: {student}"
        )
        st.plotly_chart(fig, use_container_width=True)

    # -----------------
    # TAB 3: Student Comparison
    # -----------------
    with tabs[2]:
        st.header("⚖️ Student Comparison")
        students = st.multiselect("Select students to compare:", overall_scores["Name"].unique())

        if len(students) >= 2:
            comp = marks_summary[marks_summary["Name"].isin(students)]
            fig = px.line(
                comp, x="Subject", y="AverageScore", color="Name", markers=True,
                title="Comparison Across Subjects"
            )
            st.plotly_chart(fig, use_container_width=True)

    # -----------------
    # TAB 4: Attendance vs Performance
    # -----------------
    with tabs[3]:
        st.header("🔍 Attendance vs Performance Correlation")
        fig = px.scatter(
            merged, x="AttendanceRate", y="AverageScore", hover_data=["Name"],
            trendline="ols", title="Attendance Rate vs Average Score"
        )
        st.plotly_chart(fig, use_container_width=True)

    # -----------------
    # TAB 5: Trends Over Time
    # -----------------
    with tabs[4]:
        st.header("📈 Trends Over Time")
        student = st.selectbox("Select a student for timeline:", overall_scores["Name"].unique())
        timeline = marks[marks["Name"] == student]

        fig = px.line(
            timeline, x="ExamNumber", y="Marks", color="Subject", markers=True,
            title=f"Performance Over Time: {student}"
        )
        st.plotly_chart(fig, use_container_width=True)

    # -----------------
    # TAB 6: Subject Analysis
    # -----------------
    with tabs[5]:
        st.header("📚 Subject Analysis")
        subject_avg = marks.groupby("Subject")["Marks"].mean().reset_index()
        fig = px.bar(subject_avg, x="Subject", y="Marks", title="Average Scores by Subject")
        st.plotly_chart(fig, use_container_width=True)

    # -----------------
    # TAB 7: Clustering Insights
    # -----------------
    with tabs[6]:
        st.header("🤖 Clustering Insights")

        X = merged[["AttendanceRate", "AverageScore"]].dropna()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        kmeans = KMeans(n_clusters=3, random_state=42)
        merged["Cluster"] = kmeans.fit_predict(X_scaled)

        fig = px.scatter(
            merged, x="AttendanceRate", y="AverageScore", color="Cluster", hover_data=["Name"],
            title="Student Clusters Based on Attendance & Performance"
        )
        st.plotly_chart(fig, use_container_width=True)

else:
    st.warning("⚠️ Please upload both attendance and marks CSV files to start.")
