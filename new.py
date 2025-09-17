import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ----------------------
# CONFIG
# ----------------------
st.set_page_config(page_title="Super Dashboard", layout="wide")
st.title("🌍 Super Duper Student Dashboard")

# ----------------------
# UPLOAD FILES
# ----------------------
st.sidebar.header("Upload Data")
att_file = st.sidebar.file_uploader("Upload Attendance CSV", type=["csv"])
marks_file = st.sidebar.file_uploader("Upload Marks CSV", type=["csv"])

if att_file and marks_file:
    attendance = pd.read_csv(att_file)
    marks = pd.read_csv(marks_file)

    # ----------------------
    # CLEAN ATTENDANCE
    # ----------------------
    def normalize_status(x):
        if str(x).strip().lower() in ["present", "p", "1", "yes"]:
            return 1
        return 0

    attendance["AttValue"] = attendance["Status"].apply(normalize_status)

    att_summary = (
        attendance.groupby(["ID", "Name"])["AttValue"]
        .mean()
        .reset_index(name="AttendanceRate")
    )
    att_summary["AttendanceRate"] *= 100

    # ----------------------
    # CLEAN MARKS
    # ----------------------
    if "WasAbsent" not in marks.columns:
        marks["WasAbsent"] = marks["Marks"].isna()  # fallback

    marks_summary = (
        marks.groupby(["ID", "Name"])
        .apply(lambda df: df.loc[~df["WasAbsent"], "Marks"].mean())
        .reset_index(name="AvgScore")
    )

    # Align merge keys
    for df in [att_summary, marks_summary]:
        df["ID"] = df["ID"].astype(str).str.strip()
        df["Name"] = df["Name"].astype(str).str.strip()

    students = pd.merge(att_summary, marks_summary, on=["ID", "Name"], how="outer")

    # ----------------------
    # TABS
    # ----------------------
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📌 Overview",
        "👤 Student Profile",
        "⚖️ Comparison",
        "📈 Correlation",
        "📚 Subjects",
        "📊 Trends",
        "🎯 Clustering"
    ])

    # ----------------------
    # OVERVIEW
    # ----------------------
    with tab1:
        st.header("Class Overview")

        col1, col2 = st.columns(2)
        with col1:
            fig1 = px.histogram(
                students, x="AttendanceRate", nbins=20,
                color_discrete_sequence=px.colors.sequential.Teal,
                title="Attendance Distribution"
            )
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            fig2 = px.histogram(
                students, x="AvgScore", nbins=20,
                color_discrete_sequence=px.colors.sequential.OrRd,
                title="Score Distribution"
            )
            st.plotly_chart(fig2, use_container_width=True)

        fig_box = px.box(
            students, y="AvgScore", points="all",
            color_discrete_sequence=["#636EFA"],
            title="Spread of Average Scores"
        )
        st.plotly_chart(fig_box, use_container_width=True)

        with st.expander("ℹ️ Explanation"):
            st.write("""
            - Distribution shows where most students fall.
            - Boxplot highlights spread, outliers, and concentration of scores.
            """)

    # ----------------------
    # STUDENT PROFILE
    # ----------------------
    with tab2:
        st.header("Student Profile")
        student_name = st.selectbox("Choose Student", students["Name"].dropna().unique())

        profile = students[students["Name"] == student_name].iloc[0]
        st.metric("Attendance Rate", f"{profile['AttendanceRate']:.1f}%")
        st.metric("Average Score", f"{profile['AvgScore']:.1f}")

        subj = marks[marks["Name"].str.strip() == student_name]
        subj_avg = subj.groupby("Subject")["Marks"].mean().reset_index()

        # Radar chart
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=subj_avg["Marks"], theta=subj_avg["Subject"],
            fill='toself', name=student_name
        ))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=False)
        st.plotly_chart(fig_radar, use_container_width=True)

        with st.expander("ℹ️ Explanation"):
            st.write("""
            Radar chart shows subject-wise strengths and weaknesses of the student.
            """)

    # ----------------------
    # COMPARISON
    # ----------------------
    with tab3:
        st.header("Compare Students")
        s1 = st.selectbox("Select Student 1", students["Name"].dropna().unique(), key="s1")
        s2 = st.selectbox("Select Student 2", students["Name"].dropna().unique(), key="s2")

        comp = students[students["Name"].isin([s1, s2])]
        fig = px.bar(comp, x="Name", y=["AttendanceRate", "AvgScore"], barmode="group",
                     color_discrete_sequence=px.colors.qualitative.Set2,
                     title="Comparison of Attendance & Scores")
        st.plotly_chart(fig, use_container_width=True)

    # ----------------------
    # CORRELATION
    # ----------------------
    with tab4:
        st.header("Attendance vs Performance Correlation")

        fig = px.scatter(
            students, x="AttendanceRate", y="AvgScore", color="AvgScore",
            color_continuous_scale="Plasma", trendline="ols", hover_data=["Name"]
        )
        st.plotly_chart(fig, use_container_width=True)

    # ----------------------
    # SUBJECT ANALYSIS
    # ----------------------
    with tab5:
        st.header("Subject Analysis")
        subj_avg = marks.loc[~marks["WasAbsent"]].groupby("Subject")["Marks"].mean().reset_index()
        fig = px.bar(subj_avg, x="Subject", y="Marks", color="Subject",
                     title="Average Score by Subject")
        st.plotly_chart(fig, use_container_width=True)

        # Attendance by subject
        subj_att = attendance.groupby("Subject")["AttValue"].mean().reset_index()
        subj_att["AttValue"] *= 100
        fig2 = px.bar(subj_att, x="Subject", y="AttValue", color="Subject",
                      title="Attendance by Subject")
        st.plotly_chart(fig2, use_container_width=True)

    # ----------------------
    # TRENDS
    # ----------------------
    with tab6:
        st.header("Trends Over Time")
        if "Exam" in marks.columns:
            exam_avg = marks.groupby("Exam")["Marks"].mean().reset_index()
            fig = px.line(exam_avg, x="Exam", y="Marks", markers=True,
                          title="Class Average Across Exams")
            st.plotly_chart(fig, use_container_width=True)

    # ----------------------
    # CLUSTERING
    # ----------------------
    with tab7:
        st.header("Student Clustering")
        df_cluster = students.dropna()
        if not df_cluster.empty:
            X = df_cluster[["AttendanceRate", "AvgScore"]]
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            kmeans = KMeans(n_clusters=3, random_state=42)
            df_cluster["Cluster"] = kmeans.fit_predict(X_scaled)

            fig = px.scatter(
                df_cluster, x="AttendanceRate", y="AvgScore",
                color="Cluster", hover_data=["Name"],
                title="Clusters of Students",
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            st.plotly_chart(fig, use_container_width=True)
