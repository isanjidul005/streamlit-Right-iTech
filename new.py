import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ----------------------
# DATA LOADING
# ----------------------
st.set_page_config(page_title="Class Dashboard", layout="wide")

st.title("📊 World-Class Student Dashboard")

st.sidebar.header("Upload Data")
att_file = st.sidebar.file_uploader("Upload Attendance CSV", type=["csv"])
marks_file = st.sidebar.file_uploader("Upload Marks CSV", type=["csv"])

if att_file and marks_file:
    attendance = pd.read_csv(att_file)
    marks = pd.read_csv(marks_file)

    # ----------------------
    # DATA CLEANING
    # ----------------------
    # Attendance summary
    att_summary = (
        attendance.groupby(["ID", "Name"])["Status"]
        .apply(lambda x: (x == "Present").mean() * 100)
        .reset_index(name="AttendanceRate")
    )

    # Marks summary (ignore absents)
    marks_summary = (
        marks.groupby(["ID", "Name"])
        .apply(lambda df: df.loc[~df["WasAbsent"], "Marks"].mean())
        .reset_index(name="AvgScore")
    )

    # --- Fix merge key issues ---
    for df in [att_summary, marks_summary]:
        df["ID"] = df["ID"].astype(str).str.strip()
        df["Name"] = df["Name"].astype(str).str.strip()

    students = pd.merge(att_summary, marks_summary, on=["ID", "Name"], how="outer")

    # Track unmatched students
    unmatched_att = att_summary[~att_summary["ID"].isin(students["ID"])]
    unmatched_marks = marks_summary[~marks_summary["ID"].isin(students["ID"])]

    # ----------------------
    # DASHBOARD
    # ----------------------
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📌 Class Overview",
        "👤 Student Profile",
        "⚖️ Compare Students",
        "📈 Correlation",
        "📚 Subject Analysis",
        "🎯 Clustering"
    ])

    # ----------------------
    # CLASS OVERVIEW
    # ----------------------
    with tab1:
        st.header("Class Overview")

        col1, col2 = st.columns(2)
        with col1:
            fig1 = px.histogram(students, x="AttendanceRate", nbins=20, title="Attendance Distribution")
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            fig2 = px.histogram(students, x="AvgScore", nbins=20, title="Score Distribution")
            st.plotly_chart(fig2, use_container_width=True)

        with st.expander("ℹ️ Explanation"):
            st.write("""
            - The left chart shows how student attendance is spread.
            - The right chart shows how student average scores are spread.
            - Both help identify weak and strong groups in the class.
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

        # Subject-wise performance
        subj = marks[marks["Name"].str.strip() == student_name]
        subj_avg = subj.groupby("Subject")["Marks"].mean().reset_index()
        fig = px.bar(subj_avg, x="Subject", y="Marks", title="Subject Performance")
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("ℹ️ Explanation"):
            st.write("""
            - Attendance and average performance of the selected student.
            - Subject performance shows where the student is strong or weak.
            """)

    # ----------------------
    # STUDENT COMPARISON
    # ----------------------
    with tab3:
        st.header("Compare Two Students")
        col1, col2 = st.columns(2)
        with col1:
            s1 = st.selectbox("Select Student 1", students["Name"].dropna().unique(), key="s1")
        with col2:
            s2 = st.selectbox("Select Student 2", students["Name"].dropna().unique(), key="s2")

        comp = students[students["Name"].isin([s1, s2])]
        fig = px.bar(comp, x="Name", y=["AttendanceRate", "AvgScore"], barmode="group",
                     title="Attendance vs Score Comparison")
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("ℹ️ Explanation"):
            st.write("""
            Compare two students side by side in terms of attendance and average marks.
            """)

    # ----------------------
    # CORRELATION
    # ----------------------
    with tab4:
        st.header("Attendance vs Performance Correlation")

        fig = px.scatter(
            students, x="AttendanceRate", y="AvgScore", color="AvgScore",
            trendline="ols", hover_data=["Name"]
        )
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("ℹ️ Explanation"):
            st.write("""
            - Each point is a student.
            - X-axis = attendance percentage.
            - Y-axis = average score (ignores absent exams).
            - Regression line shows if more attendance means better marks.
            """)

    # ----------------------
    # SUBJECT ANALYSIS
    # ----------------------
    with tab5:
        st.header("Subject Analysis")
        subj_avg = marks.loc[~marks["WasAbsent"]].groupby("Subject")["Marks"].mean().reset_index()
        fig = px.bar(subj_avg, x="Subject", y="Marks", title="Average Score by Subject")
        st.plotly_chart(fig, use_container_width=True)

    # ----------------------
    # CLUSTERING
    # ----------------------
    with tab6:
        st.header("Student Segmentation (Clustering)")
        df_cluster = students.dropna()
        X = df_cluster[["AttendanceRate", "AvgScore"]]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        kmeans = KMeans(n_clusters=3, random_state=42)
        df_cluster["Cluster"] = kmeans.fit_predict(X_scaled)

        fig = px.scatter(df_cluster, x="AttendanceRate", y="AvgScore",
                         color="Cluster", hover_data=["Name"],
                         title="Clusters of Students")
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("ℹ️ Explanation"):
            st.write("""
            Students are grouped into clusters:
            - High attendance & high scores
            - Low attendance & low scores
            - Mixed profiles
            """)

    # ----------------------
    # DATA INTEGRITY CHECK
    # ----------------------
    with st.sidebar:
        st.subheader("🔎 Data Integrity")
        if not unmatched_att.empty:
            st.warning(f"{len(unmatched_att)} students in attendance not in marks")
        if not unmatched_marks.empty:
            st.warning(f"{len(unmatched_marks)} students in marks not in attendance")

else:
    st.info("⬅️ Please upload both Attendance and Marks CSV files to start.")
