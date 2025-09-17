# new.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Class Dashboard", layout="wide")

# ------------------------
# FILE UPLOAD SECTION
# ------------------------
st.sidebar.header("Upload Data Files")
att_file = st.sidebar.file_uploader("Upload Attendance CSV", type="csv")
marks_file = st.sidebar.file_uploader("Upload Marks CSV", type="csv")

if att_file and marks_file:
    attendance = pd.read_csv(att_file)
    marks = pd.read_csv(marks_file)

    # ------------------------
    # DATA PREP
    # ------------------------
    att_summary = (
        attendance.groupby(["ID", "Name"])["Status"]
        .apply(lambda x: (x == "Present").mean() * 100)
        .reset_index(name="AttendanceRate")
    )

    marks_summary = (
        marks.groupby(["ID", "Name"])
        .apply(lambda df: df.loc[~df["WasAbsent"], "Marks"].mean())
        .reset_index(name="AvgScore")
    )

    students = pd.merge(att_summary, marks_summary, on=["ID", "Name"], how="outer")

    # ------------------------
    # TABS
    # ------------------------
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 Class Overview", "👤 Student Profile", "⚖️ Student Comparison", "🔍 Trends & Clusters"]
    )

    # ========================
    # TAB 1: CLASS OVERVIEW
    # ========================
    with tab1:
        st.subheader("Class Overview")
        st.write("Overall class performance and attendance insights.")

        # Scatter: Attendance vs Performance
        fig_scatter = px.scatter(
            students,
            x="AttendanceRate",
            y="AvgScore",
            text="Name",
            trendline="ols",
            labels={"AttendanceRate": "Attendance (%)", "AvgScore": "Average Score"},
            title="Attendance vs Performance",
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        if st.checkbox("Show Explanation (Attendance vs Performance)"):
            st.info(
                "This scatter plot shows how attendance rate relates to performance. "
                "The regression line indicates whether consistent attendance improves scores. "
                "Outliers highlight students who may need extra support."
            )

        # Subject Average
        subject_avg = (
            marks.loc[~marks["WasAbsent"]].groupby("Subject")["Marks"].mean().reset_index()
        )
        fig_subject = px.bar(
            subject_avg,
            x="Subject",
            y="Marks",
            title="Average Marks by Subject",
            color="Marks",
            text_auto=True,
        )
        st.plotly_chart(fig_subject, use_container_width=True)

    # ========================
    # TAB 2: STUDENT PROFILE
    # ========================
    with tab2:
        st.subheader("Student Profile")
        student = st.selectbox("Select Student", students["Name"].unique())

        stu_att = att_summary.loc[att_summary["Name"] == student, "AttendanceRate"].values[0]
        stu_marks = marks.loc[(marks["Name"] == student) & (~marks["WasAbsent"])]

        col1, col2 = st.columns(2)
        col1.metric("Attendance Rate", f"{stu_att:.1f}%")
        col2.metric("Average Score", f"{stu_marks['Marks'].mean():.1f}")

        # Radar chart
        subj_perf = stu_marks.groupby("Subject")["Marks"].mean().reset_index()
        fig_radar = go.Figure()
        fig_radar.add_trace(
            go.Scatterpolar(
                r=subj_perf["Marks"],
                theta=subj_perf["Subject"],
                fill="toself",
                name=student,
            )
        )
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            title="Subject-wise Performance",
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        if st.checkbox("Show Explanation (Radar Chart)"):
            st.info(
                "The radar chart shows subject strengths and weaknesses. "
                "The further out from the center, the better the performance."
            )

    # ========================
    # TAB 3: STUDENT COMPARISON
    # ========================
    with tab3:
        st.subheader("Student Comparison")
        s1, s2 = st.columns(2)
        stu1 = s1.selectbox("Select First Student", students["Name"].unique())
        stu2 = s2.selectbox("Select Second Student", students["Name"].unique())

        comp = marks.loc[
            (marks["Name"].isin([stu1, stu2])) & (~marks["WasAbsent"])
        ].groupby(["Name", "Subject"])["Marks"].mean().reset_index()

        fig_comp = px.bar(
            comp,
            x="Subject",
            y="Marks",
            color="Name",
            barmode="group",
            title=f"{stu1} vs {stu2} - Subject Comparison",
        )
        st.plotly_chart(fig_comp, use_container_width=True)

        if st.checkbox("Show Explanation (Student Comparison)"):
            st.info(
                "This chart compares two students across subjects. "
                "It only includes exams they actually attempted (absences excluded)."
            )

    # ========================
    # TAB 4: TRENDS & CLUSTERS
    # ========================
    with tab4:
        st.subheader("Trends & Clusters")

        # Time series
        avg_time = (
            marks.loc[~marks["WasAbsent"]]
            .groupby(["ExamNumber"])["Marks"]
            .mean()
            .reset_index()
        )
        fig_time = px.line(
            avg_time,
            x="ExamNumber",
            y="Marks",
            title="Class Average Over Exams",
            markers=True,
        )
        st.plotly_chart(fig_time, use_container_width=True)

        if st.checkbox("Show Explanation (Class Average Trend)"):
            st.info(
                "This line chart shows how the class average evolves across exam numbers. "
                "It helps identify improvement or decline over time."
            )

        # Clustering
        X = students[["AttendanceRate", "AvgScore"]].dropna()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        X["Cluster"] = clusters
        X["Name"] = students.loc[X.index, "Name"]

        fig_cluster = px.scatter(
            X,
            x="AttendanceRate",
            y="AvgScore",
            color="Cluster",
            text="Name",
            title="Student Clusters",
        )
        st.plotly_chart(fig_cluster, use_container_width=True)

        if st.checkbox("Show Explanation (Clusters)"):
            st.info(
                "Students are grouped into clusters based on attendance and performance. "
                "This helps identify groups like 'High Achievers', 'Struggling Students', "
                "and 'Disengaged but Bright'."
            )

else:
    st.warning("Please upload both Attendance and Marks CSV files to continue.")
