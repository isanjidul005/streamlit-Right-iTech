import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans

# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="Class Dashboard", layout="wide")

st.title("📚 Class Performance & Attendance Dashboard")

# -------------------------
# File Upload
# -------------------------
st.sidebar.header("Upload Your Data")
uploaded_att = st.sidebar.file_uploader("Upload Attendance CSV", type="csv")
uploaded_marks = st.sidebar.file_uploader("Upload Marks CSV", type="csv")

if uploaded_att and uploaded_marks:
    # Load data
    att = pd.read_csv(uploaded_att)
    marks = pd.read_csv(uploaded_marks)

    st.success("✅ Data successfully loaded!")

    # -------------------------
    # Previews
    # -------------------------
    with st.expander("👀 Preview Attendance Data"):
        st.dataframe(att.head(20))

    with st.expander("👀 Preview Marks Data"):
        st.dataframe(marks.head(20))

    # -------------------------
    # Processed Data
    # -------------------------
    # Attendance summary per student
    attendance_summary = (
        att.groupby(["ID", "Name", "Gender"])
        .agg(
            total_days=("Date", "count"),
            days_present=("Status", lambda x: (x == "Present").sum()),
        )
        .reset_index()
    )
    attendance_summary["attendance_rate"] = (
        attendance_summary["days_present"] / attendance_summary["total_days"] * 100
    )

    # Marks summary per student
    marks_clean = marks.copy()
    marks_clean["Marks"] = pd.to_numeric(marks_clean["Marks"], errors="coerce")

    marks_summary = (
        marks_clean.groupby(["ID", "Name"])  # 👈 FIXED: no Gender here
        .agg(avg_marks=("Marks", "mean"))
        .reset_index()
    )

    # Merge both
    merged = pd.merge(attendance_summary, marks_summary, on=["ID", "Name"], how="left")

    # -------------------------
    # Tabs
    # -------------------------
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "📊 Class Overview",
            "👤 Student Profile",
            "⚖️ Compare Students",
            "📈 Trends",
            "🧩 Clustering",
        ]
    )

    # -------------------------
    # TAB 1: Class Overview
    # -------------------------
    with tab1:
        st.header("📊 Class Overview")

        col1, col2 = st.columns(2)

        with col1:
            fig = px.histogram(
                merged, x="attendance_rate", nbins=10, color="Gender",
                title="Attendance Rate Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
            with st.expander("ℹ️ Explanation"):
                st.write("""
                This histogram shows the spread of attendance rates.  
                Peaks indicate the most common attendance levels.  
                Color coding highlights gender differences.
                """)

        with col2:
            fig = px.histogram(
                merged, x="avg_marks", nbins=10, color="Gender",
                title="Marks Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
            with st.expander("ℹ️ Explanation"):
                st.write("""
                This histogram shows how marks are distributed across students.  
                It helps identify clusters of low and high achievers.
                """)

        # Correlation scatter
        fig = px.scatter(
            merged, x="attendance_rate", y="avg_marks", color="Gender",
            trendline="ols", title="Attendance vs Marks"
        )
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("ℹ️ Explanation"):
            st.write("""
            This scatter plot reveals the relationship between attendance and performance.  
            - A strong upward trend suggests attendance improves marks.  
            - Outliers can be spotted (high attendance but low marks, or vice versa).  
            """)

    # -------------------------
    # TAB 2: Student Profile
    # -------------------------
    with tab2:
        st.header("👤 Student Profile")
        student = st.selectbox("Select a student:", merged["Name"].unique())

        stud_att = attendance_summary[attendance_summary["Name"] == student]
        stud_marks = marks_clean[marks_clean["Name"] == student]

        st.subheader(f"📌 Profile: {student}")
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Attendance Rate (%)", f"{stud_att['attendance_rate'].values[0]:.2f}")

        with col2:
            st.metric("Average Marks", f"{stud_marks['Marks'].mean():.2f}")

        # Radar chart for subject performance
        radar_data = (
            stud_marks.groupby("Subject")
            .agg(avg=("Marks", "mean"))
            .reset_index()
        )
        fig = px.line_polar(radar_data, r="avg", theta="Subject", line_close=True)
        st.plotly_chart(fig, use_container_width=True)

    # -------------------------
    # TAB 3: Compare Students
    # -------------------------
    with tab3:
        st.header("⚖️ Compare Students")
        students = st.multiselect(
            "Select students to compare:", merged["Name"].unique(), default=merged["Name"].unique()[:2]
        )
        compare_data = marks_clean[marks_clean["Name"].isin(students)]
        fig = px.line(
            compare_data, x="ExamNumber", y="Marks", color="Name", facet_col="Subject",
            title="Student Comparison by Subject"
        )
        st.plotly_chart(fig, use_container_width=True)

    # -------------------------
    # TAB 4: Trends
    # -------------------------
    with tab4:
        st.header("📈 Trends Over Time")
        subj = st.selectbox("Select Subject:", marks_clean["Subject"].unique())
        subj_data = marks_clean[marks_clean["Subject"] == subj]

        fig = px.line(
            subj_data, x="ExamNumber", y="Marks", color="Name",
            title=f"{subj} - Performance Over Time"
        )
        st.plotly_chart(fig, use_container_width=True)

    # -------------------------
    # TAB 5: Clustering
    # -------------------------
    with tab5:
        st.header("🧩 Student Clustering")

        # KMeans clustering
        X = merged[["attendance_rate", "avg_marks"]].dropna()
        if len(X) >= 3:  # ensure enough students
            kmeans = KMeans(n_clusters=3, random_state=42).fit(X)
            merged["cluster"] = kmeans.labels_
        else:
            merged["cluster"] = 0

        fig = px.scatter(
            merged, x="attendance_rate", y="avg_marks", color="cluster",
            hover_data=["Name"], title="Clustering Students by Attendance & Marks"
        )
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("ℹ️ Explanation"):
            st.write("""
            This clustering groups students into categories such as:  
            - High Attendance & High Performance  
            - Low Attendance & Low Performance  
            - Outliers with mixed behaviors  
            """)

else:
    st.warning("👆 Please upload both **Attendance** and **Marks** CSV files to begin.")
