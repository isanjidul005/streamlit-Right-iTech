import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

st.set_page_config(page_title="Class Dashboard", layout="wide")

st.title("📊 World-Class Student Dashboard")
st.markdown("Upload your attendance and marks data to get started.")

# ================= FILE UPLOAD =====================
att_file = st.file_uploader("Upload Attendance CSV", type="csv")
marks_file = st.file_uploader("Upload Marks CSV", type="csv")

if att_file and marks_file:
    attendance = pd.read_csv(att_file)
    marks = pd.read_csv(marks_file)

    # ================= DATA PROCESSING =====================
    # Attendance summary
    att_summary = (
        attendance.groupby(["ID", "Name"])
        .agg(Present=("Status", lambda x: (x == "Present").sum()),
             Total=("Status", "count"))
        .reset_index()
    )
    att_summary["AttendanceRate"] = (att_summary["Present"] / att_summary["Total"]) * 100

    # Marks summary
    marks_summary = (
        marks.groupby(["ID", "Name"])
        .agg(AverageScore=("Marks", "mean"))
        .reset_index()
    )

    # --- Ensure consistent merge keys ---
    att_summary["ID"] = att_summary["ID"].astype(str)
    marks_summary["ID"] = marks_summary["ID"].astype(str)
    att_summary["Name"] = att_summary["Name"].astype(str).str.strip()
    marks_summary["Name"] = marks_summary["Name"].astype(str).str.strip()

    # --- Merge datasets ---
    if set(["ID", "Name"]).issubset(marks_summary.columns):
        students = pd.merge(att_summary, marks_summary, on=["ID", "Name"], how="left")
    else:
        students = pd.merge(att_summary, marks_summary, on="Name", how="left")

    # ================= TABS =====================
    tabs = st.tabs([
        "📈 Class Overview",
        "👤 Student Profile",
        "⚖️ Student Comparison",
        "🔍 Clustering Insights"
    ])

    # -------- CLASS OVERVIEW TAB --------
    with tabs[0]:
        st.header("📈 Class Overview")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Attendance Distribution")
            fig, ax = plt.subplots()
            sns.histplot(students["AttendanceRate"], bins=10, kde=True, ax=ax)
            st.pyplot(fig)

            with st.expander("ℹ️ What this means"):
                st.write("""
                This chart shows how often students attend class.  
                - **Left side = low attendance**  
                - **Right side = high attendance**  
                Peaks show where most students fall.  
                """)

        with col2:
            st.subheader("Performance Distribution")
            fig, ax = plt.subplots()
            sns.histplot(students["AverageScore"], bins=10, kde=True, ax=ax)
            st.pyplot(fig)

            with st.expander("ℹ️ What this means"):
                st.write("""
                This shows how exam scores are spread across the class.  
                The higher the peak, the more students fall into that score range.  
                """)

        st.subheader("Attendance vs Performance")
        fig, ax = plt.subplots()
        sns.scatterplot(
            data=students,
            x="AttendanceRate",
            y="AverageScore",
            hue="AverageScore",
            palette="viridis",
            ax=ax
        )
        st.pyplot(fig)

        with st.expander("ℹ️ What this means"):
            st.write("""
            Each point = one student.  
            - **X-axis:** Attendance %  
            - **Y-axis:** Average Score  
            A rising trend suggests that better attendance = better marks.  
            """)

    # -------- STUDENT PROFILE TAB --------
    with tabs[1]:
        st.header("👤 Student Profile")

        selected_student = st.selectbox("Select Student", students["Name"].unique())
        student_data = students[students["Name"] == selected_student]

        if not student_data.empty:
            st.metric("Attendance Rate (%)", round(student_data["AttendanceRate"].iloc[0], 2))
            st.metric("Average Score", round(student_data["AverageScore"].iloc[0], 2))

            st.subheader("📅 Subject-wise Performance")
            subj_perf = marks[marks["Name"] == selected_student]

            fig, ax = plt.subplots()
            sns.barplot(data=subj_perf, x="Subject", y="Marks", ax=ax)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            st.pyplot(fig)

            with st.expander("ℹ️ What this means"):
                st.write("""
                This shows how the student performs in each subject.  
                You can quickly see their strengths and weaknesses.  
                """)

    # -------- STUDENT COMPARISON TAB --------
    with tabs[2]:
        st.header("⚖️ Compare Two Students")

        col1, col2 = st.columns(2)
        student1 = col1.selectbox("Select Student 1", students["Name"].unique())
        student2 = col2.selectbox("Select Student 2", students["Name"].unique())

        s1 = students[students["Name"] == student1]
        s2 = students[students["Name"] == student2]

        if not s1.empty and not s2.empty:
            col1.metric(student1 + " Avg Score", round(s1["AverageScore"].iloc[0], 2))
            col2.metric(student2 + " Avg Score", round(s2["AverageScore"].iloc[0], 2))

            # Subject-wise comparison
            comp = marks[marks["Name"].isin([student1, student2])]
            fig, ax = plt.subplots()
            sns.barplot(data=comp, x="Subject", y="Marks", hue="Name", ax=ax)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            st.pyplot(fig)

            with st.expander("ℹ️ What this means"):
                st.write("""
                Here you see how both students perform in each subject side-by-side.  
                It's a direct comparison of strengths and weaknesses.  
                """)

    # -------- CLUSTERING TAB --------
    with tabs[3]:
        st.header("🔍 Clustering Insights")

        X = students[["AttendanceRate", "AverageScore"]].fillna(0)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        students["Cluster"] = kmeans.fit_predict(X_scaled)

        fig, ax = plt.subplots()
        sns.scatterplot(
            data=students,
            x="AttendanceRate",
            y="AverageScore",
            hue="Cluster",
            palette="Set1",
            ax=ax
        )
        st.pyplot(fig)

        with st.expander("ℹ️ What this means"):
            st.write("""
            Students are grouped into clusters:
            - High Attendance + High Performance  
            - Low Attendance + Low Performance  
            - Mixed behavior  
            
            This helps identify who needs support and who excels.  
            """)

else:
    st.warning("⬆️ Please upload both attendance and marks CSV files.")
