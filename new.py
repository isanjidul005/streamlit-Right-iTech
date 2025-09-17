import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Class Dashboard", layout="wide")

# -----------------
# FILE UPLOAD
# -----------------
st.sidebar.header("📂 Upload Your Data")
att_file = st.sidebar.file_uploader("Upload Clean Attendance CSV", type=["csv"])
marks_file = st.sidebar.file_uploader("Upload Clean Marks CSV", type=["csv"])

if att_file and marks_file:
    attendance = pd.read_csv(att_file)
    marks = pd.read_csv(marks_file)

    # Preprocess attendance
    att_summary = (
        attendance.groupby(["ID", "Name"])
        .agg(Present=("Status", lambda x: (x == "Present").sum()),
             Total=("Status", "count"))
        .reset_index()
    )
    att_summary["AttendanceRate"] = (att_summary["Present"] / att_summary["Total"]) * 100

    # Preprocess marks
    marks_summary = (
        marks.groupby(["ID", "Name"])
        .agg(AverageScore=("Marks", "mean"))
        .reset_index()
    )

    # Merge
    students = pd.merge(att_summary, marks_summary, on=["ID", "Name"], how="left")

    # -----------------
    # DASHBOARD TABS
    # -----------------
    tabs = st.tabs([
        "📊 Class Overview",
        "🧑 Student Profile",
        "⚖️ Student Comparison",
        "📘 Subject Analysis",
        "📈 Attendance vs Performance",
        "⏳ Trends Over Time",
        "🔎 Clustering & Segmentation"
    ])

    # -----------------
    # TAB 1: CLASS OVERVIEW
    # -----------------
    with tabs[0]:
        st.header("📊 Class Overview")

        # Attendance distribution
        st.subheader("Attendance Distribution")
        cutoff = st.slider("⚠️ Attendance risk cutoff (%)", 50, 100, 75)
        fig1 = px.violin(students, y="AttendanceRate", box=True, points="all", color_discrete_sequence=["#636EFA"])
        st.plotly_chart(fig1, use_container_width=True)

        low_att = students[students["AttendanceRate"] < cutoff]
        st.markdown(f"**Insight:** {len(low_att)} students fall below {cutoff}% attendance. "
                    "These may be at academic risk.")

        with st.expander("📘 Explanation"):
            st.write("""
            A violin plot shows the distribution of attendance across students.
            - The thicker parts = many students at that rate.
            - Thin ends = outliers (very high/low).
            - White box = quartiles & median.
            """)

        # Score distribution
        st.subheader("Score Distribution")
        fig2 = px.histogram(students, x="AverageScore", nbins=20, marginal="box",
                            color_discrete_sequence=["#EF553B"])
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("**Insight:** This shows how scores are spread. Left skew = many low performers, right skew = many high performers.")

    # -----------------
    # TAB 2: STUDENT PROFILE
    # -----------------
    with tabs[1]:
        st.header("🧑 Student Profile")
        selected = st.selectbox("Select a Student", students["Name"].unique())

        profile = students[students["Name"] == selected].iloc[0]

        st.metric("Attendance Rate", f"{profile.AttendanceRate:.1f}%")
        st.metric("Average Score", f"{profile.AverageScore:.1f}")

        # Subject radar
        sub_scores = marks[marks["Name"] == selected].groupby("Subject")["Marks"].mean()
        radar = go.Figure()
        radar.add_trace(go.Scatterpolar(
            r=sub_scores.values,
            theta=sub_scores.index,
            fill='toself',
            name=selected
        ))
        st.plotly_chart(radar, use_container_width=True)

        with st.expander("📘 Explanation"):
            st.write("""
            A radar chart shows performance across subjects.
            - Each spoke = one subject.
            - Further out = higher score.
            - Shape gives a quick view of strengths/weaknesses.
            """)

    # -----------------
    # TAB 3: STUDENT COMPARISON
    # -----------------
    with tabs[2]:
        st.header("⚖️ Student Comparison")
        stu1, stu2 = st.select_slider("Select two students", options=students["Name"].unique(), value=(
            students["Name"].iloc[0], students["Name"].iloc[1]))

        s1 = marks[marks["Name"] == stu1].groupby("Subject")["Marks"].mean()
        s2 = marks[marks["Name"] == stu2].groupby("Subject")["Marks"].mean()
        compare = pd.DataFrame({"Subject": s1.index, stu1: s1.values, stu2: s2.values})

        fig3 = px.bar(compare, x="Subject", y=[stu1, stu2], barmode="group")
        st.plotly_chart(fig3, use_container_width=True)

        st.markdown("**Insight:** This bar chart compares subject performance side-by-side.")

    # -----------------
    # TAB 4: SUBJECT ANALYSIS
    # -----------------
    with tabs[3]:
        st.header("📘 Subject Analysis")
        sub = st.selectbox("Select Subject", marks["Subject"].unique())
        sub_data = marks[marks["Subject"] == sub]

        fig4 = px.box(sub_data, x="Subject", y="Marks", points="all", color="Subject")
        st.plotly_chart(fig4, use_container_width=True)

        avg = sub_data["Marks"].mean()
        st.markdown(f"**Insight:** Average score in {sub} = {avg:.1f}. "
                    f"{len(sub_data[sub_data['Marks'] < avg])} students are below average.")

    # -----------------
    # TAB 5: ATTENDANCE VS PERFORMANCE
    # -----------------
    with tabs[4]:
        st.header("📈 Attendance vs Performance")

        fig5 = px.scatter(students, x="AttendanceRate", y="AverageScore", trendline="ols", color="AttendanceRate")
        st.plotly_chart(fig5, use_container_width=True)

        corr = students["AttendanceRate"].corr(students["AverageScore"])
        st.markdown(f"**Insight:** Correlation between attendance and performance = **{corr:.2f}**. "
                    "Closer to 1 = strong positive link.")

    # -----------------
    # TAB 6: TRENDS OVER TIME
    # -----------------
    with tabs[5]:
        st.header("⏳ Trends Over Time")
        name = st.selectbox("Select Student", students["Name"].unique())
        trend = marks[marks["Name"] == name]

        fig6 = px.line(trend, x="ExamNumber", y="Marks", color="Subject", markers=True)
        st.plotly_chart(fig6, use_container_width=True)

        st.markdown(f"**Insight:** This shows {name}'s performance over time, subject by subject.")

    # -----------------
    # TAB 7: CLUSTERING
    # -----------------
    with tabs[6]:
        st.header("🔎 Clustering & Segmentation")

        features = students[["AttendanceRate", "AverageScore"]].dropna()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(features)

        k = st.slider("Number of clusters", 2, 6, 3)
        kmeans = KMeans(n_clusters=k, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)

        features["Cluster"] = clusters
        fig7 = px.scatter(features, x="AttendanceRate", y="AverageScore", color="Cluster")
        st.plotly_chart(fig7, use_container_width=True)

        st.markdown("**Insight:** Students are grouped into clusters like 'High Attendance + High Score', "
                    "'Low Attendance + Low Score', etc. Useful for interventions.")

else:
    st.warning("⬅️ Please upload both Attendance and Marks CSV files to start.")
