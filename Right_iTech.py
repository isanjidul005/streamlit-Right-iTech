# ================================
# Right iTech - Student Insights App
# ================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import base64
from datetime import datetime
import tempfile

# For PDF export
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4

st.set_page_config(page_title="Right iTech", layout="wide", initial_sidebar_state="expanded")

# Unified color palette
PALETTE = px.colors.qualitative.Set2
SUBJECT_COLORS = {}

def assign_subject_colors(subjects):
    for i, sub in enumerate(sorted(subjects)):
        SUBJECT_COLORS[sub] = PALETTE[i % len(PALETTE)]

# ================================
# File Uploads
# ================================
st.sidebar.header("Upload Data")
marks_file = st.sidebar.file_uploader("Upload Marks CSV", type="csv")
att_file = st.sidebar.file_uploader("Upload Attendance CSV", type="csv")

if marks_file is not None and att_file is not None:
    marks_df = pd.read_csv(marks_file)
    att_df = pd.read_csv(att_file)
else:
    marks_df = pd.read_csv("/mnt/data/cleanest_marks.csv")
    att_df = pd.read_csv("/mnt/data/combined_attendance.csv")

# ================================
# Data Cleaning
# ================================
marks_df.columns = [c.strip() for c in marks_df.columns]
att_df.columns = [c.strip() for c in att_df.columns]

marks_df['Marks'] = pd.to_numeric(marks_df['Marks'], errors="coerce")
marks_df['FullMarks'] = pd.to_numeric(marks_df['FullMarks'], errors="coerce")

att_df['Date'] = pd.to_datetime(att_df['Date'], dayfirst=True, errors="coerce")
att_df['_present_flag_'] = att_df['Status'].astype(str).str.upper().map(
    {"P": 1, "PRESENT": 1, "A": 0, "ABSENT": 0}
)

if "Subject" in marks_df.columns:
    assign_subject_colors(marks_df["Subject"].dropna().unique())

# ================================
# Tabs
# ================================
tabs = st.tabs([
    "Class Overview",
    "Student Profile",
    "Comparison Between Students",
    "Attendance Explorer",
    "Insights & Alerts"
])

# ================================
# Class Overview
# ================================
with tabs[0]:
    st.title("📊 Class Overview")

    total_students = marks_df["ID"].nunique()
    avg_marks = marks_df["Marks"].mean()
    avg_att = att_df["_present_flag_"].mean()

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Students", total_students)
    c2.metric("Avg Marks", f"{avg_marks:.1f}")
    c3.metric("Avg Attendance", f"{avg_att:.1%}")

    st.subheader("Overall Score Distribution")
    fig = px.histogram(marks_df, x="Marks", nbins=20,
                       color="Subject", color_discrete_map=SUBJECT_COLORS)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Average Marks by Subject")
    subj_avg = marks_df.groupby("Subject")["Marks"].mean().reset_index()
    fig2 = px.bar(subj_avg, x="Subject", y="Marks",
                  color="Subject", color_discrete_map=SUBJECT_COLORS)
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Subjects Ranked by Difficulty (Lowest Avg)")
    subj_rank = subj_avg.sort_values("Marks")
    fig3 = px.bar(subj_rank, x="Marks", y="Subject",
                  orientation="h", color="Subject", color_discrete_map=SUBJECT_COLORS)
    st.plotly_chart(fig3, use_container_width=True)

# ================================
# Student Profile
# ================================
with tabs[1]:
    st.title("👤 Student Profile")
    student = st.selectbox("Select Student", marks_df["Name"].unique())
    sdata = marks_df[marks_df["Name"] == student]
    s_att = att_df[att_df["Name"] == student]

    if not sdata.empty:
        sid = sdata["ID"].iloc[0]
        sroll = sdata["Roll"].iloc[0]
        sname = sdata["Name"].iloc[0]

        st.markdown(
            f"""
            ### {sname}  
            **ID:** {sid} | **Roll:** {sroll}  
            **Average Marks:** {sdata['Marks'].mean():.1f}  
            **Attendance Rate:** {s_att['_present_flag_'].mean()*100:.1f}%
            """
        )

        st.subheader("Marks by Exam")
        fig = px.bar(sdata, x="ExamNumber", y="Marks",
                     color="Subject", barmode="group",
                     color_discrete_map=SUBJECT_COLORS)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Average Marks by Subject")
        subj_avg = sdata.groupby("Subject")["Marks"].mean().reset_index()
        fig2 = px.bar(subj_avg, x="Subject", y="Marks",
                      color="Subject", color_discrete_map=SUBJECT_COLORS)
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Attendance Summary")
        if not s_att.empty:
            att_month = s_att.groupby(s_att["Date"].dt.to_period("M"))["_present_flag_"].mean().reset_index()
            att_month["Date"] = att_month["Date"].astype(str)
            fig3 = px.bar(att_month, x="Date", y="_present_flag_", color="_present_flag_",
                          color_continuous_scale="RdYlGn")
            st.plotly_chart(fig3, use_container_width=True)

        # Export buttons
        st.subheader("📥 Export Student Report")
        excel_buffer = BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
            sdata.to_excel(writer, sheet_name="Marks", index=False)
            s_att.to_excel(writer, sheet_name="Attendance", index=False)
        st.download_button("Download Excel Report",
                           excel_buffer.getvalue(),
                           f"{sname}_report.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        pdf_buffer = BytesIO()
        doc = SimpleDocTemplate(pdf_buffer, pagesize=A4)
        styles = getSampleStyleSheet()
        flowables = [
            Paragraph(f"Student Report - {sname}", styles["Title"]),
            Spacer(1, 12),
            Paragraph(f"ID: {sid} | Roll: {sroll}", styles["Normal"]),
            Spacer(1, 12),
            Paragraph("Marks Summary", styles["Heading2"]),
            Paragraph(f"Average Marks: {sdata['Marks'].mean():.2f}", styles["Normal"]),
            Spacer(1, 12),
            Paragraph("Attendance Summary", styles["Heading2"]),
            Paragraph(f"Attendance Rate: {s_att['_present_flag_'].mean()*100:.1f}%", styles["Normal"]),
        ]
        doc.build(flowables)
        st.download_button("Download PDF Report",
                           pdf_buffer.getvalue(),
                           f"{sname}_report.pdf",
                           "application/pdf")

# ================================
# Comparison Between Students
# ================================
with tabs[2]:
    st.title("🤝 Compare Students")
    students = st.multiselect("Select Students", marks_df["Name"].unique())
    if len(students) >= 2:
        comp = marks_df[marks_df["Name"].isin(students)]
        avg = comp.groupby(["Name", "Subject"])["Marks"].mean().reset_index()

        st.subheader("Average Marks by Subject (Comparison)")
        fig = px.bar(avg, x="Subject", y="Marks",
                     color="Name", barmode="group")
        st.plotly_chart(fig, use_container_width=True)

        exam = st.selectbox("Select Exam Number", comp["ExamNumber"].unique())
        exam_data = comp[comp["ExamNumber"] == exam]
        st.subheader(f"Exam {exam} Comparison")
        fig2 = px.bar(exam_data, x="Subject", y="Marks",
                      color="Name", barmode="group")
        st.plotly_chart(fig2, use_container_width=True)

# ================================
# Attendance Explorer
# ================================
with tabs[3]:
    st.title("📅 Attendance Explorer")
    att_over_time = att_df.groupby(att_df["Date"].dt.date)["_present_flag_"].mean().reset_index().rename(columns={"_present_flag_": "attendance_rate"})
    fig = px.line(att_over_time, x="Date", y="attendance_rate")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Attendance by Student")
    att_summary = att_df.groupby("Name")["_present_flag_"].mean().reset_index()
    att_summary["attendance_rate"] = att_summary["_present_flag_"] * 100
    fig2 = px.bar(att_summary, x="Name", y="attendance_rate", color="attendance_rate",
                  color_continuous_scale="RdYlGn")
    st.plotly_chart(fig2, use_container_width=True)

# ================================
# Insights & Alerts
# ================================
with tabs[4]:
    st.title("🚨 Insights & Alerts")
    avg_marks = marks_df.groupby("Name")["Marks"].mean().reset_index()
    avg_att = att_df.groupby("Name")["_present_flag_"].mean().reset_index()
    merged = pd.merge(avg_marks, avg_att, on="Name", how="inner")
    flagged = merged[(merged["Marks"] < 40) | (merged["_present_flag_"] < 0.75)]

    st.subheader("Flagged Students")
    flagged_display = flagged.copy()
    flagged_display["_present_flag_"] = flagged_display["_present_flag_"].apply(lambda x: f"{x:.1%}")
    st.table(flagged_display.set_index("Name"))

    st.download_button("Download Flagged Students CSV",
                       flagged.to_csv(index=False).encode("utf-8"),
                       "flagged_students.csv",
                       "text/csv")

    st.subheader("Attendance Leaders")
    top_present = avg_att.sort_values("_present_flag_", ascending=False).head(5).copy()
    top_present["_present_flag_"] = top_present["_present_flag_"].apply(lambda x: f"{x:.1%}")
    st.table(top_present.set_index("Name"))

    st.subheader("Lowest Attendance")
    top_absent = avg_att.sort_values("_present_flag_").head(5).copy()
    top_absent["_present_flag_"] = top_absent["_present_flag_"].apply(lambda x: f"{x:.1%}")
    st.table(top_absent.set_index("Name"))
