# Right_iTech_final_polished.py
# Streamlit app — Right iTech (professional, theme-agnostic, Plotly visuals)

import os
from datetime import date
import io

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="Right iTech", layout="wide", initial_sidebar_state="expanded")
px.defaults.template = "plotly_white"  # theme-agnostic baseline

# -------------------------
# Professional color palette (distinct colors)
# -------------------------
DISTINCT_PALETTE = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#9467bd",  # purple
    "#d62728",  # red
    "#17becf",  # teal
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # gray
]
ATT_PRESENT_COLOR = "#2ca02c"
ATT_ABSENT_COLOR = "#d62728"
NEUTRAL = "#4a4a4a"

# -------------------------
# Sidebar: upload + options
# -------------------------
st.sidebar.header("Upload data & preferences")
att_upload = st.sidebar.file_uploader("Attendance CSV (optional)", type=["csv"])
marks_upload = st.sidebar.file_uploader("Marks CSV (optional)", type=["csv"])

# fallback paths (if files already on server)
FALLBACK_ATT = "/mnt/data/combined_attendance.csv"
FALLBACK_MARKS = "/mnt/data/cleanest_marks.csv"

@st.cache_data
def safe_read_csv(uploaded_file, fallback_path):
    try:
        if uploaded_file is not None:
            return pd.read_csv(uploaded_file)
        if fallback_path and os.path.exists(fallback_path):
            return pd.read_csv(fallback_path)
    except Exception:
        return pd.DataFrame()
    return pd.DataFrame()

att_df = safe_read_csv(att_upload, FALLBACK_ATT)
marks_df = safe_read_csv(marks_upload, FALLBACK_MARKS)

st.sidebar.markdown("---")
auto_expand = st.sidebar.checkbox("Auto-expand explanations", value=False)
pass_threshold = st.sidebar.number_input("Pass threshold (marks)", min_value=0, max_value=100, value=40)
flag_score_threshold = st.sidebar.number_input("Flag if avg score < (score)", min_value=0, max_value=100, value=40)
flag_att_threshold_pct = st.sidebar.slider("Flag if attendance < (%)", 0, 100, 75)

# -------------------------
# Title (no white box) — centered and readable
# -------------------------
st.markdown(
    "<h1 style='text-align:center; color:#1f77b4; margin-bottom:4px;'>Right iTech</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align:center; color:#666; margin-top:0px; margin-bottom:12px;'>Professional, readable analytics for marks & attendance — interactive visuals.</p>",
    unsafe_allow_html=True,
)
st.write("---")

# -------------------------
# Defensive cleaning & normalization
# -------------------------
if att_df is None:
    att_df = pd.DataFrame()
if marks_df is None:
    marks_df = pd.DataFrame()

# normalize column names
if not att_df.empty:
    att_df.columns = [c.strip() for c in att_df.columns]
if not marks_df.empty:
    marks_df.columns = [c.strip() for c in marks_df.columns]

# parse Date safely
if not att_df.empty and "Date" in att_df.columns:
    att_df["Date"] = pd.to_datetime(att_df["Date"], dayfirst=True, errors="coerce")

# unified present flag
if not att_df.empty:
    if "Status" in att_df.columns:
        att_df["_present_flag_"] = att_df["Status"].astype(str).str.upper().map({
            "P":1,"PRESENT":1,"1":1,"Y":1,"YES":1,
            "A":0,"ABSENT":0,"0":0,"N":0,"NO":0
        })
        att_df["_present_flag_"] = att_df["_present_flag_"].fillna(att_df["Status"].astype(str).str[0].map({"P":1,"A":0}))
    elif "Attendance" in att_df.columns:
        att_df["_present_flag_"] = att_df["Attendance"].astype(str).str.upper().map({"PRESENT":1,"ABSENT":0,"P":1,"A":0})
    else:
        att_df["_present_flag_"] = np.nan
else:
    att_df["_present_flag_"] = pd.Series(dtype=float)

# ensure numeric Marks
if not marks_df.empty:
    if "Marks" in marks_df.columns:
        marks_df["Marks"] = pd.to_numeric(marks_df["Marks"], errors="coerce")
    else:
        marks_df["Marks"] = np.nan
    if "FullMarks" in marks_df.columns:
        marks_df["FullMarks"] = pd.to_numeric(marks_df["FullMarks"], errors="coerce")
else:
    marks_df = pd.DataFrame(columns=["ID","Roll","Name","Subject","ExamNumber","Exam","ExamType","Marks","FullMarks"])

# subject color mapping (distinct & consistent)
def assign_subject_colors(subjects):
    subs = sorted([s for s in subjects if pd.notna(s)])
    mapping = {}
    for i, s in enumerate(subs):
        mapping[s] = DISTINCT_PALETTE[i % len(DISTINCT_PALETTE)]
    return mapping

SUBJECT_COLORS = assign_subject_colors(marks_df["Subject"].unique()) if ("Subject" in marks_df.columns and not marks_df.empty) else {}

# -------------------------
# Utility helpers
# -------------------------
def safe_mean(series):
    return series.mean() if series.notna().any() else np.nan

def student_summary_df(mdf):
    if mdf.empty:
        return pd.DataFrame()
    s = mdf.groupby(["ID","Roll","Name"], as_index=False).agg(
        avg_score=("Marks","mean"),
        exams_taken=("ExamNumber","nunique"),
        records=("Marks","count")
    )
    return s

# -------------------------
# If no data, show message and stop
# -------------------------
if att_df.empty and marks_df.empty:
    st.warning("No data detected. Upload Attendance and Marks CSVs in the sidebar or place fallback files at /mnt/data/*.csv")
    st.stop()

# -------------------------
# Global filters (sidebar)
# -------------------------
st.sidebar.header("Global filters")
if not att_df.empty and "Date" in att_df.columns and not att_df["Date"].isna().all():
    min_date = att_df["Date"].min().date()
    max_date = att_df["Date"].max().date()
else:
    min_date = date.today(); max_date = date.today()

date_range = st.sidebar.date_input("Attendance date range", value=(min_date, max_date))

subject_options = sorted(marks_df["Subject"].dropna().unique().tolist()) if (not marks_df.empty and "Subject" in marks_df.columns) else []
subject_filter = st.sidebar.multiselect("Filter subjects", options=subject_options, default=subject_options)

exam_options = sorted(marks_df["ExamNumber"].dropna().unique().tolist()) if (not marks_df.empty and "ExamNumber" in marks_df.columns) else []
exam_filter = st.sidebar.multiselect("Filter exams", options=exam_options, default=exam_options)

name_search = st.sidebar.text_input("Search student name (partial)")

# -------------------------
# Layout tabs
# -------------------------
tabs = st.tabs(["Class Overview","Student Dashboard","Compare Students","Attendance","Marks","Insights"])

# ===== Tab 0: Class Overview =====
with tabs[0]:
    st.header("Class Overview (snapshot)")

    # top metrics
    col1, col2, col3, col4 = st.columns(4)
    ids_set = set()
    if not marks_df.empty and "ID" in marks_df.columns:
        ids_set.update(marks_df["ID"].dropna().astype(str).tolist())
    if not att_df.empty and "ID" in att_df.columns:
        ids_set.update(att_df["ID"].dropna().astype(str).tolist())
    total_students = len(ids_set) if ids_set else (marks_df["Name"].nunique() if not marks_df.empty and "Name" in marks_df.columns else 0)
    col1.metric("Total students", total_students)

    # gender counts
    if not att_df.empty and "Gender" in att_df.columns:
        gender_counts = att_df.drop_duplicates(subset=["ID"])[["ID","Gender"]].groupby("Gender").size().to_dict()
        boys = gender_counts.get("M", gender_counts.get("Male", 0))
        girls = gender_counts.get("F", gender_counts.get("Female", 0))
        col2.metric("Boys", int(boys))
        col3.metric("Girls", int(girls))
    else:
        col2.metric("Boys", "N/A"); col3.metric("Girls", "N/A")

    avg_att = att_df["_present_flag_"].mean() if not att_df.empty else np.nan
    col4.metric("Avg attendance", f"{avg_att*100:.1f}%" if not np.isnan(avg_att) else "N/A")

    st.markdown("---")

    # filtered marks based on subject/exam/name filters
    marks_filtered = marks_df.copy()
    if subject_filter:
        marks_filtered = marks_filtered[marks_filtered["Subject"].isin(subject_filter)]
    if exam_filter:
        marks_filtered = marks_filtered[marks_filtered["ExamNumber"].isin(exam_filter)]
    if name_search:
        marks_filtered = marks_filtered[marks_filtered["Name"].str.contains(name_search, case=False, na=False)]

    # Score histogram
    st.subheader("Score distribution")
    if not marks_filtered.empty and "Marks" in marks_filtered.columns:
        fig_hist = px.histogram(marks_filtered, x="Marks", nbins=25, title="Distribution of marks (filtered)", color_discrete_sequence=[DISTINCT_PALETTE[0]])
        st.plotly_chart(fig_hist, use_container_width=True)
        with st.expander("Explanation", expanded=auto_expand):
            st.write(
                "Histogram of marks across the dataset (filters applied). "
                "This helps spot overall performance: many low bars = many low scores; a high concentration near the top means many high performers."
            )
    else:
        st.info("Not enough marks data to show distribution.")

    st.markdown("---")

    # Attendance quality buckets (student-level)
    st.subheader("Attendance quality (student buckets)")
    if not att_df.empty and "Name" in att_df.columns and "_present_flag_" in att_df.columns:
        try:
            sd, ed = (date_range[0], date_range[1]) if isinstance(date_range, (list,tuple)) and len(date_range)==2 else (date_range, date_range)
        except Exception:
            sd, ed = min_date, max_date
        if "Date" in att_df.columns:
            mask = (att_df["Date"].dt.date >= sd) & (att_df["Date"].dt.date <= ed)
            att_range = att_df[mask]
        else:
            att_range = att_df.copy()

        per_student = att_range.groupby("Name")["_present_flag_"].mean().reset_index().rename(columns={"_present_flag_":"att_mean"})
        per_student["pct"] = per_student["att_mean"] * 100
        def bucket(v):
            if v >= 90: return "Excellent (90%+)"
            if v >= 75: return "Good (75–90%)"
            if v >= 50: return "Average (50–75%)"
            return "Poor (<50%)"
        per_student["bucket"] = per_student["pct"].apply(bucket)
        counts = per_student["bucket"].value_counts().reindex(["Excellent (90%+)","Good (75–90%)","Average (50–75%)","Poor (<50%)"]).fillna(0).reset_index()
        counts.columns = ["Category","Count"]
        fig_pie = px.pie(counts, names="Category", values="Count", title="Students by attendance quality",
                         color_discrete_sequence=["#2ca02c","#17becf","#ff7f0e","#d62728"])
        st.plotly_chart(fig_pie, use_container_width=True)
        with st.expander("Explanation", expanded=auto_expand):
            st.write(
                "Students are bucketed by their personal attendance percentage over the selected date range. "
                "This quickly shows how many students fall into good vs at-risk categories."
            )
    else:
        st.info("Not enough attendance data for student-level buckets.")

    st.markdown("---")

    # Subject-level summary (avg and pass rate)
    st.subheader("Subject-level summary")
    if not marks_df.empty and "Subject" in marks_df.columns and "Marks" in marks_df.columns:
        subj = marks_df.groupby("Subject").agg(avg_score=("Marks","mean"), median_score=("Marks","median"), count=("Marks","count")).reset_index().sort_values("avg_score", ascending=False)
        fig_sub_avg = px.bar(subj, x="Subject", y="avg_score", title="Average score by subject", color="Subject", color_discrete_map=SUBJECT_COLORS)
        st.plotly_chart(fig_sub_avg, use_container_width=True)
        with st.expander("Explanation", expanded=auto_expand):
            st.write(
                "Average score per subject — helps identify strong and weak subjects at class level."
            )

        # pass rate
        subj_pass = marks_df.copy()
        subj_pass["pass"] = subj_pass["Marks"] >= pass_threshold
        pass_summary = subj_pass.groupby("Subject")["pass"].mean().reset_index().rename(columns={"pass":"pass_rate"}).sort_values("pass_rate", ascending=False)
        fig_pass = px.bar(pass_summary, x="Subject", y="pass_rate", title=f"Pass rate per subject (>= {pass_threshold})", color="Subject", color_discrete_map=SUBJECT_COLORS)
        fig_pass.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig_pass, use_container_width=True)
        with st.expander("Explanation", expanded=auto_expand):
            st.write("Pass rate shows the % of records meeting the pass threshold — a clear way to prioritize subjects for intervention.")
    else:
        st.info("Not enough marks data for subject summary.")

# ===== Tab 1: Student Dashboard =====
with tabs[1]:
    st.header("Student Dashboard")
    students = []
    if "Name" in marks_df.columns and not marks_df.empty:
        students = sorted(marks_df["Name"].dropna().unique().tolist())
    elif "Name" in att_df.columns and not att_df.empty:
        students = sorted(att_df["Name"].dropna().unique().tolist())

    if not students:
        st.info("No student names found in data.")
    else:
        student = st.selectbox("Select student", students)
        s_marks = marks_df[marks_df["Name"]==student] if not marks_df.empty else pd.DataFrame()
        s_att = att_df[att_df["Name"]==student] if not att_df.empty else pd.DataFrame()

        # profile card
        st.subheader(student)
        sid = (s_marks["ID"].iloc[0] if (not s_marks.empty and "ID" in s_marks.columns) else (s_att["ID"].iloc[0] if (not s_att.empty and "ID" in s_att.columns) else "N/A"))
        sroll = (s_marks["Roll"].iloc[0] if (not s_marks.empty and "Roll" in s_marks.columns) else (s_att["Roll"].iloc[0] if (not s_att.empty and "Roll" in s_att.columns) else "N/A"))
        colA, colB = st.columns([2,3])
        with colA:
            st.markdown(f"**ID:** {sid}  \n**Roll:** {sroll}")
        with colB:
            avg_mark = s_marks["Marks"].mean() if not s_marks.empty and "Marks" in s_marks.columns else np.nan
            att_rate = s_att["_present_flag_"].mean() if not s_att.empty and "_present_flag_" in s_att.columns else np.nan
            st.metric("Average mark", f"{avg_mark:.1f}" if not np.isnan(avg_mark) else "N/A")
            st.metric("Attendance rate", f"{att_rate*100:.1f}%" if not np.isnan(att_rate) else "N/A")

        st.markdown("---")

        # Radar / polar for subject strengths (simple)
        st.subheader("Subject strengths (radar)")
        if not s_marks.empty and "Subject" in s_marks.columns:
            subj_avg = s_marks.groupby("Subject")["Marks"].mean().reset_index()
            subj_avg = subj_avg.sort_values("Marks", ascending=False)
            # prepare for polar: need closed loop
            polar_df = subj_avg.copy()
            polar_df = polar_df.append(polar_df.iloc[0], ignore_index=True) if len(polar_df)>0 else polar_df
            if not polar_df.empty:
                fig_polar = go.Figure()
                fig_polar.add_trace(go.Scatterpolar(
                    r=polar_df["Marks"],
                    theta=polar_df["Subject"],
                    fill='toself',
                    name=student,
                    marker=dict(color=DISTINCT_PALETTE[0])
                ))
                fig_polar.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=False, title="Subject average radar")
                st.plotly_chart(fig_polar, use_container_width=True)
                with st.expander("Explanation", expanded=auto_expand):
                    st.write("Radar chart shows relative strength across subjects for this student. Larger area = stronger overall.")
        else:
            st.info("Not enough subject-level marks for radar.")

        st.markdown("---")

        # Marks trend by exam (line chart)
        st.subheader("Marks across exams")
        if not s_marks.empty and "ExamNumber" in s_marks.columns:
            exam_trend = s_marks.groupby(["ExamNumber"])["Marks"].mean().reset_index()
            fig_ex = px.line(exam_trend, x="ExamNumber", y="Marks", markers=True, title="Average marks by exam for this student")
            st.plotly_chart(fig_ex, use_container_width=True)
            with st.expander("Explanation", expanded=auto_expand):
                st.write("Shows the student's average marks per exam — look for improvement or decline.")
        else:
            st.info("No exam-level marks to plot.")

        st.markdown("---")

        # Attendance by month
        st.subheader("Attendance by month")
        if not s_att.empty and "Date" in s_att.columns:
            s_att2 = s_att.copy()
            s_att2["month"] = s_att2["Date"].dt.to_period("M").astype(str)
            monthly = s_att2.groupby("month")["_present_flag_"].mean().reset_index()
            fig_attm = px.bar(monthly, x="month", y="_present_flag_", title="Monthly attendance")
            fig_attm.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig_attm, use_container_width=True)
            with st.expander("Explanation", expanded=auto_expand):
                st.write("Monthly attendance gives a smoother, easier-to-read summary than day-by-day dots.")
        else:
            st.info("No attendance records for this student.")

# ===== Tab 2: Compare Students =====
with tabs[2]:
    st.header("Compare Students")
    candidate_names = sorted(set(marks_df["Name"].dropna().tolist())) if ("Name" in marks_df.columns and not marks_df.empty) else []
    selected = st.multiselect("Select students (up to 6)", options=candidate_names, max_selections=6)
    exam_choice = st.selectbox("Exam filter (All for averages)", options=["All"] + ([str(e) for e in sorted(marks_df["ExamNumber"].dropna().unique().tolist())] if ("ExamNumber" in marks_df.columns and not marks_df.empty) else []))

    if not selected or len(selected) < 2:
        st.info("Select two or more students to compare.")
    else:
        comp = marks_df[marks_df["Name"].isin(selected)].copy()
        if exam_choice != "All":
            comp = comp[comp["ExamNumber"].astype(str) == exam_choice]

        st.subheader("Subject-wise comparison")
        comp_avg = comp.groupby(["Name","Subject"])["Marks"].mean().reset_index()
        if not comp_avg.empty:
            fig_cmp = px.bar(comp_avg, x="Subject", y="Marks", color="Name", barmode="group")
            st.plotly_chart(fig_cmp, use_container_width=True)
            with st.expander("Explanation", expanded=auto_expand):
                st.write("Compares subject averages across selected students — helpful to see who is stronger in which subject.")
        else:
            st.info("No subject averages for the selected students/exam.")

        st.markdown("---")
        st.subheader("Attendance vs Marks (scatter)")
        if not att_df.empty and not marks_df.empty:
            marks_avg = marks_df.groupby("Name")["Marks"].mean().reset_index().rename(columns={"Marks":"avg_marks"})
            att_avg = att_df.groupby("Name")["_present_flag_"].mean().reset_index().rename(columns={"_present_flag_":"att_rate"})
            merged = pd.merge(marks_avg, att_avg, on="Name", how="inner")
            merged = merged[merged["Name"].isin(selected)]
            if not merged.empty:
                fig_sc = px.scatter(merged, x="att_rate", y="avg_marks", text="Name", size="avg_marks",
                                    labels={"att_rate":"Attendance %","avg_marks":"Avg Marks"}, color_discrete_sequence=[DISTINCT_PALETTE[1]])
                fig_sc.update_xaxes(tickformat=".0%")
                st.plotly_chart(fig_sc, use_container_width=True)
                with st.expander("Explanation", expanded=auto_expand):
                    st.write("Scatter of attendance vs average marks for selected students — bottom-left is both low attendance and low marks (intervention candidates).")
            else:
                st.info("No combined attendance+marks records for selected students.")
        else:
            st.info("Need both attendance and marks data to show correlation.")

# ===== Tab 3: Attendance =====
with tabs[3]:
    st.header("Attendance")

    if att_df.empty:
        st.info("No attendance data.")
    else:
        # date selection in-tab
        min_d = att_df["Date"].min().date() if ("Date" in att_df.columns and not att_df["Date"].isna().all()) else date.today()
        max_d = att_df["Date"].max().date() if ("Date" in att_df.columns and not att_df["Date"].isna().all()) else date.today()
        dr = st.date_input("Select date range", value=(min_d, max_d))
        try:
            sd, ed = (dr[0], dr[1]) if isinstance(dr, (list,tuple)) and len(dr)==2 else (dr, dr)
        except Exception:
            sd, ed = min_d, max_d

        if "Date" in att_df.columns:
            mask = (att_df["Date"].dt.date >= sd) & (att_df["Date"].dt.date <= ed)
            att_filtered = att_df[mask]
        else:
            att_filtered = att_df.copy()

        st.subheader("Daily class attendance %")
        if not att_filtered.empty and "_present_flag_" in att_filtered.columns:
            att_trend = att_filtered.groupby(att_filtered["Date"].dt.date)["_present_flag_"].mean().reset_index().rename(columns={"_present_flag_":"attendance_rate"})
            fig = px.line(att_trend, x="Date", y="attendance_rate", markers=True, title="Daily attendance %")
            fig.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)
            with st.expander("Explanation", expanded=auto_expand):
                st.write("Daily class attendance percentage over the selected date range.")
        else:
            st.info("No daily attendance records for the selected range.")

        st.markdown("---")
        st.subheader("Average attendance by weekday")
        try:
            att_filtered["weekday"] = att_filtered["Date"].dt.day_name()
            weekday_avg = att_filtered.groupby("weekday")["_present_flag_"].mean().reindex(["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]).reset_index().dropna()
            weekday_avg.columns = ["weekday","attendance_rate"]
            fig_w = px.bar(weekday_avg, x="weekday", y="attendance_rate", title="Average attendance by weekday")
            fig_w.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig_w, use_container_width=True)
            with st.expander("Explanation", expanded=auto_expand):
                st.write("Shows typical attendance per weekday — useful when scheduling important lessons or exams.")
        except Exception:
            st.info("Unable to compute weekday summary for your dataset.")

        st.markdown("---")
        st.subheader("Attendance leaderboard (top N)")
        n = st.number_input("Show top N students by attendance", min_value=5, max_value=200, value=20)
        if not att_filtered.empty:
            leader = att_filtered.groupby("Name")["_present_flag_"].mean().reset_index().sort_values("_present_flag_", ascending=False).reset_index(drop=True)
            leader["attendance_pct"] = leader["_present_flag_"].apply(lambda x: f"{x*100:.1f}%")
            st.dataframe(leader[["Name","attendance_pct"]].head(n).set_index("Name"))
            with st.expander("Explanation", expanded=auto_expand):
                st.write("Top students by attendance in the selected range — great for recognition or to model best practices.")
        else:
            st.info("No attendance summary available.")

# ===== Tab 4: Marks =====
with tabs[4]:
    st.header("Marks")

    st.subheader("Boxplot: marks by subject")
    if not marks_df.empty and "Marks" in marks_df.columns and "Subject" in marks_df.columns:
        box_df = marks_df.copy()
        if subject_filter:
            box_df = box_df[box_df["Subject"].isin(subject_filter)]
        fig_box = px.box(box_df, x="Subject", y="Marks", color="Subject", color_discrete_map=SUBJECT_COLORS, title="Distribution by subject (boxplot)")
        st.plotly_chart(fig_box, use_container_width=True)
        with st.expander("Explanation", expanded=auto_expand):
            st.write("Boxplots show median, quartiles and outliers per subject — useful for spotting variability and extremes.")
    else:
        st.info("Insufficient marks data to show boxplots.")

    st.markdown("---")
    st.subheader("Top & bottom performers (configurable)")
    if not marks_df.empty and "Name" in marks_df.columns:
        k = st.slider("How many top/bottom students", min_value=1, max_value=30, value=5)
        avg_by_name = marks_df.groupby("Name")["Marks"].mean().reset_index().dropna().sort_values("Marks", ascending=False)
        topk = avg_by_name.head(k)
        botk = avg_by_name.tail(k).sort_values("Marks")
        fig_top = px.bar(topk, x="Name", y="Marks", title=f"Top {k} students (avg marks)", color_discrete_sequence=[DISTINCT_PALETTE[2]])
        fig_bot = px.bar(botk, x="Name", y="Marks", title=f"Bottom {k} students (avg marks)", color_discrete_sequence=[DISTINCT_PALETTE[4]])
        st.plotly_chart(fig_top, use_container_width=True)
        st.plotly_chart(fig_bot, use_container_width=True)
        with st.expander("Explanation", expanded=auto_expand):
            st.write("Top and bottom performers help you recognize and plan interventions.")
    else:
        st.info("No marks data to compute performers.")

# ===== Tab 5: Insights =====
with tabs[5]:
    st.header("Insights (auto-generated)")
    bullets = []

    # class-level numeric insights
    if not marks_df.empty and "Marks" in marks_df.columns:
        overall_avg = marks_df["Marks"].mean()
        overall_median = marks_df["Marks"].median()
        bullets.append(f"Class average mark: {overall_avg:.1f}")
        bullets.append(f"Class median mark: {overall_median:.1f}")

        # subject with highest avg
        subj_avg = marks_df.groupby("Subject")["Marks"].mean().reset_index().dropna()
        if not subj_avg.empty:
            best_sub = subj_avg.sort_values("Marks", ascending=False).iloc[0]
            worst_sub = subj_avg.sort_values("Marks", ascending=True).iloc[0]
            bullets.append(f"Best subject (class avg): {best_sub['Subject']} ({best_sub['Marks']:.1f})")
            bullets.append(f"Weakest subject (class avg): {worst_sub['Subject']} ({worst_sub['Marks']:.1f})")

    if not att_df.empty and "_present_flag_" in att_df.columns:
        avg_att_overall = att_df["_present_flag_"].mean()
        bullets.append(f"Average attendance overall: {avg_att_overall*100:.1f}%")

    # present bullets
    if bullets:
        st.subheader("Quick numbers")
        for b in bullets:
            st.write("•", b)
    else:
        st.info("Not enough data to generate quick insights.")

    st.markdown("---")
    st.subheader("Who to prioritize (simple rules)")
    if not marks_df.empty and not att_df.empty and "Name" in marks_df.columns and "Name" in att_df.columns:
        marks_avg = marks_df.groupby("Name")["Marks"].mean().reset_index().rename(columns={"Marks":"avg_marks"})
        att_avg = att_df.groupby("Name")["_present_flag_"].mean().reset_index().rename(columns={"_present_flag_":"att_rate"})
        merged = pd.merge(marks_avg, att_avg, on="Name", how="inner")
        flagged = merged[(merged["avg_marks"] < flag_score_threshold) | (merged["att_rate"] < (flag_att_threshold_pct/100.0))]
        if not flagged.empty:
            flagged = flagged.sort_values(["att_rate","avg_marks"])
            flagged["attendance_pct"] = flagged["att_rate"].apply(lambda x: f"{x*100:.1f}%")
            st.dataframe(flagged[["Name","avg_marks","attendance_pct"]].rename(columns={"avg_marks":"Avg Marks"}).set_index("Name"))
            with st.expander("Explanation", expanded=auto_expand):
                st.write("Students above are flagged for either low average marks or low attendance — consider interventions.")
        else:
            st.success("No students flagged with current thresholds.")
    else:
        st.info("Need both marks and attendance to identify priority students.")

st.caption("Right iTech — professional, distinct colors, many simple visuals. Tell me any individual plot or color you'd like tweaked and I will update exactly that portion.")
