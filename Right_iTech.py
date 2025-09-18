# Right_iTech_complete.py
# Streamlit dashboard — full, polished, copy-paste ready.

import os
from datetime import date
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="Right iTech", layout="wide", initial_sidebar_state="expanded")
px.defaults.template = "plotly_white"

# -------------------------
# Distinct professional palette for subjects (reused across app)
# -------------------------
DISTINCT_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#d62728",
    "#17becf", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#393b79"
]
PRESENT_COLOR = "#2ca02c"
ABSENT_COLOR = "#d62728"
NEUTRAL = "#4a4a4a"

# -------------------------
# Sidebar: uploads + controls
# -------------------------
st.sidebar.header("Upload data & preferences")

att_upload = st.sidebar.file_uploader("Attendance CSV (optional)", type=["csv"])
marks_upload = st.sidebar.file_uploader("Marks CSV (optional)", type=["csv"])

# fallback paths (if files are already on the server)
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
# Title (theme-agnostic, no white box)
# -------------------------
st.markdown("<h1 style='text-align:center; color:#1f77b4; margin-bottom:4px;'>Right iTech</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#666; margin-top:0px; margin-bottom:12px;'>Professional, readable analytics for marks & attendance — interactive visuals.</p>", unsafe_allow_html=True)
st.write("---")

# -------------------------
# Defensive normalisation
# -------------------------
if att_df is None:
    att_df = pd.DataFrame()
if marks_df is None:
    marks_df = pd.DataFrame()

# strip column names
if not att_df.empty:
    att_df.columns = [c.strip() for c in att_df.columns]
if not marks_df.empty:
    marks_df.columns = [c.strip() for c in marks_df.columns]

# safe date parsing
if not att_df.empty and "Date" in att_df.columns:
    att_df["Date"] = pd.to_datetime(att_df["Date"], dayfirst=True, errors="coerce")

# unify present flag in attendance dataframe
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

# ensure marks numeric
if not marks_df.empty:
    if "Marks" in marks_df.columns:
        marks_df["Marks"] = pd.to_numeric(marks_df["Marks"], errors="coerce")
    else:
        marks_df["Marks"] = np.nan
    if "FullMarks" in marks_df.columns:
        marks_df["FullMarks"] = pd.to_numeric(marks_df["FullMarks"], errors="coerce")
else:
    # create columns so code later won't crash
    marks_df = pd.DataFrame(columns=["ID","Roll","Name","Subject","ExamNumber","Exam","ExamType","Marks","FullMarks"])

# subject color mapping
def assign_subject_colors(subjects):
    subs = sorted([s for s in subjects if pd.notna(s)])
    mapping = {}
    for i,s in enumerate(subs):
        mapping[s] = DISTINCT_PALETTE[i % len(DISTINCT_PALETTE)]
    return mapping

SUBJECT_COLORS = assign_subject_colors(marks_df["Subject"].unique()) if ("Subject" in marks_df.columns and not marks_df.empty) else {}

# -------------------------
# Small helpers
# -------------------------
def safe_count_gender(df):
    """Robustly compute boys/girls counts from either attendance or marks df."""
    if df is None or df.empty:
        return 0, 0
    # prefer unique students by ID if available, else by Name
    if "ID" in df.columns:
        uniq = df.drop_duplicates(subset=["ID"]).copy()
    elif "Name" in df.columns:
        uniq = df.drop_duplicates(subset=["Name"]).copy()
    else:
        return 0, 0
    if "Gender" not in uniq.columns:
        return 0, 0

    # normalize and match common encodings more flexibly
    g = uniq["Gender"].astype(str).str.strip().str.lower().fillna("")
    male_mask = g.str.match(r'^(m|male|boy|man)\b', na=False)
    female_mask = g.str.match(r'^(f|female|girl|woman)\b', na=False)

    boys = int(male_mask.sum())
    girls = int(female_mask.sum())

    # If still zero (odd encodings), try containing matches (looser)
    if boys == 0 and girls == 0:
        boys = int(g.str.contains(r'male|^m\b|boy', na=False).sum())
        girls = int(g.str.contains(r'female|^f\b|girl', na=False).sum())

    return boys, girls

def ensure_list(x):
    return x if isinstance(x, list) else [x]

# -------------------------
# Stop if no data at all
# -------------------------
if att_df.empty and marks_df.empty:
    st.warning("No data detected. Upload Attendance and Marks CSVs in the sidebar or place fallback files at /mnt/data/*.csv")
    st.stop()

# -------------------------
# Global filters in sidebar
# -------------------------
st.sidebar.header("Global filters")
if not att_df.empty and "Date" in att_df.columns and not att_df["Date"].isna().all():
    min_date = att_df["Date"].min().date()
    max_date = att_df["Date"].max().date()
else:
    min_date = date.today(); max_date = date.today()

date_range = st.sidebar.date_input("Attendance date range", value=(min_date, max_date))

subject_options = sorted(marks_df["Subject"].dropna().unique().tolist()) if (not marks_df.empty and "Subject" in marks_df.columns) else []
subject_filter = st.sidebar.multiselect("Filter subjects (global)", options=subject_options, default=subject_options)

exam_options = sorted(marks_df["ExamNumber"].dropna().unique().tolist()) if (not marks_df.empty and "ExamNumber" in marks_df.columns) else []
exam_filter = st.sidebar.multiselect("Filter exams (global)", options=exam_options, default=exam_options)

name_search = st.sidebar.text_input("Quick search student name (partial)")

# -------------------------
# Tabs
# -------------------------
tabs = st.tabs(["Class Overview","Student Dashboard","Compare Students","Attendance","Marks","Insights"])

# ===== Tab: Class Overview =====
with tabs[0]:
    st.header("Class Overview")

    # total students (prefer unique ID)
    total_students = 0
    if "ID" in marks_df.columns and not marks_df.empty:
        total_students = marks_df["ID"].nunique()
    elif "Name" in marks_df.columns and not marks_df.empty:
        total_students = marks_df["Name"].nunique()
    elif "ID" in att_df.columns and not att_df.empty:
        total_students = att_df["ID"].nunique()
    elif "Name" in att_df.columns and not att_df.empty:
        total_students = att_df["Name"].nunique()

    # compute boys/girls robustly - prefer attendance then marks
    boys, girls = 0, 0
    if not att_df.empty:
        boys, girls = safe_count_gender(att_df)
    if boys == 0 and girls == 0 and not marks_df.empty:
        boys, girls = safe_count_gender(marks_df)

    # avg attendance
    avg_att = att_df["_present_flag_"].mean() if not att_df.empty and "_present_flag_" in att_df.columns else np.nan

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total students", total_students)
    col2.metric("Boys", int(boys) if boys is not None else "N/A")
    col3.metric("Girls", int(girls) if girls is not None else "N/A")
    col4.metric("Avg attendance", f"{avg_att*100:.1f}%" if not np.isnan(avg_att) else "N/A")

    st.markdown("")  # small spacer

    # Keep ONLY Avg Present & Avg Absent capsules visually minimal
    cap1, cap2, _ = st.columns([1,1,6])
    avg_present = avg_att
    avg_absent = (1 - avg_att) if not np.isnan(avg_att) else np.nan
    cap1.markdown(
        f"<div style='padding:10px;border-radius:8px;text-align:center;background:transparent'><div style='color:#0b3d91;font-weight:700'>Avg Present</div><div style='font-size:18px'>{f'{avg_present*100:.1f}%' if not np.isnan(avg_present) else 'N/A'}</div></div>",
        unsafe_allow_html=True
    )
    cap2.markdown(
        f"<div style='padding:10px;border-radius:8px;text-align:center;background:transparent'><div style='color:#7f1f1f;font-weight:700'>Avg Absent</div><div style='font-size:18px'>{f'{avg_absent*100:.1f}%' if not np.isnan(avg_absent) else 'N/A'}</div></div>",
        unsafe_allow_html=True
    )

    st.markdown("---")

    # Attendance trend line (aggregate)
    st.subheader("Class attendance trend")
    if not att_df.empty and "Date" in att_df.columns and "_present_flag_" in att_df.columns:
        # apply global date_range filter
        try:
            sd, ed = (date_range[0], date_range[1]) if isinstance(date_range, (list,tuple)) and len(date_range)==2 else (date_range, date_range)
        except Exception:
            sd, ed = min_date, max_date
        mask = (att_df["Date"].dt.date >= sd) & (att_df["Date"].dt.date <= ed)
        att_range = att_df[mask]
        if not att_range.empty:
            att_over_time = att_range.groupby(att_range["Date"].dt.date)["_present_flag_"].mean().reset_index().rename(columns={"_present_flag_":"attendance_rate"})
            fig_att = px.line(att_over_time, x="Date", y="attendance_rate", markers=True, title="Daily class attendance %")
            fig_att.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig_att, use_container_width=True)
            with st.expander("Explanation", expanded=auto_expand):
                st.write("Daily attendance percentage (class average). Use this to spot dips and trends.")
        else:
            st.info("No attendance records in selected date range.")
    else:
        st.info("Attendance data not available to show trend.")

    st.markdown("---")

    # Subject averages bar chart
    st.subheader("Average marks by subject (filters applied)")
    if not marks_df.empty and "Marks" in marks_df.columns and "Subject" in marks_df.columns:
        dfm = marks_df.copy()
        if subject_filter:
            dfm = dfm[dfm["Subject"].isin(subject_filter)]
        if exam_filter:
            dfm = dfm[dfm["ExamNumber"].isin(exam_filter)]
        if name_search:
            dfm = dfm[dfm["Name"].str.contains(name_search, case=False, na=False)]
        subj_avg = dfm.groupby("Subject")["Marks"].mean().reset_index()
        if not subj_avg.empty:
            fig_subj = px.bar(subj_avg.sort_values("Marks", ascending=False), x="Subject", y="Marks", color="Subject", color_discrete_map=SUBJECT_COLORS, title="Average marks by subject (filters applied)")
            st.plotly_chart(fig_subj, use_container_width=True)
        else:
            st.info("No marks match current global filters.")
    else:
        st.info("Not enough marks data to show subject averages.")

    st.markdown("---")

    # Subject threshold stacked chart (subject-wise counts or percentages above/below pass_threshold)
    st.subheader("For each subject: students ≥ threshold vs < threshold")
    if not marks_df.empty and "Marks" in marks_df.columns and "Subject" in marks_df.columns and "Name" in marks_df.columns:
        dfm2 = marks_df.copy()
        # apply global filters
        if subject_filter:
            dfm2 = dfm2[dfm2["Subject"].isin(subject_filter)]
        if exam_filter:
            dfm2 = dfm2[dfm2["ExamNumber"].isin(exam_filter)]
        if name_search:
            dfm2 = dfm2[dfm2["Name"].str.contains(name_search, case=False, na=False)]

        # compute per-student-per-subject average (so repeated exam marks don't double count students)
        per_student_subject = dfm2.groupby(["Subject","Name"])["Marks"].mean().reset_index()

        # compute counts per subject
        if per_student_subject.empty:
            st.info("No marks match current global filters for threshold calculation.")
        else:
            # Use pass_threshold from sidebar by default
            threshold = int(pass_threshold)
            # UI: allow override locally if desired (keeps compatibility with your sidebar)
            local_override = st.checkbox("Override sidebar threshold for this chart", value=False)
            if local_override:
                threshold = st.slider("Set subject threshold (local)", min_value=0, max_value=100, value=threshold)

            # choose Y axis mode
            y_mode = st.radio("Y-axis mode", options=["Count", "Percentage"], index=0, horizontal=True)

            subj_counts = per_student_subject.groupby("Subject").apply(
                lambda g: pd.Series({
                    "n_total": g["Name"].nunique(),
                    "n_above": int((g["Marks"] >= threshold).sum())
                })
            ).reset_index()
            subj_counts["n_below"] = subj_counts["n_total"] - subj_counts["n_above"]
            # For percentage mode, compute fractions
            subj_counts["pct_above"] = subj_counts.apply(lambda r: (r["n_above"]/r["n_total"]) if r["n_total"]>0 else 0.0, axis=1)
            subj_counts["pct_below"] = subj_counts.apply(lambda r: (r["n_below"]/r["n_total"]) if r["n_total"]>0 else 0.0, axis=1)

            # Build long DataFrame for stacked bar plot
            long_rows = []
            for _, row in subj_counts.iterrows():
                if y_mode == "Count":
                    long_rows.append({"Subject": row["Subject"], "Category": f"≥ {threshold}", "Value": int(row["n_above"]), "Hover": f"{int(row['n_above'])} students ({row['pct_above']*100:.1f}%)"})
                    long_rows.append({"Subject": row["Subject"], "Category": f"< {threshold}", "Value": int(row["n_below"]), "Hover": f"{int(row['n_below'])} students ({row['pct_below']*100:.1f}%)"})
                else:
                    long_rows.append({"Subject": row["Subject"], "Category": f"≥ {threshold}", "Value": row["pct_above"], "Hover": f"{row['pct_above']*100:.1f}% students"})
                    long_rows.append({"Subject": row["Subject"], "Category": f"< {threshold}", "Value": row["pct_below"], "Hover": f"{row['pct_below']*100:.1f}% students"})

            long_df = pd.DataFrame(long_rows)

            # Color mapping
            color_map = {f"≥ {threshold}": DISTINCT_PALETTE[0], f"< {threshold}": DISTINCT_PALETTE[4]}

            if y_mode == "Count":
                fig_thresh_subj = px.bar(long_df, x="Subject", y="Value", color="Category", color_discrete_map=color_map, title=f"Subjects: count of students ≥ {threshold} vs < {threshold}", hover_data=["Hover"])
                fig_thresh_subj.update_yaxes(title_text="Number of students")
            else:
                fig_thresh_subj = px.bar(long_df, x="Subject", y="Value", color="Category", color_discrete_map=color_map, title=f"Subjects: % of students ≥ {threshold} vs < {threshold}", hover_data=["Hover"])
                fig_thresh_subj.update_yaxes(title_text="Share of students", tickformat=".0%")

            fig_thresh_subj.update_layout(barmode="stack", xaxis={'categoryorder':'total descending'})
            st.plotly_chart(fig_thresh_subj, use_container_width=True)
            with st.expander("Explanation", expanded=auto_expand):
                st.write("This chart shows for every subject how many students (or what percentage) have a per-subject average >= the selected threshold. The counts are computed per student-per-subject (so multiple exam entries for the same student/subject are averaged first). Use this to spot subjects where a majority of students are under-performing.")
    else:
        st.info("Not enough marks data to compute the subject-threshold breakdown.")

# ===== Tab: Student Dashboard =====
with tabs[1]:
    st.header("Student Dashboard")
    # list students from marks or attendance
    student_list = []
    if "Name" in marks_df.columns and not marks_df.empty:
        student_list = sorted(marks_df["Name"].dropna().unique().tolist())
    elif "Name" in att_df.columns and not att_df.empty:
        student_list = sorted(att_df["Name"].dropna().unique().tolist())

    if not student_list:
        st.info("No student names found in the dataset.")
    else:
        student = st.selectbox("Select student", student_list)
        s_marks = marks_df[marks_df["Name"]==student] if not marks_df.empty else pd.DataFrame()
        s_att = att_df[att_df["Name"]==student] if not att_df.empty else pd.DataFrame()

        # profile & quick metrics
        st.subheader(student)
        sid = s_marks["ID"].iloc[0] if (not s_marks.empty and "ID" in s_marks.columns) else (s_att["ID"].iloc[0] if (not s_att.empty and "ID" in s_att.columns) else "N/A")
        sroll = s_marks["Roll"].iloc[0] if (not s_marks.empty and "Roll" in s_marks.columns) else (s_att["Roll"].iloc[0] if (not s_att.empty and "Roll" in s_att.columns) else "N/A")
        gender = None
        if not s_att.empty and "Gender" in s_att.columns:
            gender = s_att["Gender"].dropna().iloc[0] if s_att["Gender"].dropna().size>0 else None
        elif not s_marks.empty and "Gender" in s_marks.columns:
            gender = s_marks["Gender"].dropna().iloc[0] if s_marks["Gender"].dropna().size>0 else None

        c1, c2 = st.columns([2,3])
        with c1:
            st.markdown(f"**ID:** {sid}  \n**Roll:** {sroll}  \n**Gender:** {gender if gender is not None else 'N/A'}")
        with c2:
            avg_mark = s_marks["Marks"].mean() if not s_marks.empty and "Marks" in s_marks.columns else np.nan
            att_rate = s_att["_present_flag_"].mean() if not s_att.empty and "_present_flag_" in s_att.columns else np.nan
            st.metric("Average mark", f"{avg_mark:.1f}" if not np.isnan(avg_mark) else "N/A")
            st.metric("Attendance rate", f"{att_rate*100:.1f}%" if not np.isnan(att_rate) else "N/A")

        st.markdown("---")
        # Subject bar for student
        st.subheader("Subject performance (student)")
        if not s_marks.empty and "Subject" in s_marks.columns:
            subj_avg = s_marks.groupby("Subject")["Marks"].mean().reset_index().sort_values("Marks", ascending=False)
            fig_sub = px.bar(subj_avg, x="Subject", y="Marks", color="Subject", color_discrete_map=SUBJECT_COLORS, title="Student avg by subject")
            st.plotly_chart(fig_sub, use_container_width=True)
            with st.expander("Explanation", expanded=auto_expand):
                st.write("Subject wise averages for the selected student to quickly spot strengths and weaknesses.")
        else:
            st.info("No subject marks for this student.")

        st.markdown("---")
        # Marks across exams for the student (subject lines)
        st.subheader("Marks across exams (subject lines)")
        if not s_marks.empty and "ExamNumber" in s_marks.columns and "Subject" in s_marks.columns:
            s_trend = s_marks.groupby(["ExamNumber","Subject"])["Marks"].mean().reset_index()
            subj_s = sorted(s_trend["Subject"].unique().tolist())
            chosen_subs = st.multiselect("Subjects (student) to show", options=subj_s, default=subj_s)
            plot_s = s_trend[s_trend["Subject"].isin(chosen_subs)]
            if not plot_s.empty:
                fig = px.line(plot_s, x="ExamNumber", y="Marks", color="Subject", markers=True, color_discrete_map=SUBJECT_COLORS, title="Student marks by exam per subject")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No subject-exam points to plot after filtering.")
        else:
            st.info("Not enough exam-level marks for this student.")

        st.markdown("---")
        # Attendance monthly for student
        st.subheader("Attendance (monthly)")
        if not s_att.empty and "Date" in s_att.columns:
            s_att2 = s_att.copy()
            s_att2["month"] = s_att2["Date"].dt.to_period("M").astype(str)
            monthly = s_att2.groupby("month")["_present_flag_"].mean().reset_index()
            figm = px.bar(monthly, x="month", y="_present_flag_", title="Monthly attendance (student)")
            figm.update_yaxes(tickformat=".0%")
            st.plotly_chart(figm, use_container_width=True)
        else:
            st.info("No attendance records for this student.")

# ===== Tab: Compare Students =====
with tabs[2]:
    st.header("Compare Students")
    candidate_names = sorted(set(marks_df["Name"].dropna().tolist())) if ("Name" in marks_df.columns and not marks_df.empty) else []
    selected = st.multiselect("Select students (2-6)", options=candidate_names, max_selections=6)
    exam_choice = st.selectbox("Exam filter (All for averages)", options=["All"] + ([str(e) for e in sorted(marks_df["ExamNumber"].dropna().unique().tolist())] if ("ExamNumber" in marks_df.columns and not marks_df.empty) else []))

    if not selected or len(selected) < 2:
        st.info("Select two or more students to compare.")
    else:
        comp = marks_df[marks_df["Name"].isin(selected)].copy()
        if exam_choice != "All":
            comp = comp[comp["ExamNumber"].astype(str) == exam_choice]

        st.subheader("Subject-wise averages (selected students)")
        comp_avg = comp.groupby(["Name","Subject"])["Marks"].mean().reset_index()
        if not comp_avg.empty:
            fig_cmp = px.bar(comp_avg, x="Subject", y="Marks", color="Name", barmode="group", title="Subject-wise averages")
            st.plotly_chart(fig_cmp, use_container_width=True)

        st.markdown("---")
        st.subheader("Attendance vs marks (selected students)")
        if not att_df.empty and not marks_df.empty:
            marks_avg_all = marks_df.groupby("Name")["Marks"].mean().reset_index().rename(columns={"Marks":"avg_marks"})
            att_avg_all = att_df.groupby("Name")["_present_flag_"].mean().reset_index().rename(columns={"_present_flag_":"att_rate"})
            merged = pd.merge(marks_avg_all, att_avg_all, on="Name", how="inner")
            merged_sel = merged[merged["Name"].isin(selected)]
            if not merged_sel.empty:
                fig_sc = px.scatter(merged_sel, x="att_rate", y="avg_marks", color="Name", size="avg_marks", hover_name="Name")
                fig_sc.update_xaxes(tickformat=".0%")
                st.plotly_chart(fig_sc, use_container_width=True)
            else:
                st.info("No combined attendance+marks records for the selected students.")
        else:
            st.info("Need both marks and attendance to show correlation.")

# ===== Tab: Attendance =====
with tabs[3]:
    st.header("Attendance")

    if att_df.empty:
        st.info("No attendance data.")
    else:
        # date range inside tab
        min_d = att_df["Date"].min().date() if ("Date" in att_df.columns and not att_df["Date"].isna().all()) else date.today()
        max_d = att_df["Date"].max().date() if ("Date" in att_df.columns and not att_df["Date"].isna().all()) else date.today()
        dr = st.date_input("Select date range", value=(min_d, max_d))
        try:
            sd, ed = (dr[0], dr[1]) if isinstance(dr, (list,tuple)) and len(dr)==2 else (dr, dr)
        except Exception:
            sd, ed = min_d, max_d

        if "Date" in att_df.columns:
            mask = (att_df["Date"].dt.date >= sd) & (att_df["Date"].dt.date <= ed)
            att_filtered = att_df[mask].copy()
        else:
            att_filtered = att_df.copy()

        # daily aggregate line
        st.subheader("Daily class attendance %")
        if not att_filtered.empty and "_present_flag_" in att_filtered.columns:
            daily = att_filtered.groupby(att_filtered["Date"].dt.date)["_present_flag_"].mean().reset_index().rename(columns={"_present_flag_":"attendance_rate"})
            fig_d = px.line(daily, x="Date", y="attendance_rate", markers=True, title="Daily attendance %")
            fig_d.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig_d, use_container_width=True)
        else:
            st.info("No attendance records in this range.")

        st.markdown("---")
        # daily attendance heatmap per student (select students)
        st.subheader("Daily attendance per student (heatmap)")
        students_all = sorted(att_filtered["Name"].dropna().unique().tolist()) if "Name" in att_filtered.columns else []
        chosen = st.multiselect("Select students (empty = top 20 by records)", options=students_all, default=students_all[:20] if students_all else [])
        if chosen:
            heat_df = att_filtered[att_filtered["Name"].isin(chosen)].copy()
        else:
            heat_df = att_filtered.copy()

        if not heat_df.empty and "Date" in heat_df.columns and "Name" in heat_df.columns:
            # pivot to Name x Date
            pivot = heat_df.pivot_table(index="Name", columns=heat_df["Date"].dt.date, values="_present_flag_", aggfunc="mean")
            pivot = pivot.sort_index()
            if pivot.shape[1] == 0:
                st.info("No date columns to show.")
            else:
                # limit columns for readability (last 60)
                max_dates = 60
                if pivot.shape[1] > max_dates:
                    pivot = pivot.iloc[:, -max_dates:]
                z = pivot.fillna(np.nan).values
                x = [str(d) for d in pivot.columns]
                y = pivot.index.tolist()
                # custom discrete colorscale mapping 0->red, 1->green
                colorscale = [[0.0, ABSENT_COLOR], [1.0, PRESENT_COLOR]]
                fig_h = go.Figure(data=go.Heatmap(z=z, x=x, y=y, colorscale=[[0, ABSENT_COLOR], [0.5, "#ffffff"], [1, PRESENT_COLOR]], zmin=0, zmax=1, colorbar=dict(title="Present (1) / Absent (0)")))
                fig_h.update_layout(height=max(300, 24*len(y)))
                st.plotly_chart(fig_h, use_container_width=True)
                with st.expander("Explanation", expanded=auto_expand):
                    st.write("Heatmap shows daily present (green) or absent (red) per selected student. Hover any cell for details.")
        else:
            st.info("Not enough data to create per-student heatmap.")

# ===== Tab: Marks =====
with tabs[4]:
    st.header("Marks")

    # subject-wise trend across exams (subject lines)
    st.subheader("Subject trends across exams (subject lines)")
    if not marks_df.empty and "ExamNumber" in marks_df.columns and "Subject" in marks_df.columns and "Marks" in marks_df.columns:
        exam_inside = st.multiselect("Filter exams to include (leave empty = all)", options=sorted(marks_df["ExamNumber"].dropna().unique().tolist()), default=sorted(marks_df["ExamNumber"].dropna().unique().tolist()))
        subj_inside = st.multiselect("Subjects to include (leave empty = all)", options=sorted(marks_df["Subject"].dropna().unique().tolist()), default=sorted(marks_df["Subject"].dropna().unique().tolist()))
        dfm = marks_df.copy()
        if exam_inside:
            dfm = dfm[dfm["ExamNumber"].isin(exam_inside)]
        if subj_inside:
            dfm = dfm[dfm["Subject"].isin(subj_inside)]
        trend = dfm.groupby(["ExamNumber","Subject"])["Marks"].mean().reset_index()
        if not trend.empty:
            fig_tr = px.line(trend, x="ExamNumber", y="Marks", color="Subject", markers=True, color_discrete_map=SUBJECT_COLORS, title="Subject-wise average across exams")
            st.plotly_chart(fig_tr, use_container_width=True)
            with st.expander("Explanation", expanded=auto_expand):
                st.write("Each line is a subject showing the class average across exams. Use filters to focus on specific subjects or exams.")
        else:
            st.info("No trend data after filters.")
    else:
        st.info("Marks data not sufficient to draw subject trends.")

    st.markdown("---")
    # distribution and top/bottom
    st.subheader("Distribution & performers")
    if not marks_df.empty and "Marks" in marks_df.columns:
        # histogram
        fig_hist = px.histogram(marks_df, x="Marks", nbins=25, color_discrete_sequence=[DISTINCT_PALETTE[0]])
        st.plotly_chart(fig_hist, use_container_width=True)
        with st.expander("Explanation", expanded=auto_expand):
            st.write("Histogram of all recorded marks (respecting global subject/exam filters in side panel).")

        # top / bottom performers
        k = st.slider("How many top/bottom performers to show", min_value=1, max_value=30, value=5)
        avg_by_name = marks_df.groupby("Name")["Marks"].mean().reset_index().dropna().sort_values("Marks", ascending=False)
        topk = avg_by_name.head(k)
        botk = avg_by_name.tail(k).sort_values("Marks")
        fig_top = px.bar(topk, x="Name", y="Marks", title=f"Top {k} (avg marks)", color_discrete_sequence=[DISTINCT_PALETTE[2]])
        fig_bot = px.bar(botk, x="Name", y="Marks", title=f"Bottom {k} (avg marks)", color_discrete_sequence=[DISTINCT_PALETTE[4]])
        st.plotly_chart(fig_top, use_container_width=True)
        st.plotly_chart(fig_bot, use_container_width=True)
    else:
        st.info("Insufficient marks data for distribution or performers.")

# ===== Tab: Insights =====
with tabs[5]:
    st.header("Insights")

    bullets = []
    if not marks_df.empty and "Marks" in marks_df.columns:
        overall_avg = marks_df["Marks"].mean()
        overall_median = marks_df["Marks"].median()
        bullets.append(f"Class average mark: {overall_avg:.1f}")
        bullets.append(f"Class median mark: {overall_median:.1f}")
        subj_avg = marks_df.groupby("Subject")["Marks"].mean().reset_index().dropna()
        if not subj_avg.empty:
            best = subj_avg.sort_values("Marks", ascending=False).iloc[0]
            worst = subj_avg.sort_values("Marks", ascending=True).iloc[0]
            bullets.append(f"Best subject (class avg): {best['Subject']} ({best['Marks']:.1f})")
            bullets.append(f"Weakest subject (class avg): {worst['Subject']} ({worst['Marks']:.1f})")
    if not att_df.empty and "_present_flag_" in att_df.columns:
        bullets.append(f"Average attendance overall: {att_df['_present_flag_'].mean()*100:.1f}%")

    if bullets:
        st.subheader("Quick numbers")
        for b in bullets:
            st.write("•", b)
    else:
        st.info("Not enough data to auto-generate insights.")

    st.markdown("---")
    st.subheader("Who to prioritize (low marks or low attendance)")
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
                st.write("Students flagged either for low average marks or low attendance; consider follow-ups/interventions.")
        else:
            st.success("No students currently flagged.")
    else:
        st.info("Need both marks and attendance data to compute flags.")

st.caption("Right iTech — complete. Tell me one small change you'd like next (color tweak, plot type swap, or small UI adjustment) and I will update that single item only.")
