# right_itech_app_optimized.py
# Right iTech — Optimized Student Insights App
# Full-featured, improved UX and visualizations per user's request.

import os
import io
import tempfile
from datetime import date

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# For PDF export
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4

# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="Right iTech", layout="wide", initial_sidebar_state="expanded")

# -------------------------
# Constants / Palette
# -------------------------
PALETTE = px.colors.qualitative.Set2
ATT_PRESENT_COLOR = "#2ca02c"  # green
ATT_ABSENT_COLOR = "#d62728"   # red

# -------------------------
# Helpers
# -------------------------
@st.cache_data
def load_csv(uploaded_file, fallback_path):
    """Read uploaded file or fallback if present. Return empty DataFrame if nothing available."""
    try:
        if uploaded_file is not None:
            return pd.read_csv(uploaded_file)
        if fallback_path and os.path.exists(fallback_path):
            return pd.read_csv(fallback_path)
    except Exception:
        return pd.DataFrame()
    return pd.DataFrame()

def safe_to_numeric(df, col):
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    else:
        df[col] = np.nan
    return df

def parse_dates(df, col="Date"):
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], dayfirst=True, errors="coerce")
    return df

def standardize_present_flag(df, status_col="Status"):
    if status_col in df.columns:
        df["_present_flag_"] = df[status_col].astype(str).str.upper().map({
            "P": 1, "PRESENT": 1, "1": 1,
            "A": 0, "ABSENT": 0, "0": 0
        })
        # fallback to first char
        df["_present_flag_"] = df["_present_flag_"].fillna(df[status_col].astype(str).str[0].map({"P":1,"A":0}))
    else:
        df["_present_flag_"] = np.nan
    return df

def assign_subject_colors(subjects):
    subjects = sorted([s for s in subjects if pd.notna(s)])
    mapping = {}
    for i, s in enumerate(subjects):
        mapping[s] = PALETTE[i % len(PALETTE)]
    return mapping

def fig_to_png_bytes(fig, scale=2):
    """Return PNG bytes for a Plotly figure (requires kaleido or plotly image export)."""
    try:
        return fig.to_image(format="png", scale=scale)
    except Exception:
        # If fig.to_image not available, return None
        return None

# -------------------------
# Load data (uploads + fallback)
# -------------------------
st.sidebar.header("Upload datasets (optional)")
uploaded_marks = st.sidebar.file_uploader("Marks CSV (cleanest_marks.csv)", type=["csv"])
uploaded_att = st.sidebar.file_uploader("Attendance CSV (combined_attendance.csv)", type=["csv"])

FALLBACK_MARKS = "/mnt/data/cleanest_marks.csv"
FALLBACK_ATT = "/mnt/data/combined_attendance.csv"

marks_df = load_csv(uploaded_marks, FALLBACK_MARKS)
att_df = load_csv(uploaded_att, FALLBACK_ATT)

if marks_df.empty and att_df.empty:
    st.sidebar.warning("No data loaded. Upload CSVs or place fallback files on server.")
# continue — UI shows messages per tab if data missing

# -------------------------
# Normalize & clean
# -------------------------
if not marks_df.empty:
    marks_df.columns = [c.strip() for c in marks_df.columns]
else:
    marks_df = pd.DataFrame()

if not att_df.empty:
    att_df.columns = [c.strip() for c in att_df.columns]
else:
    att_df = pd.DataFrame()

# Standardize marks columns
marks_df = safe_to_numeric(marks_df, "Marks")
marks_df = safe_to_numeric(marks_df, "FullMarks")
if "WasAbsent" in marks_df.columns:
    marks_df["WasAbsent"] = marks_df["WasAbsent"].astype(str)
else:
    marks_df["WasAbsent"] = "False"

# Attendance parsing
if not att_df.empty:
    att_df = parse_dates(att_df, "Date")
    att_df = standardize_present_flag(att_df, "Status")

# Build attendance summary if possible
if not att_df.empty and "ID" in att_df.columns:
    att_summary = att_df.groupby(["ID","Roll","Name"]).agg(
        total_days=("Date","nunique"),
        present_count=("_present_flag_","sum")
    ).reset_index()
    att_summary["attendance_rate"] = att_summary["present_count"] / att_summary["total_days"]
else:
    att_summary = pd.DataFrame()

# Subject summary
if not marks_df.empty and "Subject" in marks_df.columns:
    subj_summary = marks_df.groupby("Subject").agg(
        exams=("ExamNumber","nunique"),
        avg_score=("Marks","mean"),
        entries=("Marks","count")
    ).reset_index()
    SUBJECT_COLORS = assign_subject_colors(subj_summary["Subject"].unique())
else:
    subj_summary = pd.DataFrame()
    SUBJECT_COLORS = {}

# Student-level marks summary
def student_marks_summary(df):
    if df.empty:
        return pd.DataFrame()
    s = df.groupby(["ID","Roll","Name"]).agg(
        avg_score=("Marks","mean"),
        exams_taken=("ExamNumber","nunique"),
        records_count=("Marks","count"),
        absent_count=("WasAbsent", lambda x: x.astype(str).str.lower().isin(["true","1","yes"]).sum())
    ).reset_index()
    return s

student_mark_summary = student_marks_summary(marks_df)

# -------------------------
# Sidebar controls
# -------------------------
st.sidebar.header("Controls")
min_score_threshold = st.sidebar.number_input("Flag if avg score < (threshold)", min_value=0, max_value=100, value=40)
min_att_threshold_pct = st.sidebar.slider("Flag if attendance < (%)", min_value=0, max_value=100, value=75)
auto_expand_explanations = st.sidebar.checkbox("Auto-expand explanations", value=False)

# -------------------------
# App header (clean)
# -------------------------
st.title("📊 Right iTech — Student Insights")
st.markdown("A clean, teacher-friendly dashboard for marks and attendance. Use the sidebar to upload datasets and set thresholds.")
st.write("---")

# -------------------------
# Tabs
# -------------------------
tabs = st.tabs(["Class overview","Student Profile","Compare students","Attendance explorer","Insights & Export"])

# -------------------------
# Tab: Class overview
# -------------------------
with tabs[0]:
    st.header("Class overview")

    # Basic class counts
    st.subheader("Class snapshot")
    col1, col2, col3, col4 = st.columns(4)
    # total students (from marks or attendance)
    unique_students = set()
    if "ID" in marks_df.columns:
        unique_students.update(marks_df["ID"].dropna().unique().tolist())
    if "ID" in att_df.columns:
        unique_students.update(att_df["ID"].dropna().unique().tolist())
    total_students = len(unique_students) if unique_students else (marks_df["ID"].nunique() if "ID" in marks_df.columns else ("N/A" if marks_df.empty else marks_df["Name"].nunique()))
    # gender breakdown if available
    gender_counts = None
    if "Gender" in att_df.columns:
        gender_counts = att_df.drop_duplicates(subset=["ID"])[["ID","Gender"]].groupby("Gender").size().to_dict()
    # average attendance
    avg_att = att_summary["attendance_rate"].mean() if not att_summary.empty else np.nan
    avg_absent_pct = 1 - avg_att if not np.isnan(avg_att) else np.nan

    col1.metric("Total students", total_students)
    if gender_counts:
        boys = gender_counts.get("Boys", 0) + gender_counts.get("Male", 0)
        girls = gender_counts.get("Girls", 0) + gender_counts.get("Female", 0)
        col2.metric("Boys", boys)
        col3.metric("Girls", girls)
    else:
        col2.metric("Boys", "N/A")
        col3.metric("Girls", "N/A")
    col4.metric("Avg attendance", f"{avg_att*100:.1f}%" if not np.isnan(avg_att) else "N/A")

    st.markdown("---")

    # Subject ranking (replacement for correlation)
    st.subheader("Subject ranking — where the class stands")
    if not subj_summary.empty:
        subj_sorted = subj_summary.sort_values("avg_score", ascending=False)
        fig = px.bar(subj_sorted, x="avg_score", y="Subject", orientation="h", color="Subject", color_discrete_map=SUBJECT_COLORS)
        fig.update_layout(xaxis_title="Average score", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("Explanation", expanded=auto_expand_explanations):
            # Data-aware explanation
            best = subj_sorted.iloc[0]["Subject"]
            worst = subj_sorted.iloc[-1]["Subject"]
            st.write(
                f"The bar chart ranks subjects by class average score. In this dataset, the highest-average subject is **{best}** "
                f"and the lowest-average subject is **{worst}**. This helps prioritize where the class may need support."
            )
    else:
        st.info("Not enough subject data to show rankings.")

    st.markdown("---")

    # More class stats and visuals
    st.subheader("Class-level statistics & distributions")
    # Score distribution
    if not marks_df.empty:
        fig = px.histogram(marks_df, x="Marks", nbins=25, color="Subject" if "Subject" in marks_df.columns else None, color_discrete_map=SUBJECT_COLORS or None)
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("Explanation", expanded=auto_expand_explanations):
            st.write(
                "Distribution of all recorded marks. Use filters and hover to inspect counts. "
                "A concentration on lower scores indicates possible broad learning gaps."
            )
    else:
        st.info("No marks data to visualize distribution.")

    # Attendance summary for class
    st.subheader("Attendance summary")
    if not att_summary.empty:
        # top present and top absent
        top_present = att_summary.sort_values("attendance_rate", ascending=False).head(5)[["Name","attendance_rate"]]
        top_absent = att_summary.sort_values("attendance_rate", ascending=True).head(5)[["Name","attendance_rate"]]
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Top attendance (5)**")
            st.table(top_present.style.format({"attendance_rate":"{:.1%}"}).set_index("Name"))
        with c2:
            st.markdown("**Lowest attendance (5)**")
            st.table(top_absent.style.format({"attendance_rate":"{:.1%}"}).set_index("Name"))

        # overall present/absent pie (derived)
        overall_present = att_df["_present_flag_"].sum()
        overall_total = att_df.shape[0]
        overall_absent = overall_total - overall_present
        pie_df = pd.DataFrame({"status":["Present","Absent"], "count":[int(overall_present), int(overall_absent)]})
        fig = px.pie(pie_df, values="count", names="status", color="status", color_discrete_map={"Present":ATT_PRESENT_COLOR,"Absent":ATT_ABSENT_COLOR})
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("Explanation", expanded=auto_expand_explanations):
            st.write(
                "Pie chart shows the overall share of present vs absent records in the dataset. "
                "If absent share is high, check school events, holidays, or data issues."
            )
    else:
        st.info("No attendance data to compute class-level stats.")

# -------------------------
# Tab: Student Profile (renamed)
# -------------------------
with tabs[1]:
    st.header("Student Profile")
    # choose student from marks or attendance
    candidates = []
    if not marks_df.empty and "Name" in marks_df.columns:
        candidates = sorted(marks_df["Name"].dropna().unique().tolist())
    elif not att_df.empty and "Name" in att_df.columns:
        candidates = sorted(att_df["Name"].dropna().unique().tolist())

    if not candidates:
        st.info("No student names found. Upload data with a Name column.")
    else:
        student = st.selectbox("Select student", candidates)

        # Data slices
        s_marks = marks_df[marks_df["Name"] == student] if not marks_df.empty else pd.DataFrame()
        s_att = att_df[att_df["Name"] == student] if not att_df.empty else pd.DataFrame()

        # Profile card
        if not s_marks.empty:
            sid = s_marks["ID"].iloc[0] if "ID" in s_marks.columns else "N/A"
            sroll = s_marks["Roll"].iloc[0] if "Roll" in s_marks.columns else "N/A"
        elif not s_att.empty:
            sid = s_att["ID"].iloc[0] if "ID" in s_att.columns else "N/A"
            sroll = s_att["Roll"].iloc[0] if "Roll" in s_att.columns else "N/A"
        else:
            sid = "N/A"
            sroll = "N/A"

        # Display profile nicely
        st.markdown(
            f"""
            <div style="display:flex;gap:20px;align-items:center">
              <div style="background:#f0f2f6;padding:16px;border-radius:10px;min-width:240px">
                <div style="font-weight:700;font-size:18px">{student}</div>
                <div style="color:#555;margin-top:6px">ID: {sid} • Roll: {sroll}</div>
                <div style="margin-top:8px">
            """, unsafe_allow_html=True)
        # small metrics
        avg_mark_text = f"{s_marks['Marks'].mean():.2f}" if not s_marks.empty and s_marks['Marks'].notna().any() else "N/A"
        att_rate_text = f"{int(s_att['_present_flag_'].sum())}/{len(s_att)} ({(s_att['_present_flag_'].mean()*100):.1f}%)" if (not s_att.empty and "_present_flag_" in s_att.columns and len(s_att)>0) else "N/A"
        st.markdown(f"<div style='margin-top:8px'><b>Avg mark:</b> {avg_mark_text}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='margin-top:4px'><b>Attendance:</b> {att_rate_text}</div>", unsafe_allow_html=True)
        st.markdown("</div></div>", unsafe_allow_html=True)

        st.write("---")

        # Visuals for student: Subject breakdown (bar), Exam-wise grouped bar, Attendance monthly
        st.subheader("Marks: subject breakdown")
        if not s_marks.empty and "Subject" in s_marks.columns:
            subj_avg = s_marks.groupby("Subject")["Marks"].mean().reset_index()
            fig = px.bar(subj_avg, x="Subject", y="Marks", color="Subject", color_discrete_map=SUBJECT_COLORS)
            st.plotly_chart(fig, use_container_width=True)
            with st.expander("What this shows", expanded=auto_expand_explanations):
                st.write(
                    "This shows the student's average per subject. Use it to quickly spot strengths and weaknesses."
                )
        else:
            st.info("No subject marks for this student.")

        st.subheader("Marks across exams (grouped by subject)")
        if not s_marks.empty and "ExamNumber" in s_marks.columns and "Subject" in s_marks.columns:
            # grouped bar: x=ExamNumber, color=Subject
            exam_plot = s_marks.groupby(["ExamNumber","Subject"])["Marks"].mean().reset_index()
            fig = px.bar(exam_plot, x="ExamNumber", y="Marks", color="Subject", barmode="group", color_discrete_map=SUBJECT_COLORS)
            fig.update_layout(xaxis_title="Exam number", yaxis_title="Avg marks")
            st.plotly_chart(fig, use_container_width=True)
            with st.expander("What this shows", expanded=auto_expand_explanations):
                st.write(
                    "Grouped bars show per-subject performance for each exam. This is easier to read than a single line when multiple subjects exist."
                )
        else:
            st.info("Not enough exam/subject structure to draw grouped bars.")

        st.subheader("Attendance by month (bars)")
        if not s_att.empty and "Date" in s_att.columns:
            s_att["month"] = s_att["Date"].dt.to_period("M").astype(str)
            monthly = s_att.groupby("month")["_present_flag_"].mean().reset_index()
            fig = px.bar(monthly, x="month", y="_present_flag_", labels={"_present_flag_":"attendance_rate"})
            fig.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)
            with st.expander("What this shows", expanded=auto_expand_explanations):
                st.write("Monthly attendance percentage for the student — easier to interpret than day-level dots.")
        else:
            st.info("No attendance records to show monthly trend.")

        st.write("---")
        # Exports (Excel + PDF)
        st.subheader("Export student report")
        colx, coly = st.columns(2)

        def build_student_excel_bytes(name):
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                s_m = marks_df[marks_df["Name"]==name] if not marks_df.empty else pd.DataFrame()
                s_a = att_df[att_df["Name"]==name] if not att_df.empty else pd.DataFrame()
                s_m.to_excel(writer, sheet_name="Marks", index=False)
                s_a.to_excel(writer, sheet_name="Attendance", index=False)
            buf.seek(0)
            return buf.getvalue()

        def build_student_pdf_bytes(name):
            s_m = marks_df[marks_df["Name"]==name] if not marks_df.empty else pd.DataFrame()
            s_a = att_df[att_df["Name"]==name] if not att_df.empty else pd.DataFrame()
            out_buf = io.BytesIO()
            doc = SimpleDocTemplate(out_buf, pagesize=A4)
            styles = getSampleStyleSheet()
            story = []
            story.append(Paragraph(f"Student Report — {name}", styles["Title"]))
            story.append(Spacer(1,12))
            # basic text
            if not s_m.empty:
                story.append(Paragraph(f"Average marks: {s_m['Marks'].mean():.2f}", styles["Normal"]))
            if not s_a.empty and "_present_flag_" in s_a.columns and len(s_a)>0:
                p = int(s_a["_present_flag_"].sum()); t = len(s_a)
                story.append(Paragraph(f"Attendance: {p}/{t} ({(p/t*100):.1f}%)", styles["Normal"]))
            story.append(Spacer(1,12))

            # Try to embed charts (requires fig.to_image)
            try:
                imgs = []
                if not s_m.empty and "Subject" in s_m.columns:
                    subj_avg = s_m.groupby("Subject")["Marks"].mean().reset_index()
                    f = px.bar(subj_avg, x="Subject", y="Marks", color="Subject", color_discrete_map=SUBJECT_COLORS)
                    img = fig_to_png_bytes(f)
                    if img:
                        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                        tmp.write(img); tmp.flush()
                        imgs.append(tmp.name)
                        story.append(RLImage(tmp.name, width=450, height=200))
                        story.append(Spacer(1,12))
                if not s_a.empty and "Date" in s_a.columns:
                    s_a_sorted = s_a.sort_values("Date")
                    f2 = px.bar(s_a.groupby(s_a["Date"].dt.to_period("M").astype(str))["_present_flag_"].mean().reset_index(),
                                x="Date", y="_present_flag_")
                    img2 = fig_to_png_bytes(f2)
                    if img2:
                        tmp2 = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                        tmp2.write(img2); tmp2.flush()
                        imgs.append(tmp2.name)
                        story.append(RLImage(tmp2.name, width=450, height=150))
                        story.append(Spacer(1,12))
                # If image export isn't available, we'll just have text
            except Exception:
                pass

            doc.build(story)
            out_buf.seek(0)
            pdf_bytes = out_buf.getvalue()
            # cleanup temps
            try:
                for p in imgs:
                    os.unlink(p)
            except Exception:
                pass
            return pdf_bytes

        with colx:
            excel_bytes = build_student_excel_bytes(student)
            st.download_button("Download Excel", data=excel_bytes, file_name=f"{student}_report.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        with coly:
            pdf_bytes = build_student_pdf_bytes(student)
            st.download_button("Download PDF", data=pdf_bytes, file_name=f"{student}_report.pdf", mime="application/pdf")

# -------------------------
# Tab: Compare students
# -------------------------
with tabs[2]:
    st.header("Compare students")
    # allow exam-level or overall comparison
    exam_options = sorted(marks_df["ExamNumber"].dropna().unique().tolist()) if "ExamNumber" in marks_df.columns else []
    exam_filter = st.selectbox("Compare by exam (choose 'All' for overall averages)", options=["All"] + [str(e) for e in exam_options])
    # student selection
    candidate_names = sorted(set(marks_df["Name"].dropna().tolist())) if "Name" in marks_df.columns else []
    selected = st.multiselect("Select up to 6 students", candidate_names, max_selections=6)
    if not selected or len(selected) < 2:
        st.info("Select two or more students to compare.")
    else:
        comp_df = marks_df[marks_df["Name"].isin(selected)].copy()
        if exam_filter != "All":
            comp_df = comp_df[comp_df["ExamNumber"].astype(str) == exam_filter]
        # Simple: side-by-side subject averages for selected students
        st.subheader("Subject averages (comparison)")
        comp_avg = comp_df.groupby(["Name","Subject"])["Marks"].mean().reset_index()
        if comp_avg.empty:
            st.info("No data for selected students/exam.")
        else:
            # wide table for plotting grouped bars by subject with student traces
            subjects = sorted(comp_avg["Subject"].unique())
            fig = go.Figure()
            for i, name in enumerate(selected):
                row = comp_avg[comp_avg["Name"]==name]
                y_vals = [row[row["Subject"]==s]["Marks"].values[0] if s in row["Subject"].values else 0 for s in subjects]
                fig.add_trace(go.Bar(name=name, x=subjects, y=y_vals))
            fig.update_layout(barmode="group", xaxis_title="Subject", yaxis_title="Avg marks")
            st.plotly_chart(fig, use_container_width=True)
            with st.expander("Explanation", expanded=auto_expand_explanations):
                st.write("Compare subject-wise averages between selected students. Use the exam filter to focus on a single exam.")

        # Boxplot: distributions across selected students
        st.subheader("Score distributions (boxplot)")
        dist_df = marks_df[marks_df["Name"].isin(selected)]
        if dist_df.empty:
            st.info("No distribution data for selection.")
        else:
            fig = px.box(dist_df, x="Name", y="Marks", points="all", color="Name", color_discrete_sequence=PALETTE)
            st.plotly_chart(fig, use_container_width=True)
            with st.expander("Explanation", expanded=auto_expand_explanations):
                st.write("Boxplots show distribution and outliers of marks per selected student.")

# -------------------------
# Tab: Attendance explorer
# -------------------------
with tabs[3]:
    st.header("Attendance explorer")
    if att_df.empty:
        st.info("No attendance data.")
    else:
        # date-range selector
        min_date = att_df["Date"].min().date() if "Date" in att_df.columns and not att_df["Date"].isna().all() else date.today()
        max_date = att_df["Date"].max().date() if "Date" in att_df.columns and not att_df["Date"].isna().all() else date.today()
        dr = st.date_input("Select date range", value=(min_date, max_date))
        try:
            if isinstance(dr, (list,tuple)) and len(dr)==2:
                start_d, end_d = dr[0], dr[1]
            else:
                start_d, end_d = dr, dr
        except Exception:
            start_d, end_d = min_date, max_date
        mask = (att_df["Date"].dt.date >= start_d) & (att_df["Date"].dt.date <= end_d)
        att_filtered = att_df[mask]

        # class attendance trend
        st.subheader("Class attendance trend")
        att_trend = att_filtered.groupby(att_filtered["Date"].dt.date)["_present_flag_"].mean().reset_index()
        att_trend.columns = ["Date","attendance_rate"]
        fig = px.line(att_trend, x="Date", y="attendance_rate", markers=True)
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("Explanation", expanded=auto_expand_explanations):
            st.write("Daily class attendance percentage in the selected date range. Look for dips around events/holidays.")

        # weekday heatmap (more readable than massive per-date heatmap)
        st.subheader("Attendance by weekday (heatmap)")
        try:
            att_filtered["weekday"] = att_filtered["Date"].dt.day_name()
            wk = att_filtered.groupby(["weekday","Name"])["_present_flag_"].mean().reset_index()
            # pivot to weekday x name (but we will aggregate by weekday across class)
            weekday_avg = att_filtered.groupby(att_filtered["Date"].dt.day_name())["_present_flag_"].mean().reindex(
                ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
            ).reset_index()
            weekday_avg.columns = ["weekday","attendance_rate"]
            fig = px.bar(weekday_avg, x="weekday", y="attendance_rate")
            fig.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)
            with st.expander("Explanation", expanded=auto_expand_explanations):
                st.write("Average attendance per weekday — reveals patterns like poor attendance on certain weekdays.")
        except Exception:
            st.info("Unable to produce weekday breakdown for this dataset.")

        # Attendance leaderboard (per-student attendance %)
        st.subheader("Attendance leaderboard")
        att_leader = att_filtered.groupby("Name")["_present_flag_"].mean().reset_index().sort_values("_present_flag_", ascending=False)
        if not att_leader.empty:
            st.dataframe(att_leader.head(50).rename(columns={"_present_flag_":"attendance_rate"}).style.format({"attendance_rate":"{:.1%}"}))
            with st.expander("Explanation", expanded=auto_expand_explanations):
                st.write("Leaderboard shows top students by attendance rate in the selected range.")
        else:
            st.info("No attendance summary available for the selected range.")

# -------------------------
# Tab: Insights & Export
# -------------------------
with tabs[4]:
    st.header("Insights & Export")
    # Show flags and useful summaries
    if student_mark_summary.empty:
        st.info("Not enough marks data to produce insights.")
    else:
        # Merge attendance into student level if possible
        student_level = student_mark_summary.copy()
        if not att_summary.empty and "ID" in att_summary.columns:
            student_level = student_level.merge(att_summary[["ID","attendance_rate"]], on="ID", how="left")
        student_level["low_attendance_flag"] = student_level["attendance_rate"].fillna(1) < (min_att_threshold_pct/100.0)
        student_level["low_score_flag"] = student_level["avg_score"] < min_score_threshold

        flagged = student_level[student_level["low_attendance_flag"] | student_level["low_score_flag"]]
        st.subheader("Flagged students")
        if flagged.empty:
            st.success("No students currently flagged with the thresholds set.")
        else:
            st.dataframe(flagged[["ID","Roll","Name","avg_score","attendance_rate","low_attendance_flag","low_score_flag"]].sort_values(["low_attendance_flag","low_score_flag"], ascending=False))

        with st.expander("Why these flags matter", expanded=auto_expand_explanations):
            st.write(
                "Students are flagged when their average marks or attendance fall below the thresholds. "
                "Low attendance is strongly correlated with low learning outcomes — use flagged list to plan interventions."
            )

        st.markdown("---")
        st.subheader("Quick class highlights")
        # top performers
        top_performers = student_level.sort_values("avg_score", ascending=False).head(5)[["Name","avg_score"]]
        with st.expander("Top performers", expanded=False):
            if not top_performers.empty:
                st.table(top_performers.style.format({"avg_score":"{:.2f}"}).set_index("Name"))
            else:
                st.info("No data")

        # subjects with most low scores (failure rate)
        failure_threshold = 40
        if not marks_df.empty and "Subject" in marks_df.columns:
            subj_fail = marks_df.assign(fail=marks_df["Marks"]<failure_threshold).groupby("Subject")["fail"].mean().reset_index().sort_values("fail", ascending=False)
            st.subheader("Subjects by failure rate")
            fig = px.bar(subj_fail, x="Subject", y="fail", color="Subject", color_discrete_map=SUBJECT_COLORS)
            fig.update_layout(yaxis_title="Failure rate")
            fig.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)
            with st.expander("Explanation", expanded=auto_expand_explanations):
                st.write("Shows proportion of records below the failure threshold per subject. Helps prioritize remedial work.")
        else:
            st.info("No subject-level marks to compute failure rate.")

        st.markdown("---")
        st.subheader("Export options")
        # Download flagged list CSV / Excel
        if not flagged.empty:
            csv_bytes = flagged.to_csv(index=False).encode("utf-8")
            st.download_button("Download flagged (CSV)", csv_bytes, file_name="flagged_students.csv", mime="text/csv")
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                flagged.to_excel(writer, index=False, sheet_name="Flagged")
            st.download_button("Download flagged (Excel)", buf.getvalue(), file_name="flagged_students.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        else:
            st.info("No flagged students to export under current thresholds.")

        # Export full class summary (marks + attendance)
        full_export_buf = io.BytesIO()
        with pd.ExcelWriter(full_export_buf, engine="openpyxl") as writer:
            if not marks_df.empty:
                marks_df.to_excel(writer, sheet_name="Marks", index=False)
            if not att_df.empty:
                att_df.to_excel(writer, sheet_name="Attendance", index=False)
            if not student_level.empty:
                student_level.to_excel(writer, sheet_name="StudentSummary", index=False)
        st.download_button("Download full dataset (Excel)", data=full_export_buf.getvalue(), file_name="rightitech_full_export.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.caption("Right iTech — optimized. If you want branding (logo / colors) or scheduled exports, tell me and I'll add those next.")
