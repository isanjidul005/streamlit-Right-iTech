# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

st.set_page_config(page_title="Super Class Dashboard", layout="wide")
st.title("🌟 Super Duper Class Dashboard — Attendance & Performance")

# ----------------------
# Color / style settings (consistent palette)
# ----------------------
px.defaults.template = "plotly_white"
SEQ = px.colors.sequential.Viridis  # sequential for continuous
QUAL = px.colors.qualitative.Bold    # categorical
ACCENT = "#EF553B"

# ----------------------
# Sidebar: uploads + options
# ----------------------
st.sidebar.header("Upload cleaned CSVs")
att_file = st.sidebar.file_uploader("Attendance CSV (clean_attendance.csv)", type=["csv"])
marks_file = st.sidebar.file_uploader("Marks CSV (clean_marks.csv)", type=["csv"])

st.sidebar.markdown("---")
st.sidebar.header("Analysis options")
attendance_cutoff = st.sidebar.slider("Attendance risk cutoff (%)", 40, 95, 75)
include_absents_mode = st.sidebar.checkbox("Treat absences as 0% in averages (include absents)", value=False)
n_clusters = st.sidebar.slider("Clustering: # clusters", 2, 8, 3)
st.sidebar.markdown("Tip: If you suspect merge problems, expand 'Data diagnostics' after upload.")

# Helper: normalize attendance statuses robustly
def normalize_status(val):
    s = str(val).strip()
    if s == "" or s.lower() in ["nan", "none", "na"]:
        return 0
    s_lower = s.lower()
    # ticks/present
    if any(ch in s for ch in ["✔", "✓"]) or s_lower in ["p", "present", "yes", "1", "y"]:
        return 1
    # crosses/absent
    if any(ch in s for ch in ["x", "✗", "✘"]) or s_lower in ["a", "absent", "no", "0", "n"]:
        return 0
    # time values or numeric strings -> treat as absent per your rule
    if ":" in s or s.replace(".", "", 1).isdigit():
        return 0
    # fallback: absent
    return 0

# ----------------------
# Require uploads
# ----------------------
if not att_file or not marks_file:
    st.info("Upload both cleaned attendance and marks CSV files (use the sidebar).")
    st.stop()

# ----------------------
# Load files
# ----------------------
attendance = pd.read_csv(att_file)
marks = pd.read_csv(marks_file)

# Quick preview (collapsible)
with st.expander("Preview: Attendance (first 10 rows)"):
    st.dataframe(attendance.head(10))
with st.expander("Preview: Marks (first 10 rows)"):
    st.dataframe(marks.head(10))

# ----------------------
# Basic validation & cleaning
# ----------------------
# Normalize column names
attendance.columns = attendance.columns.str.strip()
marks.columns = marks.columns.str.strip()

# If Status column exists, normalize it; otherwise try alternatives
if "Status" not in attendance.columns:
    # try common alternatives
    for alt in ["Present/Absent", "Att", "Raw"]:
        if alt in attendance.columns:
            attendance = attendance.rename(columns={alt: "Status"})
            break

attendance["StatusPresent"] = attendance["Status"].apply(normalize_status) if "Status" in attendance.columns else 0

# Parse Date if present
if "Date" in attendance.columns:
    attendance["DateParsed"] = pd.to_datetime(attendance["Date"], dayfirst=True, errors="coerce")
else:
    attendance["DateParsed"] = pd.NaT

# Ensure Marks fields
if "WasAbsent" not in marks.columns:
    marks["WasAbsent"] = marks["Marks"].isna()

marks["Marks"] = pd.to_numeric(marks["Marks"], errors="coerce")
if "FullMarks" in marks.columns:
    marks["FullMarks"] = pd.to_numeric(marks["FullMarks"], errors="coerce")
else:
    marks["FullMarks"] = np.nan

# Compute Percent per exam (Marks / FullMarks * 100) when FullMarks available; otherwise keep Marks
def compute_percent(row):
    if pd.isna(row["Marks"]):
        return np.nan
    if not pd.isna(row["FullMarks"]) and row["FullMarks"] > 0:
        return (row["Marks"] / row["FullMarks"]) * 100.0
    return row["Marks"]

marks["Percent"] = marks.apply(compute_percent, axis=1)

# ----------------------
# Attendance summary per student
# ----------------------
# Ensure ID and Name exist
for df, name in [(attendance, "attendance"), (marks, "marks")]:
    if "ID" not in df.columns or "Name" not in df.columns:
        st.error(f"ERROR: The {name} file must include 'ID' and 'Name' columns. Found: {list(df.columns)}")
        st.stop()

attendance["ID"] = attendance["ID"].astype(str).str.strip()
attendance["Name"] = attendance["Name"].astype(str).str.strip()
marks["ID"] = marks["ID"].astype(str).str.strip()
marks["Name"] = marks["Name"].astype(str).str.strip()

att_summary = (
    attendance.groupby(["ID", "Name"])
    .agg(TotalDays=("DateParsed", "count"), DaysPresent=("StatusPresent", "sum"))
    .reset_index()
)
att_summary["AttendanceRate"] = (att_summary["DaysPresent"] / att_summary["TotalDays"] * 100).round(2)

# ----------------------
# Marks summary per student (subject-agnostic)
# ----------------------
# avg excluding absents
avg_excl = marks.loc[~marks["WasAbsent"]].groupby(["ID", "Name"])["Percent"].mean().reset_index(name="AvgPct_ExcludeAbsents")
# avg including absents as zeros (only meaningful if exams per student are comparable)
exam_counts = marks.groupby(["ID", "Name"])["ExamNumber"].nunique().reset_index(name="NumExams")
avg_incl = marks.copy()
avg_incl["PercentFilled"] = avg_incl["Percent"]
avg_incl.loc[avg_incl["WasAbsent"] == True, "PercentFilled"] = 0
avg_incl = avg_incl.groupby(["ID", "Name"])["PercentFilled"].mean().reset_index(name="AvgPct_IncludeAbsents")

marks_summary = pd.merge(avg_excl, avg_incl, on=["ID", "Name"], how="outer")

# ----------------------
# Merge students (use ID only for safe merging)
# ----------------------
# sanitize keys
att_summary["ID"] = att_summary["ID"].astype(str).str.strip()
marks_summary["ID"] = marks_summary["ID"].astype(str).str.strip()
att_summary["Name"] = att_summary["Name"].astype(str).str.strip()
marks_summary["Name"] = marks_summary["Name"].astype(str).str.strip()

# Merge on ID primarily, then try to carry name
students = pd.merge(att_summary, marks_summary, on="ID", how="outer", suffixes=("_att", "_marks"))

# Reconstruct Name (prefer attendance name)
if "Name_att" in students.columns and "Name_marks" in students.columns:
    students["Name"] = students["Name_att"].combine_first(students["Name_marks"])
    students = students.drop(columns=["Name_att", "Name_marks"])
elif "Name_att" in students.columns:
    students = students.rename(columns={"Name_att": "Name"})
elif "Name_marks" in students.columns:
    students = students.rename(columns={"Name_marks": "Name"})

# Fill numeric NaNs with appropriate placeholder
students["AttendanceRate"] = students["AttendanceRate"].fillna(0)
students["AvgPct_ExcludeAbsents"] = students["AvgPct_ExcludeAbsents"]
students["AvgPct_IncludeAbsents"] = students["AvgPct_IncludeAbsents"]

# ----------------------
# Diagnostics: unmatched
# ----------------------
att_ids = set(att_summary["ID"].unique())
mark_ids = set(marks_summary["ID"].unique())
only_att = sorted(list(att_ids - mark_ids))
only_marks = sorted(list(mark_ids - att_ids))

with st.expander("🔎 Data diagnostics (unmatched students)"):
    st.write(f"Students in attendance but not in marks: {len(only_att)}")
    if len(only_att) > 0:
        st.dataframe(pd.DataFrame({"ID_only_in_attendance": only_att}).head(200))
    st.write(f"Students in marks but not in attendance: {len(only_marks)}")
    if len(only_marks) > 0:
        st.dataframe(pd.DataFrame({"ID_only_in_marks": only_marks}).head(200))

# ----------------------
# Compute chosen average column
# ----------------------
students["ChosenAvgPct"] = np.where(include_absents_mode, students["AvgPct_IncludeAbsents"], students["AvgPct_ExcludeAbsents"])

# convenience: replace possible NaNs with np.nan (keep NaN for absent)
# ----------------------
# Build subject pivot heatmap
subject_pivot = marks.loc[~marks["WasAbsent"]].groupby(["Name", "Subject"])["Percent"].mean().reset_index()
subject_heat = subject_pivot.pivot(index="Name", columns="Subject", values="Percent").fillna(np.nan)

# ----------------------
# UI: Tabs with many useful visualizations
# ----------------------
tabs = st.tabs([
    "Class Overview",
    "Attendance Details",
    "Subject Analysis",
    "Student Profile",
    "Compare Students",
    "Trends",
    "Clustering"
])

# ---------- TAB: Class Overview ----------
with tabs[0]:
    st.header("Class Overview + Quick Insights")
    c1, c2, c3 = st.columns(3)
    c1.metric("Students (unique)", int(students["ID"].nunique()))
    c2.metric("Avg Attendance %", f"{students['AttendanceRate'].mean():.1f}")
    mean_chosen = students["ChosenAvgPct"].mean()
    c3.metric(f"Avg Score ({'incl' if include_absents_mode else 'excl'} absents)", f"{mean_chosen:.1f}" if not np.isnan(mean_chosen) else "N/A")

    st.markdown("### Attendance distribution")
    fig = px.violin(students, y="AttendanceRate", box=True, points="all", color_discrete_sequence=[QUAL[0]])
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("What this shows"):
        st.write("Violin plot shows the distribution of attendance. Thicker parts = more students at that rate.")

    st.markdown("### Scores distribution (chosen averaging mode)")
    fig2 = px.histogram(students, x="ChosenAvgPct", nbins=25, color_discrete_sequence=[ACCENT], marginal="box")
    st.plotly_chart(fig2, use_container_width=True)
    with st.expander("What this shows"):
        st.write("Histogram of student average percentages. You can toggle include-absents mode in the sidebar.")

    # Attendance vs Performance scatter with regression
    st.markdown("### Attendance vs Performance (scatter + regression)")
    scatter_df = students.dropna(subset=["AttendanceRate", "ChosenAvgPct"]).copy()
    if not scatter_df.empty:
        fig3 = px.scatter(scatter_df, x="AttendanceRate", y="ChosenAvgPct",
                          color=attendance.get("Gender", pd.Series()).map(lambda x: x if x is not None else "Unknown") if "Gender" in attendance.columns else None,
                          color_discrete_sequence=QUAL,
                          trendline="ols", hover_data=["Name"], title="Attendance vs Average %")
        st.plotly_chart(fig3, use_container_width=True)
        corr = scatter_df["AttendanceRate"].corr(scatter_df["ChosenAvgPct"])
        st.markdown(f"**Quick insight:** correlation = **{corr:.2f}**. " + (
            "Strong relation" if corr > 0.45 else ("Mild relation" if corr > 0.2 else "Weak relation")))
    else:
        st.info("Not enough data to plot Attendance vs Performance (missing values).")

# ---------- TAB: Attendance Details ----------
with tabs[1]:
    st.header("Attendance — deep dive")
    st.write("This section inspects attendance per-date and per-student.")

    # attendance over time (class attendance % per date)
    if "DateParsed" in attendance.columns:
        class_by_date = attendance.groupby("DateParsed")["StatusPresent"].mean().reset_index(name="ClassAttendancePct")
        class_by_date["ClassAttendancePct"] *= 100
        class_by_date = class_by_date.sort_values("DateParsed")
        fig_time = px.line(class_by_date, x="DateParsed", y="ClassAttendancePct", title="Class attendance over time", markers=True, color_discrete_sequence=[QUAL[1]])
        st.plotly_chart(fig_time, use_container_width=True)
        with st.expander("Explanation"):
            st.write("Shows how attendance percentage for the whole class changes across dates.")
    else:
        st.info("Attendance file has no Date column — cannot show time series.")

    # Top / bottom attendance students
    st.markdown("### Top / bottom attendance students")
    topk = st.slider("Show top/bottom K students", 3, 30, 6)
    top = students.sort_values("AttendanceRate", ascending=False).head(topk)
    bottom = students.sort_values("AttendanceRate", ascending=True).head(topk)
    st.write("Top attendance")
    st.dataframe(top[["ID", "Name", "AttendanceRate"]])
    st.write("Bottom attendance")
    st.dataframe(bottom[["ID", "Name", "AttendanceRate"]])

# ---------- TAB: Subject Analysis ----------
with tabs[2]:
    st.header("Subject Analysis")
    st.write("Average scores per subject and heatmap of students vs subjects")

    subj_avg = marks.loc[~marks["WasAbsent"]].groupby("Subject")["Percent"].mean().reset_index()
    fig_sub = px.bar(subj_avg, x="Subject", y="Percent", color="Percent", color_continuous_scale=SEQ, title="Average % by Subject")
    st.plotly_chart(fig_sub, use_container_width=True)
    with st.expander("Explanation"):
        st.write("Bars show mean % per subject (only exams attempted are counted).")

    st.markdown("### Students vs Subjects heatmap (avg %)")
    if not subject_heat.empty:
        fig_heat = px.imshow(subject_heat.fillna(0), labels=dict(x="Subject", y="Student", color="Avg %"), color_continuous_scale=SEQ)
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.info("No subject-level marks to build heatmap.")

# ---------- TAB: Student Profile ----------
with tabs[3]:
    st.header("Student Profile")
    names = students["Name"].dropna().unique().tolist()
    selected = st.selectbox("Choose student", names)
    srow = students[students["Name"] == selected]
    if srow.empty:
        st.warning("Student not found after merge.")
    else:
        s = srow.iloc[0]
        st.subheader(f"{s['Name']} — summary")
        c1, c2, c3 = st.columns(3)
        c1.metric("Attendance %", f"{s['AttendanceRate']:.1f}")
        c2.metric("Avg % (excl absents)", f"{s['AvgPct_ExcludeAbsents']:.1f}" if pd.notna(s['AvgPct_ExcludeAbsents']) else "N/A")
        c3.metric("Avg % (incl absents)", f"{s['AvgPct_IncludeAbsents']:.1f}" if pd.notna(s['AvgPct_IncludeAbsents']) else "N/A")

        st.markdown("#### Recent exam records")
        rec = marks[marks["Name"] == selected].sort_values(by=["Subject","ExamNumber"])
        st.dataframe(rec[["Subject","ExamType","ExamNumber","Marks","FullMarks","Percent","WasAbsent"]].head(50))

        st.markdown("#### Subject radar")
        subj_row = subject_heat.loc[selected] if selected in subject_heat.index else None
        if subj_row is not None and subj_row.dropna().size>0:
            labels = subj_row.index.tolist()
            values = [0 if np.isnan(v) else v for v in subj_row.values]
            fig_r = go.Figure(go.Scatterpolar(r=values, theta=labels, fill='toself', name=selected))
            fig_r.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,100])))
            st.plotly_chart(fig_r, use_container_width=True)
        else:
            st.info("No subject-level marks available for this student.")

# ---------- TAB: Compare Students ----------
with tabs[4]:
    st.header("Compare Students")
    names = students["Name"].dropna().unique().tolist()
    chosen = st.multiselect("Select students (2–6)", names, default=names[:2])
    if len(chosen) < 2:
        st.info("Pick at least two students to compare.")
    else:
        comp_df = subject_pivot[subject_heat.index.isin(chosen)].fillna(0) if not subject_heat.empty else None
        if comp_df is not None:
            comp_df = subject_heat.loc[chosen].fillna(0)
            fig_comp = px.bar(comp_df.T, barmode="group", title="Subject-wise comparison (selected students)", color_discrete_sequence=QUAL)
            st.plotly_chart(fig_comp, use_container_width=True)
        else:
            st.info("Subject-level marks not available to compare.")

# ---------- TAB: Trends ----------
with tabs[5]:
    st.header("Trends Over Time")
    if "ExamNumber" in marks.columns:
        # Class average trend
        trend = marks.loc[~marks["WasAbsent"]].groupby("ExamNumber")["Percent"].mean().reset_index()
        fig_tr = px.line(trend, x="ExamNumber", y="Percent", markers=True, title="Class average % by ExamNumber", color_discrete_sequence=[QUAL[2]])
        st.plotly_chart(fig_tr, use_container_width=True)
    else:
        st.info("No ExamNumber column found in marks to compute time trends.")

# ---------- TAB: Clustering ----------
with tabs[6]:
    st.header("Clustering & Segmentation")
    feat = students[["AttendanceRate", "ChosenAvgPct"]].dropna()
    if feat.shape[0] >= n_clusters:
        scaler = StandardScaler()
        X = scaler.fit_transform(feat)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        plot_df = feat.copy().reset_index()
        plot_df["Cluster"] = labels
        plot_df["Name"] = students.set_index("ID").loc[plot_df["ID"], "Name"].values
        fig_cl = px.scatter(plot_df, x="AttendanceRate", y="ChosenAvgPct", color="Cluster", hover_data=["Name"], color_discrete_sequence=QUAL)
        st.plotly_chart(fig_cl, use_container_width=True)
    else:
        st.info("Not enough students with both attendance and score to cluster.")

# ----------------------
# Export merged summary
# ----------------------
st.markdown("---")
export_df = students.copy()
export_df["ChosenAvgPct"] = students["ChosenAvgPct"]
csv = export_df.to_csv(index=False)
st.download_button("Download merged student summary (CSV)", csv, file_name="student_summary.csv", mime="text/csv")

st.caption("Colors: sequential Viridis for continuous, Bold qualitative for categories. Explanations hidden by default inside expanders.")
