import streamlit as st
import pandas as pd
import plotly.express as px

# ======================
# FILE UPLOADS
# ======================
st.title("📊 Attendance + Marks Analyzer")

att_file = st.file_uploader("Upload attendance file (CSV or XLSX)", type=["csv", "xlsx"])
marks_file = st.file_uploader("Upload marks file (CSV or XLSX)", type=["csv", "xlsx"])

if att_file and marks_file:

    # ---- Load attendance ----
    if att_file.name.endswith(".csv"):
        attendance = pd.read_csv(att_file)
    else:
        attendance = pd.read_excel(att_file)

    # ---- Load marks ----
    if marks_file.name.endswith(".csv"):
        marks = pd.read_csv(marks_file)
    else:
        marks = pd.read_excel(marks_file)

    st.success("✅ Files uploaded successfully!")

    # ======================
    # CLEAN + SUMMARIZE ATTENDANCE
    # ======================
    # Expecting columns: ID, Name, Roll, Gender, Date, Status
    attendance["Status"] = attendance["Status"].map({"P": 1, "A": 0})
    att_summary = (
        attendance.groupby(["ID", "Name"])
        .agg(TotalClasses=("Status", "count"),
             TotalPresent=("Status", "sum"))
        .reset_index()
    )
    att_summary["AttendanceRate"] = (
        att_summary["TotalPresent"] / att_summary["TotalClasses"] * 100
    )

    # ======================
    # CLEAN + SUMMARIZE MARKS
    # ======================
    # Expecting columns: ID, Name, Subject, MarksObtained, MarksTotal
    if "MarksObtained" in marks.columns and "MarksTotal" in marks.columns:
        marks["Pct"] = marks["MarksObtained"] / marks["MarksTotal"] * 100
        marks_summary = (
            marks.groupby(["ID", "Name"])
            .agg(ChosenAvgPct=("Pct", "mean"))
            .reset_index()
        )
    else:
        st.error("❌ Marks file must contain `MarksObtained` and `MarksTotal` columns.")
        st.stop()

    # ======================
    # MERGE ATTENDANCE + MARKS
    # ======================
    demo_cols = [c for c in attendance.columns if c not in ["Date", "Status", "AttValue"]]
    demo_info = attendance[demo_cols].drop_duplicates(subset=["ID", "Name"])

    students = pd.merge(
        att_summary,
        marks_summary,
        on=["ID", "Name"],
        how="outer"
    ).merge(
        demo_info,
        on=["ID", "Name"],
        how="left"
    )

    st.subheader("🧑‍🎓 Combined Student Data")
    st.dataframe(students.head(10))

    # ======================
    # ATTENDANCE DISTRIBUTION
    # ======================
    st.markdown("### Attendance Distribution")
    fig1 = px.histogram(
        att_summary,
        x="AttendanceRate",
        nbins=20,
        color_discrete_sequence=["#2E86C1"],
        title="Distribution of Attendance %"
    )
    st.plotly_chart(fig1, use_container_width=True)

    # ======================
    # MARKS DISTRIBUTION
    # ======================
    st.markdown("### Marks Distribution")
    fig2 = px.histogram(
        marks_summary,
        x="ChosenAvgPct",
        nbins=20,
        color_discrete_sequence=["#27AE60"],
        title="Distribution of Average Marks %"
    )
    st.plotly_chart(fig2, use_container_width=True)

    # ======================
    # SCATTER: ATTENDANCE VS PERFORMANCE
    # ======================
    st.markdown("### Attendance vs Performance (scatter + regression)")

    scatter_df = students.dropna(subset=["AttendanceRate", "ChosenAvgPct"]).copy()
    if not scatter_df.empty:
        fig3 = px.scatter(
            scatter_df,
            x="AttendanceRate",
            y="ChosenAvgPct",
            color="Gender" if "Gender" in scatter_df.columns else None,
            trendline="ols",
            hover_data=["Name", "Roll"] if "Roll" in scatter_df.columns else ["Name"],
            title="Attendance vs Average %"
        )
        st.plotly_chart(fig3, use_container_width=True)

        corr = scatter_df["AttendanceRate"].corr(scatter_df["ChosenAvgPct"])
        st.markdown(f"**Correlation between attendance and performance = {corr:.2f}**")

    else:
        st.info("Not enough data to plot Attendance vs Performance.")
