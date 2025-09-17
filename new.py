# ======================
# MERGE ATTENDANCE + MARKS
# ======================
# Keep demographics from attendance (Gender, Roll, etc.)
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
        color_discrete_sequence=QUAL,
        trendline="ols",
        hover_data=["Name", "Roll"] if "Roll" in scatter_df.columns else ["Name"],
        title="Attendance vs Average %"
    )
    st.plotly_chart(fig3, use_container_width=True)

    # Add correlation coefficient
    corr = scatter_df["AttendanceRate"].corr(scatter_df["ChosenAvgPct"])
    st.markdown(f"**Quick insight:** Correlation between attendance and performance = **{corr:.2f}**. " + (
        "🚀 Strong positive relation – students who attend more generally perform better."
        if corr > 0.45 else
        ("🙂 Mild relation – attendance helps but other factors matter too."
         if corr > 0.2 else
         "⚠️ Weak relation – attendance doesn’t strongly explain performance.")
    ))

    # Optional toggle for non-statisticians
    with st.expander("🔍 What does this mean?"):
        st.write(
            """
            Each point represents a student.  
            - **X-axis**: Attendance rate (% of classes attended).  
            - **Y-axis**: Average performance (% across exams).  
            - **Color**: Gender (if available).  
            
            The regression line shows the *overall trend*.  
            The correlation value quantifies the strength of the relationship:
            - Close to **1** → Strong positive link (more attendance = better performance).
            - Around **0** → No clear link.
            - Negative → More attendance correlates with worse performance (unlikely in school settings).
            """
        )
else:
    st.info("Not enough data to plot Attendance vs Performance (missing values).")
