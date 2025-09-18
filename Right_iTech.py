import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.io as pio
from io import BytesIO
from fpdf import FPDF
import tempfile

# ================================
# App Configuration
# ================================
st.set_page_config(page_title="Right iTech", layout="wide")
st.title("📊 Right iTech - Student Analytics Dashboard")

# ================================
# Sidebar Upload
# ================================
st.sidebar.header("📂 Upload Your Data")

att_file = st.sidebar.file_uploader("Upload Attendance CSV", type="csv")
marks_file = st.sidebar.file_uploader("Upload Marks CSV", type="csv")

if att_file and marks_file:
    attendance_df = pd.read_csv(att_file)
    marks_df = pd.read_csv(marks_file)
    attendance_df['Date'] = pd.to_datetime(attendance_df['Date'])
else:
    st.warning("⚠️ Please upload both Attendance and Marks CSV files to proceed.")
    st.stop()

# ================================
# Sidebar Navigation
# ================================
tabs = ["📌 Overview", "👤 Student Profile", "📈 Compare Students", "⭐ Top/Bottom Performers", "📂 Exports & Insights"]
choice = st.sidebar.radio("Navigate Dashboard", tabs)

# ================================
# Class Overview Tab
# ================================
if choice == "📌 Overview":
    st.header("Class Overview")

    total_students = attendance_df['ID'].nunique()
    girls = attendance_df[attendance_df['Gender'] == 'F']['ID'].nunique()
    boys = total_students - girls
    avg_attendance = (attendance_df['Attendance'] == 'Present').mean()
    avg_absence = 1 - avg_attendance

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("👩‍🎓 Total Students", total_students)
    col2.metric("👧 Girls", girls)
    col3.metric("👦 Boys", boys)
    col4.metric("📅 Avg Attendance", f"{avg_attendance:.1%}")

    # Attendance Pie Chart
    att_pie = pd.DataFrame({
        "Status": ["Present", "Absent"],
        "Rate": [avg_attendance, avg_absence]
    })
    fig_pie = px.pie(att_pie, names='Status', values='Rate',
                     title='Overall Attendance Distribution',
                     color='Status',
                     color_discrete_map={'Present':'#2ecc71','Absent':'#e74c3c'})
    st.plotly_chart(fig_pie, use_container_width=True)

    with st.expander("ℹ️ Explanation"):
        st.write("The overview shows the number of boys, girls, and average attendance. "
                 "The pie chart reflects the percentage of time students are present vs absent across the dataset.")

    # Trend of attendance
    att_trend = attendance_df.groupby('Date')['Attendance'].apply(lambda x: (x=='Present').mean()).reset_index(name='Attendance Rate')
    fig_trend = px.line(att_trend, x='Date', y='Attendance Rate',
                        title='Class Average Attendance Over Time')
    fig_trend.update_yaxes(tickformat='.0%')
    st.plotly_chart(fig_trend, use_container_width=True)

    with st.expander("ℹ️ Explanation"):
        st.write("This shows the daily average attendance rate across the class. "
                 "Spikes or dips highlight anomalies like holidays or exam days.")

# ================================
# Student Profile Tab
# ================================
elif choice == "👤 Student Profile":
    st.header("Individual Student Report")
    student_list = attendance_df[['ID','Name']].drop_duplicates().sort_values('Name')
    student_choice = st.selectbox("Select a Student", student_list['Name'])
    student_id = student_list[student_list['Name']==student_choice]['ID'].iloc[0]

    student_att = attendance_df[attendance_df['ID']==student_id]
    student_marks = marks_df[marks_df['ID']==student_id]

    st.subheader(f"🎓 {student_choice} ({student_id})")

    # Attendance percentage for this student
    att_rate = (student_att['Attendance']=='Present').mean()
    st.metric("Attendance %", f"{att_rate:.1%}")

    # Attendance summary
    att_counts = student_att['Attendance'].value_counts()
    fig_bar = px.bar(att_counts, x=att_counts.index, y=att_counts.values,
                     labels={'x':'Status','y':'Count'},
                     title="Attendance Summary",
                     color=att_counts.index,
                     color_discrete_map={'Present':'#2ecc71','Absent':'#e74c3c'})
    st.plotly_chart(fig_bar, use_container_width=True)

    with st.expander("ℹ️ Explanation"):
        st.write("This section highlights the student’s attendance performance. "
                 "The bar chart clearly shows the number of Presents and Absents.")

    # Marks trends
    if not student_marks.empty:
        fig_marks = px.bar(student_marks, x='Exam', y='Marks', color='Subject', barmode='group',
                           title='Marks Across Exams by Subject')
        st.plotly_chart(fig_marks, use_container_width=True)

        with st.expander("ℹ️ Explanation"):
            st.write("The grouped bar chart shows how this student performed in each subject across different exams.")

# ================================
# Compare Students Tab
# ================================
elif choice == "📈 Compare Students":
    st.header("Compare Students")
    student_options = attendance_df['Name'].unique()
    selected_students = st.multiselect("Select Students", student_options, default=student_options[:2])

    if selected_students:
        compare_df = marks_df[marks_df['Name'].isin(selected_students)]

        # Subject comparison
        subj_avg = compare_df.groupby(['Name','Subject'])['Marks'].mean().reset_index()
        fig_cmp = px.bar(subj_avg, x='Subject', y='Marks', color='Name', barmode='group',
                         title='Average Marks Comparison by Subject')
        st.plotly_chart(fig_cmp, use_container_width=True)

        with st.expander("ℹ️ Explanation"):
            st.write("Compares the average subject scores across selected students. "
                     "You can quickly identify strong and weak subjects.")

        # Interactive exam comparison
        exam_choice = st.selectbox("Select Exam for Comparison", compare_df['Exam'].unique())
        exam_cmp = compare_df[compare_df['Exam']==exam_choice]
        fig_exam = px.bar(exam_cmp, x='Subject', y='Marks', color='Name', barmode='group',
                          title=f'Marks Comparison in {exam_choice}')
        st.plotly_chart(fig_exam, use_container_width=True)

        with st.expander("ℹ️ Explanation"):
            st.write(f"Compares selected students’ performance subject-wise for the {exam_choice} exam.")

# ================================
# Top/Bottom Performers Tab
# ================================
elif choice == "⭐ Top/Bottom Performers":
    st.header("Top & Bottom Performers")

    num_students = st.slider("Select number of students", 3, 20, 5)
    exam_filter = st.selectbox("Select Exam", options=["Overall"] + list(marks_df['Exam'].unique()))

    if exam_filter == "Overall":
        avg_marks = marks_df.groupby('Name')['Marks'].mean().reset_index()
    else:
        avg_marks = marks_df[marks_df['Exam']==exam_filter].groupby('Name')['Marks'].mean().reset_index()

    top_students = avg_marks.sort_values('Marks', ascending=False).head(num_students)
    bottom_students = avg_marks.sort_values('Marks').head(num_students)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🏆 Top Performers")
        fig_top = px.bar(top_students, x='Name', y='Marks', color='Marks',
                         color_continuous_scale='greens')
        st.plotly_chart(fig_top, use_container_width=True)
    with col2:
        st.subheader("📉 Bottom Performers")
        fig_bottom = px.bar(bottom_students, x='Name', y='Marks', color='Marks',
                            color_continuous_scale='reds')
        st.plotly_chart(fig_bottom, use_container_width=True)

    with st.expander("ℹ️ Explanation"):
        st.write("This section highlights the top and bottom performers. "
                 "You can adjust how many to show and filter by a specific exam or overall average.")

# ================================
# Exports & Insights Tab
# ================================
elif choice == "📂 Exports & Insights":
    st.header("Exports & Insights")

    st.subheader("Download Data")
    csv_data = attendance_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Attendance Data (CSV)", data=csv_data, file_name="attendance_data.csv")

    csv_marks = marks_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Marks Data (CSV)", data=csv_marks, file_name="marks_data.csv")

    # PDF Export
    total_students = attendance_df['ID'].nunique()
    girls = attendance_df[attendance_df['Gender'] == 'F']['ID'].nunique()
    boys = total_students - girls
    avg_attendance = (attendance_df['Attendance'] == 'Present').mean()

    if st.button("📄 Export PDF Report"):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=14)
        pdf.cell(200, 10, txt="Right iTech - Class Report", ln=True, align='C')

        pdf.set_font("Arial", size=12)
        pdf.ln(10)
        pdf.cell(200, 10, txt=f"Total Students: {total_students}", ln=True)
        pdf.cell(200, 10, txt=f"Girls: {girls}, Boys: {boys}", ln=True)
        pdf.cell(200, 10, txt=f"Average Attendance: {avg_attendance:.1%}", ln=True)

        # --- Export charts as images and embed ---
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile1:
            att_summary = pd.DataFrame({
                "Status": ["Present", "Absent"],
                "Rate": [avg_attendance, 1-avg_attendance]
            })
            fig_pie = px.pie(att_summary, names="Status", values="Rate",
                             color="Status", color_discrete_map={"Present":"#2ecc71","Absent":"#e74c3c"})
            pio.write_image(fig_pie, tmpfile1.name, format="png", width=600, height=400)
            pdf.image(tmpfile1.name, w=150)

        pdf.ln(5)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile2:
            att_trend = attendance_df.groupby('Date')['Attendance'].apply(lambda x: (x=='Present').mean()).reset_index(name='Attendance Rate')
            fig_trend = px.line(att_trend, x='Date', y='Attendance Rate', title='Class Attendance Over Time')
            fig_trend.update_yaxes(tickformat='.0%')
            pio.write_image(fig_trend, tmpfile2.name, format="png", width=600, height=400)
            pdf.image(tmpfile2.name, w=180)

        pdf.ln(5)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile3:
            avg_marks = marks_df.groupby('Name')['Marks'].mean().reset_index()
            top_students = avg_marks.sort_values('Marks', ascending=False).head(5)
            fig_top = px.bar(top_students, x='Name', y='Marks', color='Marks',
                             color_continuous_scale='greens', title="Top 5 Performers")
            pio.write_image(fig_top, tmpfile3.name, format="png", width=600, height=400)
            pdf.image(tmpfile3.name, w=180)

        # --- Save and return as download ---
        buffer = BytesIO()
        pdf.output(buffer)
        st.download_button("📄 Download PDF Report", data=buffer.getvalue(), file_name="class_report.pdf")

    with st.expander("ℹ️ Explanation"):
        st.write("Here you can export both the attendance and marks dataset in CSV format, "
                 "as well as download a PDF summary report of the class that includes key charts.")
