import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import calendar
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(
    page_title="Student Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load data
@st.cache_data
def load_data():
    # Load attendance data
    attendance_df = pd.read_csv('cleaned_attendance_data.csv')
    
    # Load result data
    result_df = pd.read_csv('cleaned_result_data.csv')
    
    return attendance_df, result_df

# Preprocess data
@st.cache_data
def preprocess_data(attendance_df, result_df):
    # Convert Date column to datetime
    attendance_df['Date'] = pd.to_datetime(attendance_df['Date'])
    
    # Extract month and day from Date
    attendance_df['Month'] = attendance_df['Date'].dt.month
    attendance_df['Day'] = attendance_df['Date'].dt.day
    attendance_df['Day_Name'] = attendance_df['Date'].dt.day_name()
    attendance_df['Week'] = attendance_df['Date'].dt.isocalendar().week
    
    # Create a status code for attendance (Present=1, Absent=0)
    attendance_df['Status_Code'] = attendance_df['Status'].apply(lambda x: 1 if x == 'Present' else 0)
    
    # Calculate total marks for each student
    numeric_cols = result_df.select_dtypes(include=[np.number]).columns.tolist()
    if 'Merit' in numeric_cols:
        numeric_cols.remove('Merit')
    if 'ID' in numeric_cols:
        numeric_cols.remove('ID')
    if 'Roll' in numeric_cols:
        numeric_cols.remove('Roll')
    
    # Calculate average marks for each assessment category
    assessment_columns = [col for col in result_df.columns if 'WMT' in col and '.' not in col]
    assessment_categories = list(set([col.split('[')[0].strip() for col in assessment_columns]))
    
    # Create category averages
    for category in assessment_categories:
        category_cols = [col for col in result_df.columns if category in col]
        result_df[f'{category}_Avg'] = result_df[category_cols].apply(
            lambda x: pd.to_numeric(x, errors='coerce').mean(), axis=1)
    
    return attendance_df, result_df, assessment_categories

# Create heatmap calendar for attendance
def create_attendance_heatmap(student_attendance, student_name):
    # Create a pivot table for the heatmap
    heatmap_data = student_attendance.pivot_table(
        values='Status_Code', 
        index='Day_Name', 
        columns='Week', 
        aggfunc='mean',
        fill_value=0
    )
    
    # Order days of week properly
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    heatmap_data = heatmap_data.reindex(days_order)
    
    # Create heatmap
    fig = px.imshow(
        heatmap_data,
        labels=dict(x="Week", y="Day", color="Attendance Rate"),
        title=f"Attendance Calendar Heatmap for {student_name}",
        aspect="auto",
        color_continuous_scale="RdYlGn"
    )
    
    fig.update_layout(
        xaxis_title="Week of Year",
        yaxis_title="Day of Week"
    )
    
    return fig

# Create performance radar chart
def create_radar_chart(student_data, student_name, categories):
    fig = go.Figure()
    
    # Get the average scores for each category
    values = [student_data[f'{category}_Avg'].values[0] for category in categories]
    
    # Add trace
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name=student_name
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 30]
            )),
        title=f"Performance Radar Chart for {student_name}",
        showlegend=True
    )
    
    return fig

# Main app
def main():
    st.title("🎓 Student Performance & Attendance Analytics Dashboard")
    
    # Load data
    attendance_df, result_df = load_data()
    
    # Preprocess data
    attendance_df, result_df, assessment_categories = preprocess_data(attendance_df, result_df)
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏫 Class Overview", 
        "👤 Student Profile", 
        "📊 Comparison", 
        "📅 Attendance Analysis",
        "📈 Performance Trends"
    ])
    
    # Tab 1: Class Overview
    with tab1:
        st.header("Class Overview")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_students = result_df['Name'].nunique()
            st.metric("Total Students", total_students)
        
        with col2:
            avg_total_marks = result_df['Total (1170)'].mean()
            st.metric("Average Total Marks", f"{avg_total_marks:.2f}")
        
        with col3:
            attendance_rate = attendance_df['Status_Code'].mean() * 100
            st.metric("Overall Attendance Rate", f"{attendance_rate:.1f}%")
        
        # Performance distribution
        st.subheader("Performance Distribution")
        fig = px.histogram(result_df, x='Total (1170)', nbins=20, 
                          title="Distribution of Total Marks")
        st.plotly_chart(fig, use_container_width=True)
        
        # Gender performance comparison
        st.subheader("Gender Performance Comparison")
        if 'Gender' in result_df.columns:
            gender_performance = result_df.groupby('Gender')['Total (1170)'].mean().reset_index()
            fig = px.bar(gender_performance, x='Gender', y='Total (1170)', 
                        title="Average Performance by Gender")
            st.plotly_chart(fig, use_container_width=True)
        
        # Top performers
        st.subheader("Top 5 Performers")
        top_performers = result_df.nlargest(5, 'Total (1170)')[['Name', 'Total (1170)', 'Merit']]
        st.dataframe(top_performers)
    
    # Tab 2: Student Profile
    with tab2:
        st.header("Individual Student Profile")
        
        # Student selection
        student_names = result_df['Name'].unique()
        selected_student = st.selectbox("Select a Student", student_names)
        
        if selected_student:
            # Get student data
            student_result = result_df[result_df['Name'] == selected_student]
            student_attendance = attendance_df[attendance_df['Name'] == selected_student]
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_marks = student_result['Total (1170)'].values[0]
                st.metric("Total Marks", f"{total_marks}")
            
            with col2:
                merit = student_result['Merit'].values[0]
                st.metric("Merit Position", f"{merit}")
            
            with col3:
                attendance_count = student_attendance['Status_Code'].sum()
                total_days = len(student_attendance)
                attendance_percent = (attendance_count / total_days) * 100 if total_days > 0 else 0
                st.metric("Attendance Rate", f"{attendance_percent:.1f}%")
            
            with col4:
                if 'Gender' in student_result.columns:
                    gender = student_result['Gender'].values[0]
                    st.metric("Gender", gender)
            
            # Create tabs for student details
            student_tab1, student_tab2, student_tab3 = st.tabs(["Performance", "Attendance", "Progress"])
            
            with student_tab1:
                # Radar chart for assessment categories
                radar_fig = create_radar_chart(student_result, selected_student, assessment_categories)
                st.plotly_chart(radar_fig, use_container_width=True)
                
                # Show all marks
                st.subheader("Detailed Marks")
                numeric_data = student_result.select_dtypes(include=[np.number]).iloc[0]
                st.dataframe(numeric_data)
            
            with student_tab2:
                # Attendance heatmap
                heatmap_fig = create_attendance_heatmap(student_attendance, selected_student)
                st.plotly_chart(heatmap_fig, use_container_width=True)
                
                # Monthly attendance trend
                monthly_attendance = student_attendance.groupby('Month')['Status_Code'].mean().reset_index()
                monthly_attendance['Month'] = monthly_attendance['Month'].apply(lambda x: calendar.month_name[x])
                fig = px.bar(monthly_attendance, x='Month', y='Status_Code', 
                            title="Monthly Attendance Rate", labels={'Status_Code': 'Attendance Rate'})
                st.plotly_chart(fig, use_container_width=True)
            
            with student_tab3:
                # Progress over time (if we had temporal data for marks)
                st.info("Progress tracking would require temporal assessment data.")
    
    # Tab 3: Comparison
    with tab3:
        st.header("Student Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            student1 = st.selectbox("Select First Student", student_names, key="student1")
        
        with col2:
            # Filter out the first student from the second dropdown
            remaining_students = [s for s in student_names if s != student1]
            student2 = st.selectbox("Select Second Student", remaining_students, key="student2")
        
        if student1 and student2:
            # Get data for both students
            student1_data = result_df[result_df['Name'] == student1]
            student2_data = result_df[result_df['Name'] == student2]
            
            student1_attendance = attendance_df[attendance_df['Name'] == student1]
            student2_attendance = attendance_df[attendance_df['Name'] == student2]
            
            # Create comparison metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(f"{student1} Total Marks", student1_data['Total (1170)'].values[0])
                st.metric(f"{student2} Total Marks", student2_data['Total (1170)'].values[0])
            
            with col2:
                st.metric(f"{student1} Merit", student1_data['Merit'].values[0])
                st.metric(f"{student2} Merit", student2_data['Merit'].values[0])
            
            with col3:
                attendance1 = student1_attendance['Status_Code'].mean() * 100
                attendance2 = student2_attendance['Status_Code'].mean() * 100
                st.metric(f"{student1} Attendance", f"{attendance1:.1f}%")
                st.metric(f"{student2} Attendance", f"{attendance2:.1f}%")
            
            with col4:
                if 'Gender' in student1_data.columns and 'Gender' in student2_data.columns:
                    st.metric(f"{student1} Gender", student1_data['Gender'].values[0])
                    st.metric(f"{student2} Gender", student2_data['Gender'].values[0])
            
            # Comparison charts
            comparison_tab1, comparison_tab2 = st.tabs(["Performance Comparison", "Attendance Comparison"])
            
            with comparison_tab1:
                # Radar chart comparison
                fig = go.Figure()
                
                # Student 1 data
                values1 = [student1_data[f'{category}_Avg'].values[0] for category in assessment_categories]
                fig.add_trace(go.Scatterpolar(
                    r=values1,
                    theta=assessment_categories,
                    fill='toself',
                    name=student1
                ))
                
                # Student 2 data
                values2 = [student2_data[f'{category}_Avg'].values[0] for category in assessment_categories]
                fig.add_trace(go.Scatterpolar(
                    r=values2,
                    theta=assessment_categories,
                    fill='toself',
                    name=student2
                ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 30]
                        )),
                    title="Performance Comparison Radar Chart",
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with comparison_tab2:
                # Attendance comparison by day of week
                attendance_comparison = pd.DataFrame({
                    'Day': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
                    student1: [student1_attendance[student1_attendance['Day_Name'] == day]['Status_Code'].mean() for day in 
                              ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']],
                    student2: [student2_attendance[student2_attendance['Day_Name'] == day]['Status_Code'].mean() for day in 
                              ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']]
                })
                
                fig = px.bar(attendance_comparison, x='Day', y=[student1, student2], 
                            barmode='group', title="Attendance by Day of Week")
                st.plotly_chart(fig, use_container_width=True)
    
    # Tab 4: Attendance Analysis
    with tab4:
        st.header("Attendance Analysis")
        
        # Overall attendance by day of week
        attendance_by_day = attendance_df.groupby('Day_Name')['Status_Code'].mean().reset_index()
        # Order days properly
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        attendance_by_day['Day_Name'] = pd.Categorical(attendance_by_day['Day_Name'], categories=days_order, ordered=True)
        attendance_by_day = attendance_by_day.sort_values('Day_Name')
        
        fig = px.bar(attendance_by_day, x='Day_Name', y='Status_Code', 
                    title="Overall Attendance by Day of Week", labels={'Status_Code': 'Attendance Rate'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Attendance by student
        student_attendance = attendance_df.groupby('Name')['Status_Code'].mean().reset_index()
        student_attendance = student_attendance.sort_values('Status_Code', ascending=False)
        
        fig = px.bar(student_attendance, x='Name', y='Status_Code', 
                    title="Attendance Rate by Student", labels={'Status_Code': 'Attendance Rate'})
        fig.update_xaxis(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
        
        # Attendance vs Performance correlation
        performance_attendance = result_df.merge(student_attendance, on='Name')
        fig = px.scatter(performance_attendance, x='Status_Code', y='Total (1170)', 
                        trendline="ols", title="Attendance vs Performance Correlation",
                        labels={'Status_Code': 'Attendance Rate', 'Total (1170)': 'Total Marks'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Tab 5: Performance Trends
    with tab5:
        st.header("Performance Trends Analysis")
        
        # Distribution of marks by assessment category
        category_avgs = result_df[[f'{cat}_Avg' for cat in assessment_categories]]
        category_avgs.columns = assessment_categories
        
        fig = px.box(category_avgs, title="Distribution of Marks by Assessment Category")
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation heatmap between assessment categories
        corr_matrix = category_avgs.corr()
        fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", 
                       title="Correlation Between Assessment Categories")
        st.plotly_chart(fig, use_container_width=True)
        
        # Top and bottom performers analysis
        top5 = result_df.nlargest(5, 'Total (1170)')
        bottom5 = result_df.nsmallest(5, 'Total (1170)')
        
        comparison_data = pd.concat([top5, bottom5])
        comparison_data['Group'] = ['Top 5'] * 5 + ['Bottom 5'] * 5
        
        # Compare average marks in each category
        comparison_avgs = []
        for category in assessment_categories:
            top_avg = top5[f'{category}_Avg'].mean()
            bottom_avg = bottom5[f'{category}_Avg'].mean()
            comparison_avgs.append({'Category': category, 'Top 5': top_avg, 'Bottom 5': bottom_avg})
        
        comparison_df = pd.DataFrame(comparison_avgs)
        fig = px.bar(comparison_df, x='Category', y=['Top 5', 'Bottom 5'], barmode='group',
                    title="Performance Comparison: Top 5 vs Bottom 5 Students")
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
