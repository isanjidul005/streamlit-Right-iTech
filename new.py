import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import calendar
import os

# Set page configuration
st.set_page_config(
    page_title="Student Performance Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
def load_css():
    st.markdown("""
    <style>
    .main-header {font-size: 24px; color: #1f77b4; font-weight: bold;}
    .metric-card {background-color: #f0f2f6; padding: 15px; border-radius: 10px;}
    .stMetric {background-color: white; padding: 10px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);}
    </style>
    """, unsafe_allow_html=True)

load_css()

# Data loading and cleaning
@st.cache_data
def load_data():
    # Load datasets - using relative paths for Streamlit Cloud
    try:
        boys_att = pd.read_csv('three_b4_boys.xlsx - Sheet1.csv')
        girls_att = pd.read_csv('three_b4_girls.xlsx - Sheet1.csv')
        wmt_scores = pd.read_csv('three_result_combine.xlsx - Worksheet.csv')
    except FileNotFoundError:
        # Try alternative filenames (without spaces or special characters)
        try:
            boys_att = pd.read_csv('three_b4_boys.csv')
            girls_att = pd.read_csv('three_b4_girls.csv')
            wmt_scores = pd.read_csv('three_result_combine.csv')
        except FileNotFoundError:
            st.error("Data files not found. Please make sure the CSV files are in the same directory as the app.")
            st.stop()
    
    # Add gender column
    boys_att['Gender'] = 'Boy'
    girls_att['Gender'] = 'Girl'
    
    # Combine attendance data
    attendance = pd.concat([boys_att, girls_att], ignore_index=True)
    
    # Clean attendance data
    date_columns = [col for col in attendance.columns if col not in ['ID', 'Name', 'Gender']]
    attendance[date_columns] = attendance[date_columns].replace({'✔': 'Present', '✘': 'Absent'})
    
    # Melt attendance data to long format
    attendance_long = attendance.melt(
        id_vars=['ID', 'Name', 'Gender'], 
        value_vars=date_columns, 
        var_name='Date', 
        value_name='Status'
    )
    attendance_long['Date'] = pd.to_datetime(attendance_long['Date'], errors='coerce')
    attendance_long = attendance_long.dropna(subset=['Date'])
    
    # Clean score data
    wmt_scores = wmt_scores.replace('Ab', 0)
    score_columns = [col for col in wmt_scores.columns if col not in ['ID', 'Name']]
    for col in score_columns:
        wmt_scores[col] = pd.to_numeric(wmt_scores[col], errors='coerce').fillna(0)
    
    # Melt score data to long format
    wmt_long = wmt_scores.melt(
        id_vars=['ID', 'Name'], 
        value_vars=score_columns, 
        var_name='WMT', 
        value_name='Score'
    )
    
    # Merge data
    merged_df = pd.merge(wmt_long, attendance_long, on=['ID', 'Name'])
    return merged_df, wmt_scores, attendance_long

# Load the data
df, wmt_scores, attendance_long = load_data()

# Sidebar configuration
st.sidebar.title("Dashboard Controls")
section = st.sidebar.radio("Select Section", ["Class Overview", "Student Comparison", "Individual Student Dashboard"])

# Global filters
st.sidebar.subheader("Global Filters")
gender_filter = st.sidebar.selectbox("Select Class", ['All', 'Boys', 'Girls'])
min_date = df['Date'].min().date()
max_date = df['Date'].max().date()
date_range = st.sidebar.date_input("Select Date Range", [min_date, max_date])

# Filter data based on global filters
filtered_df = df.copy()
if gender_filter != 'All':
    filtered_df = filtered_df[filtered_df['Gender'] == gender_filter[:-1]]
if len(date_range) == 2:
    filtered_df = filtered_df[(filtered_df['Date'].dt.date >= date_range[0]) & 
                             (filtered_df['Date'].dt.date <= date_range[1])]

# Section 1: Class Overview Dashboard
if section == "Class Overview":
    st.header("Class Overview Dashboard")
    
    # WMT selector
    wmt_options = sorted([col for col in wmt_scores.columns if col not in ['ID', 'Name']])
    selected_wmt = st.selectbox("Select WMT", wmt_options)
    
    # Calculate metrics
    avg_score = filtered_df[filtered_df['WMT'] == selected_wmt]['Score'].mean()
    pass_rate = (filtered_df[(filtered_df['WMT'] == selected_wmt) & 
                           (filtered_df['Score'] >= 50)]['Score'].count() / 
                filtered_df[filtered_df['WMT'] == selected_wmt]['Score'].count()) * 100
    attendance_rate = (filtered_df[filtered_df['Status'] == 'Present'].shape[0] / 
                      filtered_df.shape[0]) * 100
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Average Score", f"{avg_score:.2f}")
    col2.metric("Pass Rate", f"{pass_rate:.2f}%")
    col3.metric("Attendance Rate", f"{attendance_rate:.2f}%")
    
    # Score distribution
    st.subheader("Score Distribution")
    fig_hist = px.histogram(filtered_df[filtered_df['WMT'] == selected_wmt], 
                           x="Score", nbins=20, title=f"Score Distribution for {selected_wmt}")
    st.plotly_chart(fig_hist, use_container_width=True)
    
    # Top and bottom performers
    st.subheader("Top and Bottom Performers")
    scores_wmt = filtered_df[filtered_df['WMT'] == selected_wmt]
    top_10 = scores_wmt.nlargest(10, 'Score')
    bottom_10 = scores_wmt.nsmallest(10, 'Score')
    
    fig_top = px.bar(top_10, x='Name', y='Score', title="Top 10 Performers")
    fig_bottom = px.bar(bottom_10, x='Name', y='Score', title="Bottom 10 Performers")
    
    col1, col2 = st.columns(2)
    col1.plotly_chart(fig_top, use_container_width=True)
    col2.plotly_chart(fig_bottom, use_container_width=True)
    
    # Attendance over time
    st.subheader("Attendance Rate Over Time")
    daily_attendance = filtered_df.groupby('Date')['Status'].apply(
        lambda x: (x == 'Present').sum() / x.count() * 100
    ).reset_index(name='Attendance Rate')
    
    fig_att = px.line(daily_attendance, x='Date', y='Attendance Rate', 
                     title="Daily Attendance Rate")
    st.plotly_chart(fig_att, use_container_width=True)
    
    # Early intervention flagging
    st.subheader("Students Needing Attention")
    # Calculate trends and attendance issues
    student_stats = filtered_df.groupby(['Name', 'WMT']).agg({
        'Score': 'mean',
        'Status': lambda x: (x == 'Present').mean() * 100
    }).reset_index()
    
    # Identify students with declining scores
    declining_students = []
    for name in student_stats['Name'].unique():
        student_data = student_stats[student_stats['Name'] == name].sort_values('WMT')
        if len(student_data) > 2 and student_data['Score'].is_monotonic_decreasing:
            declining_students.append(name)
    
    # Identify low attendance
    low_attendance = student_stats[student_stats['Status'] < 75]['Name'].unique()
    
    st.write("**Students with declining scores:**", ", ".join(declining_students[:5]))
    st.write("**Students with low attendance:**", ", ".join(low_attendance[:5]))

# Section 2: Student Comparison Dashboard
elif section == "Student Comparison":
    st.header("Student Comparison Dashboard")
    
    # Student selection
    student_list = sorted(filtered_df['Name'].unique())
    col1, col2 = st.columns(2)
    student1 = col1.selectbox("Select Student 1", student_list)
    student2 = col2.selectbox("Select Student 2", student_list)
    
    # Filter data for selected students
    student1_data = filtered_df[filtered_df['Name'] == student1]
    student2_data = filtered_df[filtered_df['Name'] == student2]
    
    # Score trend comparison
    st.subheader("Score Trend Comparison")
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=student1_data['WMT'].unique(), 
        y=student1_data.groupby('WMT')['Score'].mean(),
        name=student1
    ))
    fig_trend.add_trace(go.Scatter(
        x=student2_data['WMT'].unique(), 
        y=student2_data.groupby('WMT')['Score'].mean(),
        name=student2
    ))
    st.plotly_chart(fig_trend, use_container_width=True)
    
    # Attendance vs Score scatter plot
    st.subheader("Attendance vs Performance")
    scatter_data = filtered_df.groupby('Name').agg({
        'Score': 'mean',
        'Status': lambda x: (x == 'Present').mean() * 100
    }).reset_index()
    
    fig_scatter = px.scatter(scatter_data, x='Status', y='Score', hover_data=['Name'])
    # Highlight selected students
    fig_scatter.add_trace(go.Scatter(
        x=scatter_data[scatter_data['Name'] == student1]['Status'],
        y=scatter_data[scatter_data['Name'] == student1]['Score'],
        mode='markers', marker=dict(size=15, color='red'), name=student1
    ))
    fig_scatter.add_trace(go.Scatter(
        x=scatter_data[scatter_data['Name'] == student2]['Status'],
        y=scatter_data[scatter_data['Name'] == student2]['Score'],
        mode='markers', marker=dict(size=15, color='blue'), name=student2
    ))
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Performance table
    st.subheader("Performance Comparison")
    comparison_data = {
        'Metric': ['Average Score', 'Median Score', 'Attendance Rate'],
        student1: [
            student1_data['Score'].mean(),
            student1_data['Score'].median(),
            (student1_data['Status'] == 'Present').mean() * 100
        ],
        student2: [
            student2_data['Score'].mean(),
            student2_data['Score'].median(),
            (student2_data['Status'] == 'Present').mean() * 100
        ]
    }
    st.table(pd.DataFrame(comparison_data))

# Section 3: Individual Student Dashboard
else:
    st.header("Individual Student Dashboard")
    
    # Student selection
    student_list = sorted(filtered_df['Name'].unique())
    selected_student = st.selectbox("Select Student", student_list)
    
    # Filter data for selected student
    student_data = filtered_df[filtered_df['Name'] == selected_student]
    
    # Calculate metrics
    avg_score = student_data['Score'].mean()
    attendance_rate = (student_data['Status'] == 'Present').mean() * 100
    total_score = student_data['Score'].sum()
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Average Score", f"{avg_score:.2f}")
    col2.metric("Attendance Rate", f"{attendance_rate:.2f}%")
    col3.metric("Total Score", f"{total_score:.2f}")
    
    # Score trend
    st.subheader("Score Trend Over Time")
    score_trend = student_data.groupby('WMT')['Score'].mean().reset_index()
    fig_score = px.line(score_trend, x='WMT', y='Score', title="WMT Score Trend")
    st.plotly_chart(fig_score, use_container_width=True)
    
    # Attendance heatmap
    st.subheader("Attendance Heatmap")
    student_att = student_data.copy()
    student_att['Day'] = student_att['Date'].dt.day
    student_att['Month'] = student_att['Date'].dt.month
    student_att['Year'] = student_att['Date'].dt.year
    student_att['Status_num'] = student_att['Status'].map({'Present': 1, 'Absent': 0})
    
    heatmap_data = student_att.pivot_table(
        values='Status_num', index='Day', columns='Month', aggfunc='mean'
    )
    fig_heatmap = px.imshow(heatmap_data, title="Monthly Attendance Pattern")
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Subject-wise performance (assuming WMT names contain subject information)
    st.subheader("Subject-wise Performance")
    # This would require additional data mapping - placeholder implementation
    subject_data = student_data.copy()
    subject_data['Subject'] = subject_data['WMT'].str.split('_').str[0]  # Example extraction
    subject_avg = subject_data.groupby('Subject')['Score'].mean().reset_index()
    fig_subject = px.bar(subject_avg, x='Subject', y='Score', title="Average Score by Subject")
    st.plotly_chart(fig_subject, use_container_width=True)
