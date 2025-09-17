import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date
import calendar
import io

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

# Function to read both CSV and Excel files
def read_data_file(file):
    """Read CSV or Excel files and return DataFrame"""
    try:
        # Streamlit file_uploader provides file-like object
        file_extension = file.name.split('.')[-1].lower()
        if file_extension == 'csv':
            # Specify skiprows to read the correct header row
            return pd.read_csv(file, skiprows=1)
        elif file_extension in ['xlsx', 'xls']:
            # Specify skiprows to read the correct header row
            return pd.read_excel(file, skiprows=1)
        else:
            st.error(f"Unsupported file format: {file.name}")
            return None
    except Exception as e:
        st.error(f"Error reading file {file.name}: {str(e)}")
        return None

# File upload and data processing
@st.cache_data
def process_uploaded_files(attendance_files, score_file):
    # Process attendance files
    attendance_dfs = []
    
    for file in attendance_files:
        df = read_data_file(file)
        if df is None:
            continue
            
        # Convert all column names to strings
        df.columns = df.columns.astype(str)

        # Standardize column names based on file format
        if 'ID' not in df.columns or 'Name' not in df.columns:
            st.warning(f"Could not find ID/Name columns in {file.name}. Using first three columns.")
            if len(df.columns) >= 3:
                # Assuming ID, Roll, and Name are the first three columns
                df.rename(columns={df.columns[0]: 'ID', df.columns[1]: 'Roll', df.columns[2]: 'Name'}, inplace=True)
            else:
                st.error(f"File {file.name} doesn't have enough columns")
                continue
        
        # Try to detect gender from filename or add unknown
        file_name = file.name.lower()
        if 'boy' in file_name or 'male' in file_name:
            df['Gender'] = 'Boy'
        elif 'girl' in file_name or 'female' in file_name:
            df['Gender'] = 'Girl'
        else:
            df['Gender'] = 'Unknown'
            
        attendance_dfs.append(df)
    
    if not attendance_dfs:
        st.error("No valid attendance files were processed")
        return None, None, None
    
    # Combine all attendance data
    attendance = pd.concat(attendance_dfs, ignore_index=True)
    
    # Clean attendance data - identify date columns
    non_date_columns = ['ID', 'Name', 'Gender', 'Roll']
    date_columns = [col for col in attendance.columns if col not in non_date_columns]

    # Check if date_columns is empty before proceeding
    if not date_columns:
        st.error("Could not identify date columns in attendance files. Please ensure columns represent dates.")
        return None, None, None
    
    # Melt attendance data to long format
    attendance_long = attendance.melt(
        id_vars=['ID', 'Roll', 'Name', 'Gender'],
        value_vars=date_columns,
        var_name='Date',
        value_name='Status'
    )
    
    # Clean the 'Status' column from values like '✔ 10' or '✘ 1'
    attendance_long['Status'] = attendance_long['Status'].astype(str).str.strip()
    attendance_long['Status'] = attendance_long['Status'].str.extract(r'(✔|✘)')
    attendance_long['Status'] = attendance_long['Status'].replace({'✔': 'Present', '✘': 'Absent'})

    # Convert the 'Date' column to a proper datetime format
    attendance_long['Date'] = pd.to_datetime(attendance_long['Date'], format='%b %d %a', errors='coerce')
    
    attendance_long.dropna(subset=['Date'], inplace=True)
    
    # Process score file
    score_df = read_data_file(score_file)
    if score_df is None:
        return None, None, None

    # Convert all score column names to strings as well
    score_df.columns = score_df.columns.astype(str)
    
    # Explicitly check for ID and Name.
    if 'ID' not in score_df.columns or 'Name' not in score_df.columns:
        st.warning(f"Could not find ID/Name columns in score file. Renaming the first and third columns.")
        if len(score_df.columns) >= 3:
            # Assuming ID and Name are the first and third columns
            score_df.rename(columns={score_df.columns[0]: 'ID', score_df.columns[2]: 'Name'}, inplace=True)
        else:
            st.error("Score file doesn't have enough columns")
            return None, None, None
            
    # Drop "Total" and "Merit" columns as they are not needed for melting
    score_df = score_df.drop(columns=[col for col in score_df.columns if 'Total' in col or 'Merit' in col], errors='ignore')

    # Identify score columns (non-ID and non-Name columns)
    score_columns = [col for col in score_df.columns if col not in ['ID', 'Name', 'Roll']]
    
    # Check if score_columns is empty before proceeding
    if not score_columns:
        st.error("Could not identify score columns. Please ensure your score file has score data.")
        return None, None, None

    # Clean score data and convert to numeric
    for col in score_columns:
        score_df[col] = pd.to_numeric(
            score_df[col].astype(str).str.extract(r'(\d+\.?\d*)').fillna('0'), # Extract numeric values
            errors='coerce'
        ).fillna(0)
    
    # Melt score data to long format
    wmt_long = score_df.melt(
        id_vars=['ID', 'Name'],
        value_vars=score_columns,
        var_name='WMT',
        value_name='Score'
    )
    
    # Merge data
    merged_df = pd.merge(wmt_long, attendance_long, on=['ID', 'Name'], how='outer')
    
    return merged_df, score_df, attendance_long

# Main app
def main():
    st.title("📊 Student Performance Dashboard")
    
    # File upload section
    st.sidebar.header("Upload Files")
    
    # Upload attendance files
    attendance_files = st.sidebar.file_uploader(
        "Upload Attendance Files",
        type=['csv', 'xlsx', 'xls'],
        accept_multiple_files=True,
        help="Upload one or more attendance files (CSV or Excel)"
    )
    
    # Upload score file
    score_file = st.sidebar.file_uploader(
        "Upload WMT Scores File",
        type=['csv', 'xlsx', 'xls'],
        help="Upload the WMT scores file (CSV or Excel)"
    )
    
    # Check if files are uploaded
    if not attendance_files:
        st.info("👆 Please upload attendance files to begin")
        st.stop()
    
    if not score_file:
        st.info("👆 Please upload WMT scores file to begin")
        st.stop()
    
    # Process files
    with st.spinner("Processing your files..."):
        try:
            df, wmt_scores, attendance_long = process_uploaded_files(attendance_files, score_file)
            
            if df is None:
                st.error("Failed to process files. Please check your file formats.")
                st.stop()
                
        except Exception as e:
            st.error(f"Error processing files: {str(e)}")
            st.stop()
    
    # Show success message
    st.success(f"✅ Successfully processed {len(attendance_files)} attendance files and 1 score file")
    
    # Display data preview
    with st.expander("View Raw Data Preview"):
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Attendance Data Preview:**")
            st.dataframe(attendance_long.head())
        with col2:
            st.write("**Score Data Preview:**")
            st.dataframe(wmt_scores.head())
    
    # Sidebar configuration
    st.sidebar.header("Dashboard Controls")
    section = st.sidebar.radio("Select Section", ["Class Overview", "Student Comparison", "Individual Student Dashboard"])
    
    # Global filters
    st.sidebar.subheader("Global Filters")
    gender_options = ['All'] + list(df['Gender'].unique())
    gender_filter = st.sidebar.selectbox("Select Gender", gender_options)
    
    # Handle date range with proper validation
    try:
        min_date = df['Date'].min().date()
        max_date = df['Date'].max().date()
        
        # Ensure dates are valid (not NaT)
        if pd.isna(min_date) or pd.isna(max_date):
            st.error("Invalid date range detected. Please check your attendance data.")
            min_date = date.today()
            max_date = date.today()
    except:
        min_date = date.today()
        max_date = date.today()
    
    date_range = st.sidebar.date_input("Select Date Range", [min_date, max_date])
    
    # Filter data based on global filters
    filtered_df = df.copy()
    if gender_filter != 'All':
        filtered_df = filtered_df[filtered_df['Gender'] == gender_filter]
    if len(date_range) == 2:
        filtered_df = filtered_df[(filtered_df['Date'].dt.date >= date_range[0]) & 
                                 (filtered_df['Date'].dt.date <= date_range[1])]
    
    # Section 1: Class Overview Dashboard
    if section == "Class Overview":
        st.header("Class Overview Dashboard")
        
        # WMT selector
        wmt_options = sorted(filtered_df['WMT'].unique())
        if not wmt_options:
            st.error("No WMT columns found in the score data.")
            st.stop()
            
        selected_wmt = st.selectbox("Select WMT", wmt_options)
        
        # Calculate metrics
        wmt_data = filtered_df[filtered_df['WMT'] == selected_wmt]
        if len(wmt_data) == 0:
            st.warning(f"No data available for {selected_wmt}")
            avg_score = 0
            pass_rate = 0
        else:
            avg_score = wmt_data['Score'].mean()
            pass_count = wmt_data[wmt_data['Score'] >= 50]['Score'].count()
            pass_rate = (pass_count / len(wmt_data)) * 100 if len(wmt_data) > 0 else 0
        
        attendance_rate = (filtered_df[filtered_df['Status'] == 'Present'].shape[0] / 
                          filtered_df.shape[0]) * 100 if filtered_df.shape[0] > 0 else 0
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Average Score", f"{avg_score:.2f}")
        col2.metric("Pass Rate", f"{pass_rate:.2f}%")
        col3.metric("Attendance Rate", f"{attendance_rate:.2f}%")
        
        # Score distribution
        st.subheader("Score Distribution")
        if len(wmt_data) > 0:
            fig_hist = px.histogram(wmt_data, x="Score", nbins=20, title=f"Score Distribution for {selected_wmt}")
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.info("No score data available for the selected WMT")
        
        # Top and bottom performers
        st.subheader("Top and Bottom Performers")
        if len(wmt_data) > 0:
            # Drop duplicates to ensure each student is counted once per WMT
            wmt_data_unique = wmt_data.drop_duplicates(subset=['Name'])
            
            top_10 = wmt_data_unique.nlargest(min(10, len(wmt_data_unique)), 'Score')
            bottom_10 = wmt_data_unique.nsmallest(min(10, len(wmt_data_unique)), 'Score')
            
            fig_top = px.bar(top_10, x='Name', y='Score', title="Top Performers")
            fig_bottom = px.bar(bottom_10, x='Name', y='Score', title="Bottom Performers")
            
            col1, col2 = st.columns(2)
            col1.plotly_chart(fig_top, use_container_width=True)
            col2.plotly_chart(fig_bottom, use_container_width=True)
        else:
            st.info("No score data available for performance comparison")
        
        # Attendance over time
        st.subheader("Attendance Rate Over Time")
        if len(filtered_df) > 0:
            daily_attendance = filtered_df.groupby('Date')['Status'].apply(
                lambda x: (x == 'Present').sum() / x.count() * 100 if x.count() > 0 else 0
            ).reset_index(name='Attendance Rate')
            
            fig_att = px.line(daily_attendance, x='Date', y='Attendance Rate', 
                             title="Daily Attendance Rate")
            st.plotly_chart(fig_att, use_container_width=True)
        else:
            st.info("No attendance data available")
        
        # Early intervention flagging
        st.subheader("Students Needing Attention")
        if len(filtered_df) > 0:
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
            
            if len(declining_students) > 0:
                st.write("**Students with declining scores:**", ", ".join(declining_students[:5]))
            else:
                st.write("No students with declining scores found.")
                
            if len(low_attendance) > 0:
                st.write("**Students with low attendance:**", ", ".join(low_attendance[:5]))
            else:
                st.write("No students with low attendance found.")
        else:
            st.info("No data available for student analysis")
    
    # Section 2: Student Comparison Dashboard
    elif section == "Student Comparison":
        st.header("Student Comparison Dashboard")
        
        # Student selection
        student_list = sorted(filtered_df['Name'].unique())
        if len(student_list) == 0:
            st.error("No students found in the filtered data")
            st.stop()
            
        col1, col2 = st.columns(2)
        student1 = col1.selectbox("Select Student 1", student_list)
        student2 = col2.selectbox("Select Student 2", student_list)
        
        # Filter data for selected students
        student1_data = filtered_df[filtered_df['Name'] == student1]
        student2_data = filtered_df[filtered_df['Name'] == student2]
        
        # Score trend comparison
        st.subheader("Score Trend Comparison")
        if len(student1_data) > 0 or len(student2_data) > 0:
            fig_trend = go.Figure()
            
            if len(student1_data) > 0:
                student1_scores = student1_data.groupby('WMT')['Score'].mean().reset_index()
                fig_trend.add_trace(go.Scatter(
                    x=student1_scores['WMT'],
                    y=student1_scores['Score'],
                    name=student1
                ))
            
            if len(student2_data) > 0:
                student2_scores = student2_data.groupby('WMT')['Score'].mean().reset_index()
                fig_trend.add_trace(go.Scatter(
                    x=student2_scores['WMT'],
                    y=student2_scores['Score'],
                    name=student2
                ))
            
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.info("No score data available for comparison")
        
        # Attendance vs Score scatter plot
        st.subheader("Attendance vs Performance")
        if len(filtered_df) > 0:
            scatter_data = filtered_df.groupby('Name').agg({
                'Score': 'mean',
                'Status': lambda x: (x == 'Present').mean() * 100
            }).reset_index()
            
            fig_scatter = px.scatter(scatter_data, x='Status', y='Score', hover_data=['Name'])
            # Highlight selected students
            if student1 in scatter_data['Name'].values:
                fig_scatter.add_trace(go.Scatter(
                    x=scatter_data[scatter_data['Name'] == student1]['Status'],
                    y=scatter_data[scatter_data['Name'] == student1]['Score'],
                    mode='markers', marker=dict(size=15, color='red'), name=student1
                ))
            
            if student2 in scatter_data['Name'].values:
                fig_scatter.add_trace(go.Scatter(
                    x=scatter_data[scatter_data['Name'] == student2]['Status'],
                    y=scatter_data[scatter_data['Name'] == student2]['Score'],
                    mode='markers', marker=dict(size=15, color='blue'), name=student2
                ))
            
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.info("No data available for scatter plot")
        
        # Performance table
        st.subheader("Performance Comparison")
        comparison_data = {
            'Metric': ['Average Score', 'Median Score', 'Attendance Rate'],
            student1: [
                student1_data['Score'].mean() if len(student1_data) > 0 else 0,
                student1_data['Score'].median() if len(student1_data) > 0 else 0,
                (student1_data['Status'] == 'Present').mean() * 100 if len(student1_data) > 0 else 0
            ],
            student2: [
                student2_data['Score'].mean() if len(student2_data) > 0 else 0,
                student2_data['Score'].median() if len(student2_data) > 0 else 0,
                (student2_data['Status'] == 'Present').mean() * 100 if len(student2_data) > 0 else 0
            ]
        }
        st.table(pd.DataFrame(comparison_data))
    
    # Section 3: Individual Student Dashboard
    else:
        st.header("Individual Student Dashboard")
        
        # Student selection
        student_list = sorted(filtered_df['Name'].unique())
        if len(student_list) == 0:
            st.error("No students found in the filtered data")
            st.stop()
            
        selected_student = st.selectbox("Select Student", student_list)
        
        # Filter data for selected student
        student_data = filtered_df[filtered_df['Name'] == selected_student]
        
        if len(student_data) == 0:
            st.warning(f"No data available for {selected_student}")
            st.stop()
        
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
        if len(score_trend) > 0:
            fig_score = px.line(score_trend, x='WMT', y='Score', title="WMT Score Trend")
            st.plotly_chart(fig_score, use_container_width=True)
        else:
            st.info("No score trend data available")
        
        # Attendance heatmap
        st.subheader("Attendance Heatmap")
        student_att = student_data.copy()
        student_att['Day'] = student_att['Date'].dt.day
        student_att['Month'] = student_att['Date'].dt.month
        student_att['Year'] = student_att['Date'].dt.year
        student_att['Status_num'] = student_att['Status'].map({'Present': 1, 'Absent': 0})
        
        if len(student_att) > 0:
            heatmap_data = student_att.pivot_table(
                values='Status_num', index='Day', columns='Month', aggfunc='mean'
            )
            fig_heatmap = px.imshow(heatmap_data, title="Monthly Attendance Pattern")
            st.plotly_chart(fig_heatmap, use_container_width=True)
        else:
            st.info("No attendance data available for heatmap")
        
        # Subject-wise performance
        st.subheader("Subject-wise Performance")
        subject_data = student_data.copy()
        # Clean 'WMT' column to extract subject, e.g., 'WMT W1 [30]' -> 'W1'
        subject_data['Subject'] = subject_data['WMT'].str.extract(r'(W\d+)')
        subject_avg = subject_data.groupby('Subject')['Score'].mean().reset_index()
        
        if len(subject_avg) > 0:
            fig_subject = px.bar(subject_avg, x='Subject', y='Score', title="Average Score by Subject")
            st.plotly_chart(fig_subject, use_container_width=True)
        else:
            st.info("No subject-wise data available")

if __name__ == "__main__":
    main()
