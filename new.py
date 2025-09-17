import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# --------------------
# CONFIG
# --------------------
st.set_page_config(page_title="Class 3 Dashboard", layout="wide", initial_sidebar_state="expanded")

ATTENDANCE_PATH = "/mnt/data/clean_attendance.csv"
MARKS_PATH = "/mnt/data/clean_marks.csv"

# --------------------
# HELPERS
# --------------------
@st.cache_data
def load_data(att_path=ATTENDANCE_PATH, marks_path=MARKS_PATH):
    att = pd.read_csv(att_path)
    marks = pd.read_csv(marks_path)
    return att, marks


def compute_attendance_summary(att):
    grp = (
        att.groupby(["ID", "Roll", "Name", "Gender"])
        .agg(Total_Days=("Date", "count"), Days_Present=("Status", lambda x: (x == "Present").sum()))
        .reset_index()
    )
    grp["Attendance_%"] = (grp["Days_Present"] / grp["Total_Days"] * 100).round(2)
    return grp


def compute_marks_summary(marks):
    # Average per student across all subjects and exams (ignoring NaN)
    avg = (
        marks.groupby(["ID", "Roll", "Name"]).agg(Average_Score=("Marks", "mean")).reset_index()
    )
    return avg


def pivot_subject_scores(marks):
    # average marks per subject per student
    subj = (
        marks.groupby(["ID", "Roll", "Name", "Subject"]).agg(Avg_Subject_Score=("Marks", "mean")).reset_index()
    )
    subj_pivot = subj.pivot_table(index=["ID", "Roll", "Name"], columns="Subject", values="Avg_Subject_Score").reset_index()
    return subj_pivot


def student_time_series(marks, subject_filter=None):
    # Create an exam-time series by converting ExamNumber to numeric if possible
    m = marks.copy()
    m["ExamNumber_n"] = pd.to_numeric(m["ExamNumber"], errors="coerce")
    if subject_filter:
        m = m[m["Subject"] == subject_filter]
    ts = m.groupby(["ID", "Roll", "Name", "ExamNumber_n"]).agg(Avg=("Marks", "mean")).reset_index()
    return ts


# --------------------
# UI: Sidebar
# --------------------
st.sidebar.title("Controls")
att, marks = load_data()

student_list = sorted(att["Name"].unique())
all_subjects = sorted(marks["Subject"].dropna().unique())

# global filters
selected_gender = st.sidebar.multiselect("Gender", options=sorted(att["Gender"].unique()), default=sorted(att["Gender"].unique()))
min_attendance = st.sidebar.slider("Minimum attendance %", 0, 100, 0)

# clustering options
n_clusters = st.sidebar.slider("Clustering: number of clusters", 2, 8, 3)
run_clustering = st.sidebar.checkbox("Run clustering", value=True)

# --------------------
# MAIN LAYOUT: Tabs
# --------------------
st.title("🌟 Class 3 — Interactive Performance & Attendance Dashboard")

tabs = st.tabs(["Class Overview", "Student Profile", "Compare Students", "Subject Analysis", "Time Series", "Clustering"])

# Precompute summaries
att_summary = compute_attendance_summary(att)
marks_summary = compute_marks_summary(marks)
subj_pivot = pivot_subject_scores(marks)

# join attendance and marks
student_summary = pd.merge(att_summary, marks_summary, on=["ID", "Roll", "Name"], how="left")

# Apply gender and attendance filters
student_summary = student_summary[student_summary["Gender"].isin(selected_gender)]
student_summary = student_summary[student_summary["Attendance_%"] >= min_attendance]

# --- Tab 1: Class Overview ---
with tabs[0]:
    st.header("Class Overview")
    col1, col2, col3 = st.columns([3,2,2])

    # KPIs
    with col1:
        st.subheader("KPIs")
        st.metric("Students (visible)", student_summary.shape[0])
        st.metric("Average Attendance %", f"{student_summary['Attendance_%'].mean():.2f}")
        st.metric("Average Score", f"{student_summary['Average_Score'].mean():.2f}")

    with col2:
        # Attendance distribution
        fig = px.histogram(student_summary, x="Attendance_%", nbins=20, title="Attendance Distribution")
        st.plotly_chart(fig, use_container_width=True)

    with col3:
        # Score distribution
        fig = px.histogram(student_summary, x="Average_Score", nbins=20, title="Average Score Distribution")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Attendance vs Performance")
    color_by = st.selectbox("Color by", options=["Gender", "None"], index=0)
    fig = px.scatter(student_summary, x="Attendance_%", y="Average_Score", hover_data=["Name", "Roll"], color=(student_summary["Gender"] if color_by=="Gender" else None), trendline="ols", title="Attendance vs Average Score")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top / Bottom Performers")
    topk = st.slider("How many top/bottom", 1, 20, 5)
    st.write("Top performers")
    st.dataframe(student_summary.nlargest(topk, "Average_Score")[['Name','Roll','Attendance_%','Average_Score']])
    st.write("Bottom performers")
    st.dataframe(student_summary.nsmallest(topk, "Average_Score")[['Name','Roll','Attendance_%','Average_Score']])

# --- Tab 2: Student Profile ---
with tabs[1]:
    st.header("Student Profile")
    selected_student = st.selectbox("Select student", options=student_list)
    student_id = att[att['Name']==selected_student]['ID'].unique()[0]

    # Basic info
    info = student_summary[student_summary['Name']==selected_student]
    if info.empty:
        st.warning("Selected student does not match current filters. Try clearing filters or selecting another student.")
    else:
        st.subheader(selected_student)
        c1, c2, c3 = st.columns(3)
        c1.metric("Attendance %", f"{info['Attendance_%'].values[0]:.2f}")
        c2.metric("Average Score", f"{info['Average_Score'].values[0]:.2f}")
        c3.metric("Total Days", int(info['Total_Days'].values[0]))

        st.markdown("#### Subject Radar Chart")
        # prepare radar data
        row = subj_pivot[subj_pivot['Name']==selected_student]
        if row.shape[0]==0:
            st.info("No subject-level marks available for this student.")
        else:
            labels = [c for c in row.columns if c not in ['ID','Roll','Name']]
            values = row[labels].values.flatten().tolist()
            # fill nan with zeros to render radar, but keep tooltip values later
            values = [0 if np.isnan(v) else v for v in values]
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(r=values, theta=labels, fill='toself', name=selected_student))
            fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Recent Exam Performance")
        recent = marks[marks['ID']==student_id].sort_values(['Subject','ExamNumber'])
        if recent.empty:
            st.info("No marks available for this student.")
        else:
            st.dataframe(recent[['Subject','ExamType','ExamNumber','Marks','FullMarks','WasAbsent']].head(20))

# --- Tab 3: Compare Students ---
with tabs[2]:
    st.header("Compare Students")
    comp_students = st.multiselect("Select up to 4 students", options=student_list, default=student_list[:2], max_selections=4)
    if len(comp_students) < 1:
        st.info("Pick at least one student to compare")
    else:
        comp_ids = att[att['Name'].isin(comp_students)]['ID'].unique()
        comp_marks = marks[marks['ID'].isin(comp_ids)]
        # line chart over exam numbers faceted by subject
        fig = px.line(comp_marks, x='ExamNumber', y='Marks', color='Name', facet_col='Subject', markers=True, title='Student Exam Trajectories by Subject')
        st.plotly_chart(fig, use_container_width=True)

# --- Tab 4: Subject Analysis ---
with tabs[3]:
    st.header("Subject Analysis")
    subject = st.selectbox("Pick subject", options=all_subjects)
    subj_df = marks[marks['Subject']==subject]

    st.subheader(f"Distribution in {subject}")
    fig = px.histogram(subj_df, x='Marks', nbins=20)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top students in subject")
    top_subj = subj_df.groupby(['ID','Roll','Name']).agg(Avg=("Marks","mean")).reset_index().nlargest(10,'Avg')
    st.dataframe(top_subj)

# --- Tab 5: Time Series ---
with tabs[4]:
    st.header("Time Series — Progress Over Exams")
    sub = st.selectbox("Subject for time-series", options=[None]+all_subjects)
    ts = student_time_series(marks, subject_filter=sub)
    # facet grid small multiples per student might be heavy; offer sample or single student
    choice = st.radio("Plot type", options=['Class trend','Single student small multiples'])
    if choice=='Class trend':
        fig = px.line(ts.groupby('ExamNumber_n').agg(Avg=('Avg','mean')).reset_index(), x='ExamNumber_n', y='Avg', markers=True, title='Class Average Over Exam Number')
        st.plotly_chart(fig, use_container_width=True)
    else:
        student_small = st.multiselect('Pick students (up to 12)', options=student_list, default=student_list[:6], max_selections=12)
        small_ts = ts[ts['Name'].isin(student_small)]
        fig = px.line(small_ts, x='ExamNumber_n', y='Avg', color='Name', facet_col='Name', facet_col_wrap=4, markers=True)
        st.plotly_chart(fig, use_container_width=True)

# --- Tab 6: Clustering ---
with tabs[5]:
    st.header("Behavioral & Academic Clustering")
    # prepare features: attendance % and average score + subject averages
    features = student_summary[['ID','Roll','Name','Attendance_%','Average_Score']].merge(subj_pivot, on=['ID','Roll','Name'], how='left')
    feat = features.fillna(0).set_index('ID')
    feat_numeric = feat.select_dtypes(include=[np.number]).drop(columns=['Roll'])

    scaler = StandardScaler()
    X = scaler.fit_transform(feat_numeric)

    if run_clustering:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(X)
        feat['Cluster'] = labels

        # show cluster counts
        st.subheader('Cluster composition')
        st.dataframe(feat['Cluster'].value_counts().rename_axis('Cluster').reset_index(name='Count'))

        # scatter plot Attendance vs Score colored by cluster
        plot_df = feat.reset_index().merge(student_summary.set_index('ID')[['Name','Gender']], left_index=True, right_index=True, how='left')
        fig = px.scatter(plot_df, x='Attendance_%', y='Average_Score', color='Cluster', hover_data=['Name'], symbol='Gender', title='Clusters: Attendance vs Score')
        st.plotly_chart(fig, use_container_width=True)

        st.markdown('### Cluster examples')
        sel_cluster = st.selectbox('Pick cluster to inspect', options=sorted(feat['Cluster'].unique()))
        st.dataframe(plot_df[plot_df['Cluster']==sel_cluster][['Name','Roll','Attendance_%','Average_Score']].sort_values('Average_Score', ascending=False))

st.markdown("---")
st.caption("Dashboard generated by Streamlit. Customize further as needed.")
