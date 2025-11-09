"""
Social Media AI Strategy Dashboard
Beautiful, interactive web app for predictions and insights
"""

import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np

# Page configuration
st.set_page_config(
    page_title="AI Social Media Strategy Predictor",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 0rem;
    }
    .stAlert {
        background-color: #f0f2f6;
        border-left: 5px solid #4CAF50;
    }
    h1 {
        color: #1f77b4;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# Load the trained model
@st.cache_resource
def load_model():
    try:
        return joblib.load('social_media_model.pkl') 
    except FileNotFoundError:
        st.error("❌ Model not found! Please run 'python train_model.py' first.")
        st.stop()

@st.cache_resource
def load_insights():
    try:
        return joblib.load('insights.pkl') 
    except FileNotFoundError:
        return None

# Load model and data
model_package = load_model()
insights = load_insights()

classifier = model_package['classifier']
regressor = model_package['regressor']
encoders = model_package['encoders']
feature_cols = model_package['feature_columns']
training_data = model_package['training_data']

# Title and header
st.title("🎯 AI-Powered Social Media Strategy Predictor")
st.markdown("### Transform your social media analytics into actionable insights")
st.markdown("---")

# Sidebar for inputs
st.sidebar.header("📊 Input Your Content Details")
st.sidebar.markdown("Fill in the details below to get AI predictions")

# User inputs
platform = st.sidebar.selectbox(
    "Platform",
    options=list(encoders['platform'].classes_),
    help="Which social media platform are you posting on?"
)

content_type = st.sidebar.selectbox(
    "Content Type",
    options=list(encoders['content_type'].classes_),
    help="What type of content are you creating?"
)

category = st.sidebar.selectbox(
    "Content Category",
    options=list(encoders['category'].classes_),
    help="What is the main category of your content?"
)

language = st.sidebar.selectbox(
    "Language",
    options=list(encoders['language'].classes_),
    help="What language is your content in?"
)

content_length = st.sidebar.slider(
    "Content Length (characters/seconds)",
    min_value=10,
    max_value=600,
    value=150,
    help="Length of your content"
)

follower_count = st.sidebar.number_input(
    "Follower Count",
    min_value=0,
    value=50000,
    step=1000,
    help="Your current follower count"
)

age_group = st.sidebar.selectbox(
    "Target Audience Age",
    options=list(encoders['age'].classes_)
)

gender = st.sidebar.selectbox(
    "Target Audience Gender",
    options=list(encoders['gender'].classes_)
)

is_sponsored = st.sidebar.checkbox("Is this sponsored content?")

st.sidebar.markdown("---")

# Time selection
st.sidebar.subheader("⏰ Posting Schedule")
posting_date = st.sidebar.date_input("Posting Date", datetime.now())
posting_time = st.sidebar.time_input("Posting Time", datetime.now().time())

hour = posting_time.hour
day_of_week = posting_date.weekday()
month = posting_date.month

# Predict button
predict_button = st.sidebar.button("🚀 Get AI Predictions", type="primary", use_container_width=True)

# Main content area
if predict_button:
    # Prepare input data
    input_data = pd.DataFrame({
        'platform_encoded': [encoders['platform'].transform([platform])[0]],
        'content_type_encoded': [encoders['content_type'].transform([content_type])[0]],
        'category_encoded': [encoders['category'].transform([category])[0]],
        'language_encoded': [encoders['language'].transform([language])[0]],
        'content_length': [content_length],
        'hour': [hour],
        'day_of_week': [day_of_week],
        'month': [month],
        'follower_count': [follower_count],
        'age_encoded': [encoders['age'].transform([age_group])[0]],
        'gender_encoded': [encoders['gender'].transform([gender])[0]],
        'sponsored_encoded': [1 if is_sponsored else 0]
    })
    
    # Make predictions
    performance_pred = classifier.predict(input_data)[0]
    engagement_pred = regressor.predict(input_data)[0]
    
    # Calculate potential reach
    potential_views = follower_count * 0.15  # Avg 15% reach
    potential_likes = potential_views * (engagement_pred / 100) * 0.6
    potential_comments = potential_views * (engagement_pred / 100) * 0.2
    potential_shares = potential_views * (engagement_pred / 100) * 0.2
    
    # Display predictions
    st.success("✅ AI Analysis Complete!")
    
    # Main metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Performance Level",
            performance_pred,
            delta="Predicted",
            delta_color="off"
        )
    
    with col2:
        st.metric(
            "Engagement Rate",
            f"{engagement_pred:.2f}%",
            delta=f"{engagement_pred - insights['avg_engagement']:.2f}% vs avg" if insights else None
        )
    
    with col3:
        st.metric(
            "Expected Views",
            f"{int(potential_views):,}",
            delta="Estimated"
        )
    
    with col4:
        st.metric(
            "Expected Interactions",
            f"{int(potential_likes + potential_comments + potential_shares):,}",
            delta="Total"
        )
    
    st.markdown("---")
    
    # Detailed breakdown
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("📈 Expected Performance Breakdown")
        
        breakdown_df = pd.DataFrame({
            'Metric': ['Views', 'Likes', 'Comments', 'Shares'],
            'Expected Count': [
                int(potential_views),
                int(potential_likes),
                int(potential_comments),
                int(potential_shares)
            ]
        })
        
        fig = px.bar(
            breakdown_df,
            x='Metric',
            y='Expected Count',
            color='Metric',
            title='Predicted Engagement Breakdown',
            text='Expected Count',
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig.update_traces(texttemplate='%{text:,}', textposition='outside')
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col_right:
        st.subheader("🎯 AI Recommendations")
        
        # Generate recommendations
        recommendations = []
        
        if engagement_pred < insights['avg_engagement']:
            recommendations.append(f"⚠️ Your predicted engagement ({engagement_pred:.2f}%) is below average ({insights['avg_engagement']:.2f}%). Consider optimizing your content.")
        else:
            recommendations.append(f"✅ Great! Your content is predicted to perform above average!")
        
        # Time recommendation
        if hour != insights['best_hour']:
            recommendations.append(f"⏰ Best posting time: {insights['best_hour']}:00 (You selected: {hour}:00)")
        else:
            recommendations.append(f"✅ Perfect timing! {hour}:00 is the optimal posting time.")
        
        # Platform recommendation
        if platform != insights['best_platform']:
            recommendations.append(f"📱 Consider posting on {insights['best_platform']} for higher engagement")
        
        # Content type recommendation
        if content_type != insights['best_content_type']:
            recommendations.append(f"🎬 {insights['best_content_type']} content typically performs better")
        
        # Day recommendation
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        if days[day_of_week] != insights['best_day']:
            recommendations.append(f"📅 {insights['best_day']}s typically get more engagement")
        
        for rec in recommendations:
            st.info(rec)
    
    st.markdown("---")
    
    # Feature importance
    st.subheader("🔍 What Drives Engagement?")
    st.markdown("Based on AI analysis of your data, here are the key factors:")
    
    importance_df = model_package['feature_importance'].head(8).copy()
    importance_df['importance'] = importance_df['importance'] * 100
    
    # Rename features for better display
    feature_names = {
        'platform_encoded': 'Platform',
        'content_type_encoded': 'Content Type',
        'category_encoded': 'Category',
        'language_encoded': 'Language',
        'content_length': 'Content Length',
        'hour': 'Posting Time',
        'day_of_week': 'Day of Week',
        'month': 'Month',
        'follower_count': 'Follower Count',
        'age_encoded': 'Audience Age',
        'gender_encoded': 'Audience Gender',
        'sponsored_encoded': 'Sponsored'
    }
    importance_df['feature'] = importance_df['feature'].map(feature_names)
    
    fig = px.bar(
        importance_df,
        x='importance',
        y='feature',
        orientation='h',
        title='Feature Importance for Your Content',
        labels={'importance': 'Importance (%)', 'feature': 'Factor'},
        color='importance',
        color_continuous_scale='Blues'
    )
    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Comparison with historical data
    st.subheader("📊 How Does This Compare?")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Engagement by platform
        platform_data = training_data.groupby('platform')['engagement_rate'].mean().reset_index()
        fig = px.pie(
            platform_data,
            values='engagement_rate',
            names='platform',
            title='Average Engagement by Platform',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Engagement by hour
        hour_data = training_data.groupby('hour')['engagement_rate'].mean().reset_index()
        fig = px.line(
            hour_data,
            x='hour',
            y='engagement_rate',
            title='Engagement Rate by Hour of Day',
            markers=True,
            color_discrete_sequence=['#1f77b4']
        )
        fig.add_vline(x=hour, line_dash="dash", line_color="red", annotation_text="Your time")
        st.plotly_chart(fig, use_container_width=True)

else:
    # Welcome screen
    st.info("👈 Fill in your content details in the sidebar and click 'Get AI Predictions' to start!")
    
    if insights:
        st.subheader("💡 Quick Insights from Your Data")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Best Platform", insights['best_platform'])
            st.metric("Best Content Type", insights['best_content_type'])
        
        with col2:
            st.metric("Best Category", insights['best_category'])
            st.metric("Best Posting Hour", f"{insights['best_hour']}:00")
        
        with col3:
            st.metric("Best Day", insights['best_day'])
            st.metric("Avg Engagement Rate", f"{insights['avg_engagement']:.2f}%")
    
    # Sample visualization
    st.subheader("📈 Performance Trends")
    
    col1, col2 = st.columns(2)
    
    with col1:
        platform_perf = training_data.groupby('platform')['engagement_rate'].agg(['mean', 'count']).reset_index()
        fig = px.bar(
            platform_perf,
            x='platform',
            y='mean',
            title='Average Engagement by Platform',
            labels={'mean': 'Avg Engagement Rate (%)', 'platform': 'Platform'},
            color='mean',
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        content_perf = training_data.groupby('content_type')['engagement_rate'].mean().reset_index()
        fig = px.bar(
            content_perf,
            x='content_type',
            y='engagement_rate',
            title='Average Engagement by Content Type',
            labels={'engagement_rate': 'Avg Engagement Rate (%)', 'content_type': 'Content Type'},
            color='engagement_rate',
            color_continuous_scale='Plasma'
        )
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>🤖 Powered by AI & Machine Learning | 📊 Built with Streamlit</p>
        <p>💡 Tip: The more data you train with, the better the predictions!</p>
    </div>
    """, unsafe_allow_html=True)