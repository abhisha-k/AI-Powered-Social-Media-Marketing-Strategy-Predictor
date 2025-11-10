"""
Social Media AI Strategy Dashboard - Modern Edition
Beautiful, interactive web app with stunning visuals
"""

import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np
import os
import gdown

# Model file IDs from Google Drive
MODEL_FILE_ID = "1GV-_1h52wD_BG5P67dq4z-ZwaSRxoG_H"
INSIGHTS_FILE_ID = "1QMWZy3_lmG5VpafRJRnOVsMK81q3z9di"

# Function to download models if not present
def ensure_models_downloaded():
    """Download models from Google Drive if not present"""
    model_exists = os.path.exists('social_media_model.joblib')
    insights_exist = os.path.exists('insights.joblib')
    
    if not model_exists or not insights_exist:
        with st.spinner('📥 Downloading AI models from Google Drive... (first time only)'):
            try:
                if not model_exists:
                    model_url = f"https://drive.google.com/uc?id={MODEL_FILE_ID}"
                    gdown.download(model_url, 'social_media_model.joblib', quiet=True)
                
                if not insights_exist:
                    insights_url = f"https://drive.google.com/uc?id={INSIGHTS_FILE_ID}"
                    gdown.download(insights_url, 'insights.joblib', quiet=True)
                
                st.success('✅ Models downloaded successfully!')
            except Exception as e:
                st.error(f"❌ Error downloading models: {e}")
                st.error("Please run: python download_models.py")
                st.stop()

# Download models if needed
ensure_models_downloaded()

# Page configuration
st.set_page_config(
    page_title="AI Social Media Strategy",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern CSS with gradients, animations, and glassmorphism
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 0;
    }
    
    .block-container {
        padding: 3rem 2rem;
        max-width: 1400px;
    }
    
    /* Glassmorphism cards */
    .stMetric {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 25px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .stMetric:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 48px 0 rgba(31, 38, 135, 0.5);
    }
    
    .stMetric label {
        color: rgba(255, 255, 255, 0.9) !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stMetric [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-weight: 800 !important;
        font-size: 2.5rem !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .stMetric [data-testid="stMetricDelta"] {
        color: rgba(255, 255, 255, 0.8) !important;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3c72 0%, #2a5298 100%);
        border-right: 2px solid rgba(255, 255, 255, 0.1);
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stNumberInput label,
    [data-testid="stSidebar"] .stSlider label {
        color: rgba(255, 255, 255, 0.95) !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
    }
    
    /* Buttons */
    .stButton button {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        font-weight: 700;
        border: none;
        border-radius: 50px;
        padding: 15px 40px;
        font-size: 1.1rem;
        letter-spacing: 1px;
        text-transform: uppercase;
        box-shadow: 0 10px 30px rgba(245, 87, 108, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 40px rgba(245, 87, 108, 0.6);
    }
    
    /* Title styling */
    h1 {
        color: white !important;
        font-weight: 900 !important;
        font-size: 3.5rem !important;
        text-align: center;
        text-shadow: 3px 3px 6px rgba(0,0,0,0.3);
        margin-bottom: 0.5rem !important;
        background: linear-gradient(135deg, #ffffff 0%, #e0e7ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    h2, h3 {
        color: white !important;
        font-weight: 700 !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    /* Info boxes */
    .stAlert {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 15px;
        border-left: 5px solid #4ade80;
        color: white;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.2);
    }
    
    /* Chart containers */
    .js-plotly-plot {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 15px;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Subheader glow effect */
    .subtitle {
        text-align: center;
        color: rgba(255, 255, 255, 0.9);
        font-size: 1.3rem;
        font-weight: 400;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    /* Success message */
    .success-banner {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 20px;
        text-align: center;
        margin: 20px 0;
        border: 2px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 10px 40px rgba(0,0,0,0.3);
    }
    
    .success-banner h2 {
        font-size: 2.5rem;
        margin: 0;
        color: white;
    }
    
    /* Divider */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
        margin: 2rem 0;
    }
    
    /* Recommendation cards */
    [data-testid="stMarkdownContainer"] p {
        color: rgba(255, 255, 255, 0.95) !important;
        line-height: 1.6;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: rgba(255, 255, 255, 0.7);
        padding: 40px 20px;
        margin-top: 3rem;
        background: rgba(0, 0, 0, 0.2);
        border-radius: 20px;
        backdrop-filter: blur(10px);
    }
    
    /* Input fields */
    .stSelectbox > div > div,
    .stNumberInput > div > div,
    .stSlider > div > div {
        background: rgba(255, 255, 255, 0.1) !important;
        border-radius: 10px !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Load the trained model
@st.cache_resource
def load_model():
    try:
        return joblib.load('social_media_model.joblib') 
    except FileNotFoundError:
        st.error("❌ Model not found! Please run 'python train_model.py' first.")
        st.stop()

@st.cache_resource
def load_insights():
    try:
        return joblib.load('insights.joblib') 
    except FileNotFoundError:
        return None

# Load model and data
model_package = load_model()
insights = load_insights()

classifier = model_package['classifier']
regressor = model_package['regressor']
encoders = model_package['encoders']
feature_cols = model_package['feature_columns']
sample_stats = model_package.get('sample_stats', {})

# Animated header
st.markdown("""
    <div style='text-align: center; margin-bottom: 1rem;'>
        <h1>🚀 AI Social Media Strategy</h1>
        <p class='subtitle'>✨ Transform Your Content Into Viral Success ✨</p>
    </div>
    """, unsafe_allow_html=True)

# Sidebar for inputs
st.sidebar.markdown("""
    <div style='text-align: center; padding: 20px 0;'>
        <h2>📊 Content Details</h2>
        <p style='font-size: 0.9rem; opacity: 0.8;'>Configure your post settings</p>
    </div>
    """, unsafe_allow_html=True)

# User inputs with emojis
platform = st.sidebar.selectbox(
    "🌐 Platform",
    options=list(encoders['platform'].classes_)
)

content_type = st.sidebar.selectbox(
    "🎬 Content Type",
    options=list(encoders['content_type'].classes_)
)

category = st.sidebar.selectbox(
    "📁 Category",
    options=list(encoders['category'].classes_)
)

language = st.sidebar.selectbox(
    "🌍 Language",
    options=list(encoders['language'].classes_)
)

content_length = st.sidebar.slider(
    "📏 Content Length",
    min_value=10,
    max_value=600,
    value=150
)

follower_count = st.sidebar.number_input(
    "👥 Follower Count",
    min_value=0,
    value=50000,
    step=1000
)

age_group = st.sidebar.selectbox(
    "🎯 Target Age",
    options=list(encoders['age'].classes_)
)

gender = st.sidebar.selectbox(
    "👤 Target Gender",
    options=list(encoders['gender'].classes_)
)

is_sponsored = st.sidebar.checkbox("💰 Sponsored Content")

st.sidebar.markdown("<br>", unsafe_allow_html=True)
st.sidebar.markdown("""
    <div style='text-align: center; padding: 10px 0;'>
        <h3>⏰ Schedule</h3>
    </div>
    """, unsafe_allow_html=True)

posting_date = st.sidebar.date_input("📅 Date", datetime.now())

# Create hour options with labels
hour_options = {
    f"{h:02d}:00 - {['Midnight','Early Morning','Early Morning','Early Morning','Early Morning','Early Morning','Morning','Morning','Morning','Morning','Morning','Noon','Afternoon','Afternoon','Afternoon','Afternoon','Afternoon','Evening','Evening','Evening','Evening','Night','Night','Night'][h]}" : h 
    for h in range(24)
}

selected_hour_label = st.sidebar.selectbox(
    "🕐 Posting Hour",
    options=list(hour_options.keys()),
    index=datetime.now().hour
)

hour = hour_options[selected_hour_label]
day_of_week = posting_date.weekday()
month = posting_date.month

st.sidebar.markdown("<br>", unsafe_allow_html=True)
predict_button = st.sidebar.button("🎯 PREDICT SUCCESS", use_container_width=True)

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
    potential_views = follower_count * 0.15
    potential_likes = potential_views * (engagement_pred / 100) * 0.6
    potential_comments = potential_views * (engagement_pred / 100) * 0.2
    potential_shares = potential_views * (engagement_pred / 100) * 0.2
    
    # Success banner
    st.markdown("""
        <div class='success-banner'>
            <h2>✨ AI Analysis Complete! ✨</h2>
            <p style='font-size: 1.1rem; margin: 10px 0 0 0;'>Your content strategy is ready</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Main metrics with icons
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "🎯 Performance",
            performance_pred,
            delta="AI Predicted"
        )
    
    with col2:
        delta_val = f"{engagement_pred - insights['avg_engagement']:.1f}%" if insights else None
        st.metric(
            "💫 Engagement",
            f"{engagement_pred:.1f}%",
            delta=delta_val
        )
    
    with col3:
        st.metric(
            "👁️ Expected Views",
            f"{int(potential_views):,}",
            delta="Reach"
        )
    
    with col4:
        total_interactions = int(potential_likes + potential_comments + potential_shares)
        st.metric(
            "❤️ Interactions",
            f"{total_interactions:,}",
            delta="Total"
        )
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Detailed breakdown
    col_left, col_right = st.columns([1.2, 1])
    
    with col_left:
        st.markdown("### 📊 Engagement Breakdown")
        
        breakdown_df = pd.DataFrame({
            'Metric': ['👁️ Views', '❤️ Likes', '💬 Comments', '🔄 Shares'],
            'Count': [
                int(potential_views),
                int(potential_likes),
                int(potential_comments),
                int(potential_shares)
            ]
        })
        
        fig = go.Figure(data=[
            go.Bar(
                x=breakdown_df['Metric'],
                y=breakdown_df['Count'],
                text=breakdown_df['Count'].apply(lambda x: f'{x:,}'),
                textposition='outside',
                marker=dict(
                    color=breakdown_df['Count'],
                    colorscale='Sunset',
                    line=dict(color='rgba(255,255,255,0.3)', width=2)
                ),
                hovertemplate='<b>%{x}</b><br>Count: %{y:,}<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title=dict(text='Expected Performance', font=dict(size=20, color='white')),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
            height=400,
            margin=dict(t=50, b=50, l=50, r=50)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col_right:
        st.markdown("### 💡 Smart Recommendations")
        
        recommendations = []
        
        if engagement_pred < insights['avg_engagement']:
            recommendations.append(("⚠️", f"Engagement below average by {insights['avg_engagement'] - engagement_pred:.1f}%"))
        else:
            recommendations.append(("✅", "Excellent! Above average performance predicted"))
        
        if hour != insights['best_hour']:
            recommendations.append(("⏰", f"Best time: {insights['best_hour']}:00"))
        else:
            recommendations.append(("✅", "Perfect timing chosen!"))
        
        if platform != insights['best_platform']:
            recommendations.append(("📱", f"Try {insights['best_platform']} for better reach"))
        
        if content_type != insights['best_content_type']:
            recommendations.append(("🎬", f"{insights['best_content_type']} performs best"))
        
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        if days[day_of_week] != insights['best_day']:
            recommendations.append(("📅", f"{insights['best_day']} is optimal"))
        
        for emoji, rec in recommendations:
            st.markdown(f"""
                <div style='background: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px; 
                margin: 10px 0; border-left: 4px solid #4ade80; backdrop-filter: blur(10px);'>
                    <p style='margin: 0; font-size: 1rem;'><strong>{emoji}</strong> {rec}</p>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Feature importance
    st.markdown("### 🔍 Engagement Drivers")
    
    importance_df = model_package['feature_importance'].head(8).copy()
    importance_df['importance'] = importance_df['importance'] * 100
    
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
    
    fig = go.Figure(data=[
        go.Bar(
            y=importance_df['feature'],
            x=importance_df['importance'],
            orientation='h',
            marker=dict(
                color=importance_df['importance'],
                colorscale='Plasma',
                line=dict(color='rgba(255,255,255,0.3)', width=1)
            ),
            text=importance_df['importance'].apply(lambda x: f'{x:.1f}%'),
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Importance: %{x:.1f}%<extra></extra>'
        )
    ])
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', title='Importance (%)'),
        yaxis=dict(showgrid=False, title=''),
        height=400,
        margin=dict(t=20, b=50, l=150, r=50)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Comparison charts
    st.markdown("### 📈 Historical Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if sample_stats and 'avg_engagement_by_platform' in sample_stats:
            platform_data = pd.DataFrame(
                list(sample_stats['avg_engagement_by_platform'].items()),
                columns=['platform', 'rate']
            )
            
            fig = go.Figure(data=[
                go.Pie(
                    labels=platform_data['platform'],
                    values=platform_data['rate'],
                    hole=0.4,
                    marker=dict(colors=px.colors.qualitative.Set3, 
                               line=dict(color='white', width=2)),
                    textinfo='label+percent',
                    textfont=dict(size=14, color='white')
                )
            ])
            
            fig.update_layout(
                title=dict(text='Engagement by Platform', font=dict(size=18, color='white')),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                height=350,
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        if sample_stats and 'avg_engagement_by_hour' in sample_stats:
            hour_data = pd.DataFrame(
                list(sample_stats['avg_engagement_by_hour'].items()),
                columns=['hour', 'rate']
            )
            hour_data['hour'] = hour_data['hour'].astype(int)
            hour_data = hour_data.sort_values('hour')
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=hour_data['hour'],
                y=hour_data['rate'],
                mode='lines+markers',
                line=dict(color='#f093fb', width=3),
                marker=dict(size=10, color='#f5576c', 
                           line=dict(color='white', width=2)),
                fill='tozeroy',
                fillcolor='rgba(240, 147, 251, 0.2)',
                hovertemplate='<b>Hour %{x}</b><br>Engagement: %{y:.2f}%<extra></extra>'
            ))
            
            fig.add_vline(x=hour, line_dash="dash", line_color="yellow", 
                         line_width=2, annotation_text="Your Time",
                         annotation_position="top")
            
            fig.update_layout(
                title=dict(text='Engagement by Hour', font=dict(size=18, color='white')),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', 
                          title='Hour of Day'),
                yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', 
                          title='Engagement Rate (%)'),
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)

else:
    # Welcome screen
    st.markdown("""
        <div style='background: rgba(255,255,255,0.1); padding: 40px; border-radius: 20px; 
        text-align: center; backdrop-filter: blur(10px); border: 2px solid rgba(255,255,255,0.2);
        margin: 40px 0;'>
            <h2 style='font-size: 2rem; margin-bottom: 1rem;'>👋 Welcome to Your AI Strategy Center</h2>
            <p style='font-size: 1.2rem; opacity: 0.9;'>
                Configure your content details in the sidebar and unlock powerful AI predictions
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    if insights:
        st.markdown("### 💎 Top Performance Insights")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("🏆 Best Platform", insights['best_platform'])
            st.metric("🎬 Best Content", insights['best_content_type'])
        
        with col2:
            st.metric("📁 Top Category", insights['best_category'])
            st.metric("⏰ Optimal Hour", f"{insights['best_hour']}:00")
        
        with col3:
            st.metric("📅 Best Day", insights['best_day'])
            st.metric("📊 Avg Engagement", f"{insights['avg_engagement']:.1f}%")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Sample visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        if sample_stats and 'avg_engagement_by_platform' in sample_stats:
            platform_data = pd.DataFrame(
                list(sample_stats['avg_engagement_by_platform'].items()),
                columns=['platform', 'rate']
            )
            
            fig = go.Figure(data=[
                go.Bar(
                    x=platform_data['platform'],
                    y=platform_data['rate'],
                    marker=dict(
                        color=platform_data['rate'],
                        colorscale='Viridis',
                        line=dict(color='rgba(255,255,255,0.3)', width=2)
                    ),
                    text=platform_data['rate'].apply(lambda x: f'{x:.1f}%'),
                    textposition='outside',
                    hovertemplate='<b>%{x}</b><br>Engagement: %{y:.2f}%<extra></extra>'
                )
            ])
            
            fig.update_layout(
                title=dict(text='Platform Performance', font=dict(size=18, color='white')),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                xaxis=dict(showgrid=False, title='Platform'),
                yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', 
                          title='Engagement Rate (%)'),
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        if insights:
            st.markdown("""
                <div style='background: rgba(255,255,255,0.1); padding: 30px; border-radius: 15px; 
                backdrop-filter: blur(10px); height: 350px; display: flex; flex-direction: column; 
                justify-content: center;'>
                    <h3 style='text-align: center; margin-bottom: 20px;'>🎯 Quick Stats</h3>
                    <div style='text-align: center;'>
                        <p style='font-size: 3rem; margin: 10px 0; font-weight: 800;'>
                            {:.1f}%
                        </p>
                        <p style='font-size: 1.2rem; opacity: 0.8;'>Average Engagement Rate</p>
                        <p style='font-size: 1.5rem; margin-top: 20px; font-weight: 700;'>
                            {}
                        </p>
                        <p style='font-size: 1rem; opacity: 0.8;'>Top Performing Content</p>
                    </div>
                </div>
                """.format(insights['avg_engagement'], insights['best_content_type']), 
                unsafe_allow_html=True)

# Footer
st.markdown("""
    <div class='footer'>
        <h3 style='margin-bottom: 15px;'>🤖 Powered by Advanced AI & Machine Learning</h3>
        <p style='font-size: 1.1rem; margin: 10px 0;'>
            Built with ❤️ using Streamlit • Real-time Predictions • Data-Driven Insights
        </p>
        <p style='font-size: 0.9rem; opacity: 0.7; margin-top: 15px;'>
            💡 Pro Tip: More training data = More accurate predictions
        </p>
    </div>

    """, unsafe_allow_html=True)
