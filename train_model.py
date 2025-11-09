"""
Social Media AI Model Training Script
This creates the "brain" that learns from your data
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

print("🚀 Starting AI Model Training...")
print("=" * 50)

# Step 1: Load your CSV data
print("📂 Loading your data...")
df = pd.read_csv('social_media_dataset.csv')
print(f"✅ Loaded {len(df)} rows of data")

# Step 2: Clean and prepare the data
print("\n🧹 Cleaning data...")

# Convert date to useful features
df['post_date'] = pd.to_datetime(df['post_date'])
df['hour'] = df['post_date'].dt.hour
df['day_of_week'] = df['post_date'].dt.dayofweek
df['month'] = df['post_date'].dt.month

# Create engagement rate (our main success metric)
df['engagement_rate'] = (df['likes'] + df['comments_countpython -m'] + df['shares']) / df['views'] * 100
df['engagement_rate'] = df['engagement_rate'].fillna(0)

# Classify content as high/medium/low performing
df['performance_class'] = pd.cut(df['engagement_rate'], 
                                  bins=3, 
                                  labels=['Low', 'Medium', 'High'])

print("✅ Data cleaned successfully")

# Step 3: Prepare features for AI
print("\n🔧 Preparing features for AI...")

# Encode categorical variables
le_platform = LabelEncoder()
le_content_type = LabelEncoder()
le_category = LabelEncoder()
le_language = LabelEncoder()
le_age = LabelEncoder()
le_gender = LabelEncoder()

df['platform_encoded'] = le_platform.fit_transform(df['platform'])
df['content_type_encoded'] = le_content_type.fit_transform(df['content_type'])
df['category_encoded'] = le_category.fit_transform(df['content_category'])
df['language_encoded'] = le_language.fit_transform(df['language'])
df['age_encoded'] = le_age.fit_transform(df['audience_age_distribution'])
df['gender_encoded'] = le_gender.fit_transform(df['audience_gender_distribution'])
df['sponsored_encoded'] = df['is_sponsored'].map({True: 1, False: 0})

# Select features for prediction
feature_columns = [
    'platform_encoded', 'content_type_encoded', 'category_encoded',
    'language_encoded', 'content_length', 'hour', 'day_of_week',
    'month', 'follower_count', 'age_encoded', 'gender_encoded',
    'sponsored_encoded'
]

X = df[feature_columns]
y_class = df['performance_class']
y_engagement = df['engagement_rate']

print("✅ Features prepared")

# Step 4: Train Classification Model (predicts High/Medium/Low)
print("\n🧠 Training Classification AI Model...")
X_train, X_test, y_train, y_test = train_test_split(X, y_class, test_size=0.2, random_state=42)

clf_model = RandomForestClassifier(n_estimators=100, random_state=42)
clf_model.fit(X_train, y_train)

accuracy = clf_model.score(X_test, y_test)
print(f"✅ Classification Model Accuracy: {accuracy*100:.2f}%")

# Step 5: Train Regression Model (predicts exact engagement rate)
print("\n🧠 Training Regression AI Model...")
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X, y_engagement, test_size=0.2, random_state=42)

reg_model = RandomForestRegressor(n_estimators=100, random_state=42)
reg_model.fit(X_train_reg, y_train_reg)

score = reg_model.score(X_test_reg, y_test_reg)
print(f"✅ Regression Model R² Score: {score:.2f}")

# Step 6: Calculate feature importance
print("\n📊 Calculating what matters most...")
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': clf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 5 Most Important Factors:")
for idx, row in feature_importance.head(5).iterrows():
    print(f"  • {row['feature']}: {row['importance']*100:.1f}%")

# Step 7: Save everything
print("\n💾 Saving AI models and encoders...")

# Create a package with everything we need
model_package = {
    'classifier': clf_model,
    'regressor': reg_model,
    'encoders': {
        'platform': le_platform,
        'content_type': le_content_type,
        'category': le_category,
        'language': le_language,
        'age': le_age,
        'gender': le_gender
    },
    'feature_columns': feature_columns,
    'feature_importance': feature_importance,
    'training_data': df[['platform', 'content_category', 'content_type', 
                         'language', 'hour', 'day_of_week', 
                         'engagement_rate', 'performance_class']].sample(min(100, len(df)))
}

with open('social_media_model.pkl', 'wb') as f:
    pickle.dump(model_package, f)

print("✅ Model saved as 'social_media_model.pkl'")

# Step 8: Generate insights
print("\n📈 Generating Insights...")

# Best performing combinations
best_platform = df.groupby('platform')['engagement_rate'].mean().idxmax()
best_content_type = df.groupby('content_type')['engagement_rate'].mean().idxmax()
best_category = df.groupby('content_category')['engagement_rate'].mean().idxmax()
best_hour = df.groupby('hour')['engagement_rate'].mean().idxmax()
best_day = df.groupby('day_of_week')['engagement_rate'].mean().idxmax()

days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

insights = {
    'best_platform': best_platform,
    'best_content_type': best_content_type,
    'best_category': best_category,
    'best_hour': int(best_hour),
    'best_day': days[int(best_day)],
    'avg_engagement': df['engagement_rate'].mean()
}

with open('insights.pkl', 'wb') as f:
    pickle.dump(insights, f)

print("\n🎉 SUCCESS! Your AI is ready!")
print("=" * 50)
print("\n📊 Key Insights from Training:")
print(f"  • Best Platform: {insights['best_platform']}")
print(f"  • Best Content Type: {insights['best_content_type']}")
print(f"  • Best Category: {insights['best_category']}")
print(f"  • Best Time to Post: {insights['best_hour']}:00")
print(f"  • Best Day: {insights['best_day']}")
print(f"  • Average Engagement Rate: {insights['avg_engagement']:.2f}%")

print("\n✨ Next step: Run 'streamlit run app.py' to see your dashboard!")