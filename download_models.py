"""
Download models from Google Drive
Run this once before starting the app: python download_models.py
"""

import gdown
import os

print("📥 Downloading models from Google Drive...")
print("=" * 60)

# Your Google Drive file IDs
MODEL_FILE_ID = "1GV-_1h52wD_BG5P67dq4z-ZwaSRxoG_H"  # Replace with your model file ID
INSIGHTS_FILE_ID = "1QMWZy3_lmG5VpafRJRnOVsMK81q3z9di"  # Replace with your insights file ID

# Download URLs
model_url = f"https://drive.google.com/uc?id={MODEL_FILE_ID}"
insights_url = f"https://drive.google.com/uc?id={INSIGHTS_FILE_ID}"

# Download files
try:
    if not os.path.exists('social_media_model.joblib'):
        print("📦 Downloading social_media_model.joblib...")
        gdown.download(model_url, 'social_media_model.joblib', quiet=False)
        print("✅ Model downloaded successfully!")
    else:
        print("✅ Model already exists, skipping download")
    
    if not os.path.exists('insights.joblib'):
        print("📦 Downloading insights.joblib...")
        gdown.download(insights_url, 'insights.joblib', quiet=False)
        print("✅ Insights downloaded successfully!")
    else:
        print("✅ Insights already exist, skipping download")
    
    print("\n" + "=" * 60)
    print("🎉 All models downloaded successfully!")
    print("Run 'streamlit run app.py' to start the application")
    
except Exception as e:
    print(f"\n❌ Error downloading files: {e}")
    print("Please check your file IDs and internet connection")