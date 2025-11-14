import os
import sqlite3
from flask import Flask, render_template, request, redirect, url_for, session, jsonify, flash
import google.generativeai as genai
from datetime import datetime, timedelta
import cv2
import numpy as np
import base64
import json
import re
import pandas as pd
import glob
import joblib
from notifications import get_unread_notifications_count, get_user_notifications

app = Flask(__name__)
app.secret_key = 'smart_wardrobe_secret_key_2024'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['DATA_FOLDER'] = 'static/data'
# Add this after creating your Flask app
app.jinja_env.globals.update(
    get_unread_notifications_count=get_unread_notifications_count,
    get_user_notifications=get_user_notifications
)
# Configure Gemini AI
genai.configure(api_key='AIzaSyD7JLMPRNpM3ygSSWwkckNM3QPX4WLQMFo')
gemini_model = genai.GenerativeModel('gemini-2.5-flash')

# Global variables for camera
camera = None

# Load ML models
try:
    kmeans = joblib.load("ml_models/wash_cluster_model.pkl")
    cluster_encoders = joblib.load("ml_models/cluster_encoders.pkl")
    print("✅ ML models loaded successfully")
    print(f"🔧 Model categories: {cluster_encoders}")
    wash_model = joblib.load("ml_models/wash_prediction_model.pkl")
    wash_scaler = joblib.load("ml_models/wash_scaler.pkl")
    wash_encoders = joblib.load("ml_models/wash_encoders.pkl")
    print("✅ Wash prediction models loaded successfully")
except Exception as e:
    print(f"❌ Error loading ML models: {e}")
    kmeans = None
    cluster_encoders = None
    print(f"❌ Error loading wash prediction models: {e}")
    wash_model = None
    wash_scaler = None
    wash_encoders = None

# Database initialization
def init_db():
    conn = sqlite3.connect('wardrobe.db')
    c = conn.cursor()
    
    # Users table
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE NOT NULL,
                  email TEXT UNIQUE NOT NULL,
                  password TEXT NOT NULL,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    
    # Clothes table
    c.execute('''CREATE TABLE IF NOT EXISTS clothes
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER,
                  cloth_name TEXT,
                  category TEXT,
                  fabric_type TEXT,
                  color TEXT,
                  weather_conditions TEXT,
                  care_instructions TEXT,
                  image_path TEXT,
                  care_label_image_path TEXT,
                  last_worn DATE,
                  wear_count INTEGER DEFAULT 0,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY (user_id) REFERENCES users (id))''')
    
    # Outfit recommendations table
    c.execute('''CREATE TABLE IF NOT EXISTS outfit_recommendations
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER,
                  recommended_items TEXT,
                  weather_conditions TEXT,
                  recommendation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY (user_id) REFERENCES users (id))''')
    
    # Laundry tracker table
    c.execute('''CREATE TABLE IF NOT EXISTS laundry_tracker
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER,
                  cloth_id INTEGER,
                  last_washed DATE,
                  next_wash_date DATE,
                  wash_priority TEXT,
                  FOREIGN KEY (user_id) REFERENCES users (id),
                  FOREIGN KEY (cloth_id) REFERENCES clothes (id))''')
    
    # Chat history table
    c.execute('''CREATE TABLE IF NOT EXISTS chat_history
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER,
                  message TEXT,
                  response TEXT,
                  timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY (user_id) REFERENCES users (id))''')
    # Notifications table
    c.execute('''CREATE TABLE IF NOT EXISTS notifications
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER,
                  cloth_id INTEGER,
                  message TEXT,
                  type TEXT,
                  is_read BOOLEAN DEFAULT 0,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY (user_id) REFERENCES users (id),
                  FOREIGN KEY (cloth_id) REFERENCES clothes (id))''')
    
    conn.commit()
    conn.close()

init_db()

def get_db_connection():
    conn = sqlite3.connect('wardrobe.db')
    conn.row_factory = sqlite3.Row
    return conn

def load_clothes_dataset():
    """Load clothes dataset from CSV file"""
    try:
        df = pd.read_csv('clothes_dataset.csv')
        print(f"✅ Loaded dataset with {len(df)} items")
        print(f"📊 Dataset columns: {df.columns.tolist()}")
        print(f"🎯 Sample cloth_ids: {df['cloth_id'].head().tolist()}")
        return df
    except FileNotFoundError:
        print("❌ Clothes dataset CSV file not found")
        return pd.DataFrame()
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return pd.DataFrame()

def get_clothes_from_data_folder():
    """Get all clothing images from data folder"""
    data_folder = app.config['DATA_FOLDER']
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp']
    clothes = []
    
    for extension in image_extensions:
        for filepath in glob.glob(os.path.join(data_folder, extension)):
            filename = os.path.basename(filepath)
            # Extract cloth_id from filename (remove extension)
            cloth_id = os.path.splitext(filename)[0]
            clothes.append({
                'id': cloth_id,
                'image_path': f'data/{filename}',
                'filename': filename
            })
    
    print(f"✅ Found {len(clothes)} images in data folder")
    print(f"📸 Available cloth_ids: {[cloth['id'] for cloth in clothes]}")
    return clothes
def predict_wash_time(fabric_type, usage_count, days_since_last_wash, temperature_avg, humidity_avg, care_instructions, odor_detected=False, stain_detected=False):
    """Predict when a cloth needs washing using ML model with Gemini AI assistance"""
    if wash_model is None:
        print("❌ Wash prediction model not available")
        return 7  # Default fallback
    
    try:
        # Use Gemini AI to extract parameters from care instructions and detect odor/stains
        prompt = f"""
        Analyze this clothing information and determine:
        - Fabric Type: {fabric_type}
        - Usage Count: {usage_count}
        - Days Since Last Wash: {days_since_last_wash}
        - Temperature When Last Worn: {temperature_avg}°C
        - Humidity When Last Worn: {humidity_avg}%
        - Care Instructions: {care_instructions}
        
        Based on the care instructions and fabric type, determine:
        1. Standard care label category: "Cold Wash", "Gentle Wash", "Hand Wash", or "Normal Wash"
        2. Likelihood of odor based on usage and days since wash (return "Yes" or "No")
        3. Likelihood of stains based on fabric type and usage (return "Yes" or "No")
        
        Return ONLY a JSON with these exact keys:
        - care_label: "Cold Wash" or "Gentle Wash" or "Hand Wash" or "Normal Wash"
        - odor_detected: "Yes" or "No"
        - stain_detected: "Yes" or "No"
        """
        
        response = gemini_model.generate_content(prompt)
        response_text = response.text.strip()
        
        print(f"🤖 Wash analysis response: {response_text}")
        
        # Extract JSON from response
        try:
            wash_params = json.loads(response_text)
        except:
            # Default values if parsing fails
            wash_params = {
                'care_label': 'Normal Wash',
                'odor_detected': 'Yes' if days_since_last_wash > 7 else 'No',
                'stain_detected': 'Yes' if usage_count > 10 else 'No'
            }
        
        # Prepare input for ML model
        data = pd.DataFrame([{
            'fabric_type': fabric_type,
            'usage_count': usage_count,
            'days_since_last_wash': days_since_last_wash,
            'temperature_avg': temperature_avg,
            'humidity_avg': humidity_avg,
            'care_label': wash_params.get('care_label', 'Normal Wash'),
            'odor_detected': wash_params.get('odor_detected', 'No'),
            'stain_detected': wash_params.get('stain_detected', 'No')
        }])
        
        print(f"🧺 ML Wash Prediction Input: {data.iloc[0].to_dict()}")
        
        # Create a copy for encoding
        data_encoded = data.copy()
        
        # Handle categorical encoding with unseen labels
        for col, le in wash_encoders.items():
            if col in data_encoded.columns:
                available_categories = list(le.classes_)
                input_value = data_encoded[col].iloc[0]
                
                if input_value not in available_categories:
                    print(f"⚠️ Unseen label '{input_value}' in '{col}', using first available: '{available_categories[0]}'")
                    data_encoded[col] = available_categories[0]
                else:
                    data_encoded[col] = le.transform(data_encoded[col])
        
        print(f"🔢 Encoded data: {data_encoded.iloc[0].to_dict()}")
        
        # Scale the data
        data_scaled = wash_scaler.transform(data_encoded)
        print(f"📊 Scaled data shape: {data_scaled.shape}")
        
        # Predict wash days
        pred_days = wash_model.predict(data_scaled)[0]
        print(f"✅ Predicted wash due in: {pred_days:.0f} days")
        
        return max(1, int(pred_days))  # Ensure at least 1 day
        
    except Exception as e:
        print(f"❌ Error in wash prediction: {e}")
        # Fallback calculation based on simple rules
        base_days = 7
        fabric_lower = fabric_type.lower()
        if 'wool' in fabric_lower or 'silk' in fabric_lower:
            base_days = 14
        elif 'denim' in fabric_lower:
            base_days = 10
        elif 'cotton' in fabric_lower:
            base_days = 7
        
        # Adjust based on usage
        if usage_count > 10:
            base_days = max(3, base_days - 2)
        
        return base_days
def get_laundry_recommendations_with_predictions(user_id):
    """Get laundry recommendations with correct wash predictions"""
    conn = get_db_connection()
    
    clothes = conn.execute('''
        SELECT c.*, 
               CASE 
                   WHEN c.last_worn IS NULL THEN 999
                   ELSE julianday('now') - julianday(c.last_worn) 
               END as days_since_worn
        FROM clothes c
        WHERE c.user_id = ?
    ''', (user_id,)).fetchall()
    
    recommendations = []
    
    for cloth in clothes:
        days_since_worn = cloth['days_since_worn'] if cloth['days_since_worn'] != 999 else 0
        wear_count = cloth['wear_count'] or 0
        
        # Default temperature and humidity
        temperature = 25
        humidity = 60
        
        # Predict wash time using AI
        wash_due_days = predict_wash_time_ai(
            fabric_type=cloth['fabric_type'],
            usage_count=wear_count,
            days_since_last_wash=days_since_worn,
            temperature_avg=temperature,
            humidity_avg=humidity,
            care_instructions=cloth['care_instructions']
        )
        
        # CORRECTED PRIORITY LOGIC:
        if days_since_worn == 0 or cloth['last_worn'] is None:
            # Never worn or worn today - No need to wash
            priority = "Fresh"
            priority_reason = "Worn today or never worn"
        elif days_since_worn >= wash_due_days:
            # Exceeded recommended wash cycle - High priority
            priority = "High"
            priority_reason = f"Ready for washing ({days_since_worn} days worn)"
        elif days_since_worn >= (wash_due_days - 1):
            # Will need washing soon - Medium priority
            priority = "Medium"
            priority_reason = f"Wash in {wash_due_days - days_since_worn} day(s)"
        else:
            # Within wash cycle - Low priority
            priority = "Low"
            priority_reason = f"Fresh for {wash_due_days - days_since_worn} more day(s)"
        
        recommendations.append({
            'id': cloth['id'],
            'cloth_name': cloth['cloth_name'],
            'days_since_worn': int(days_since_worn) if days_since_worn != 999 else 'Never',
            'wear_count': wear_count,
            'wash_priority': priority,
            'priority_reason': priority_reason,
            'fabric_type': cloth['fabric_type'],
            'predicted_wash_days': wash_due_days,
            'care_instructions': cloth['care_instructions'],
            'last_worn': cloth['last_worn']
        })
    
    conn.close()
    
    # Filter out "Fresh" items from laundry alerts (they don't need washing)
    laundry_alerts = [item for item in recommendations if item['wash_priority'] in ['High', 'Medium', 'Low']]
    
    # Sort by priority (High first, then Medium, then Low)
    priority_order = {'High': 0, 'Medium': 1, 'Low': 2}
    laundry_alerts.sort(key=lambda x: priority_order[x['wash_priority']])
    
    return laundry_alerts
def get_current_weather():
    """Get current weather for Bangalore using Gemini AI"""
    try:
        prompt = """
        What is the current weather in Bangalore, India? 
        Return ONLY a JSON with these exact keys:
        - temperature: current temperature in Celsius
        - condition: "Sunny", "Cloudy", "Rainy", "Partly Cloudy", "Hot", "Cool", "Cold"
        - humidity: humidity percentage
        - recommendation: "Hot", "Warm", "Cool", "Cold" based on temperature
        """
        
        response = gemini_model.generate_content(prompt)
        response_text = response.text.strip()
        
        print(f"🌤️ Weather API Response: {response_text}")
        
        # Extract JSON from response
        try:
            # Clean the response text
            cleaned_text = re.sub(r'```json|```', '', response_text).strip()
            weather_data = json.loads(cleaned_text)
            
            # Ensure we have all required fields
            default_weather = {
                'temperature': 28,
                'condition': 'Partly Cloudy', 
                'humidity': 65,
                'recommendation': 'Warm'
            }
            
            # Merge with defaults for missing fields
            for key, value in default_weather.items():
                if key not in weather_data:
                    weather_data[key] = value
            
            print(f"✅ Weather for Bangalore: {weather_data}")
            return weather_data
            
        except Exception as e:
            print(f"❌ Error parsing weather data: {e}")
            # Return default Bangalore weather
            return {
                'temperature': 28,
                'condition': 'Partly Cloudy',
                'humidity': 65,
                'recommendation': 'Warm'
            }
            
    except Exception as e:
        print(f"❌ Error getting weather: {e}")
        return {
            'temperature': 28,
            'condition': 'Partly Cloudy',
            'humidity': 65,
            'recommendation': 'Warm'
        }

def get_weather_recommendations(user_id):
    """Get clothing recommendations based on current weather"""
    conn = get_db_connection()
    
    # Get user's clothes
    user_clothes = conn.execute('SELECT * FROM clothes WHERE user_id = ?', (user_id,)).fetchall()
    conn.close()
    
    if not user_clothes:
        return []
    
    # Get current weather
    weather = get_current_weather()
    
    # Prepare clothes data for AI
    clothes_data = []
    for cloth in user_clothes:
        clothes_data.append({
            'id': cloth['id'],
            'name': cloth['cloth_name'],
            'category': cloth['category'],
            'fabric': cloth['fabric_type'],
            'color': cloth['color'],
            'weather_suitable': cloth['weather_conditions']
        })
    
    try:
        prompt = f"""
        Current Weather in Bangalore:
        - Temperature: {weather['temperature']}°C
        - Condition: {weather['condition']}
        - Humidity: {weather['humidity']}%
        - Recommendation: {weather['recommendation']}
        
        User's Available Clothes:
        {json.dumps(clothes_data, indent=2)}
        
        Recommend exactly 3-5 clothing items (by ID) that are most suitable for today's weather.
        Consider:
        1. Weather suitability (hot, warm, cool, cold)
        2. Fabric type appropriateness for current conditions
        3. Practicality for the weather condition
        
        Return ONLY a JSON array of recommended cloth IDs: [1, 2, 3]
        """
        
        response = gemini_model.generate_content(prompt)
        response_text = response.text.strip()
        
        print(f"🤖 Weather Recommendations Response: {response_text}")
        
        # Extract JSON array from response
        try:
            cleaned_text = re.sub(r'```json|```', '', response_text).strip()
            recommended_ids = json.loads(cleaned_text)
            
            if isinstance(recommended_ids, list):
                # Get the actual cloth objects
                recommendations = []
                for cloth_id in recommended_ids:
                    cloth = next((c for c in user_clothes if c['id'] == cloth_id), None)
                    if cloth:
                        recommendations.append({
                            'id': cloth['id'],
                            'name': cloth['cloth_name'],
                            'category': cloth['category'],
                            'fabric': cloth['fabric_type'],
                            'color': cloth['color'],
                            'image': cloth['image_path'],
                            'weather': cloth['weather_conditions']
                        })
                
                print(f"✅ Weather-based recommendations: {[r['id'] for r in recommendations]}")
                return recommendations
                
        except Exception as e:
            print(f"❌ Error parsing recommendations: {e}")
    
    except Exception as e:
        print(f"❌ Error getting weather recommendations: {e}")
    
    # Fallback: return first 3 clothes
    return [
        {
            'id': cloth['id'],
            'name': cloth['cloth_name'],
            'category': cloth['category'],
            'fabric': cloth['fabric_type'],
            'color': cloth['color'],
            'image': cloth['image_path'],
            'weather': cloth['weather_conditions']
        }
        for cloth in user_clothes[:3]
    ]
def predict_wash_time_ai(fabric_type, usage_count, days_since_last_wash, temperature_avg, humidity_avg, care_instructions):
    """Predict wash time using AI with more realistic cycles"""
    try:
        prompt = f"""
        As a laundry expert, predict how many days until this clothing item needs washing.
        
        ITEM DETAILS:
        - Fabric: {fabric_type}
        - Times Worn: {usage_count}
        - Days Since Last Wash: {days_since_last_wash}
        - Care: {care_instructions}
        
        Return ONLY the number of days (3-14) until washing is needed.
        Consider fabric type, usage, and care instructions.
        """
        
        response = gemini_model.generate_content(prompt)
        response_text = response.text.strip()
        
        print(f"🤖 Wash Prediction: {response_text}")
        
        # Extract number from response
        try:
            # Look for numbers in the response
            import re
            numbers = re.findall(r'\d+', response_text)
            if numbers:
                wash_days = int(numbers[0])
                # Ensure reasonable range (3-14 days)
                wash_days = max(3, min(14, wash_days))
            else:
                # Default based on fabric type
                fabric_lower = fabric_type.lower()
                if 'wool' in fabric_lower or 'silk' in fabric_lower:
                    wash_days = 10
                elif 'denim' in fabric_lower:
                    wash_days = 8
                elif 'cotton' in fabric_lower:
                    wash_days = 5
                elif 'linen' in fabric_lower:
                    wash_days = 6
                else:
                    wash_days = 7
                    
        except:
            # Fallback defaults
            wash_days = 7
        
        print(f"✅ Final wash prediction: {wash_days} days")
        return wash_days
        
    except Exception as e:
        print(f"❌ Error in wash prediction: {e}")
        return 7  # Default fallback

def calculate_fallback_wash_time(fabric_type, usage_count, days_since_last_wash):
    """Fallback calculation when AI fails"""
    # Base days based on fabric type
    fabric_lower = fabric_type.lower()
    
    if 'wool' in fabric_lower or 'silk' in fabric_lower:
        base_days = 14  # Delicate fabrics can go longer
    elif 'denim' in fabric_lower:
        base_days = 10  # Denim doesn't need frequent washing
    elif 'linen' in fabric_lower:
        base_days = 8
    elif 'cotton' in fabric_lower:
        base_days = 7
    elif 'polyester' in fabric_lower or 'nylon' in fabric_lower:
        base_days = 6
    else:
        base_days = 7  # Default
    
    # Adjust based on usage
    if usage_count > 15:
        base_days = max(3, base_days - 4)
    elif usage_count > 10:
        base_days = max(4, base_days - 3)
    elif usage_count > 5:
        base_days = max(5, base_days - 2)
    
    # Adjust based on days since last wash
    if days_since_last_wash > 10:
        base_days = min(5, base_days)  # If it's been long, wash soon
    
    return max(1, base_days)
def map_fabric_to_standard(fabric_type):
    """Map fabric types to standard categories used by ML model"""
    fabric_lower = fabric_type.lower()
    
    if 'cotton' in fabric_lower:
        return 'Cotton'
    elif 'denim' in fabric_lower:
        return 'Denim'
    elif 'linen' in fabric_lower:
        return 'Linen'
    elif 'wool' in fabric_lower:
        return 'Wool'
    elif 'polyester' in fabric_lower:
        return 'Polyester'
    elif 'silk' in fabric_lower:
        return 'Silk'
    else:
        return 'Cotton'  # Default fallback

def predict_wash_cluster(fabric_type, color, care_instructions):
    """Predict wash cluster using ML model"""
    if kmeans is None or cluster_encoders is None:
        print("❌ ML models not available, using default cluster")
        return 0
    
    try:
        # Map fabric to standard category
        standard_fabric = map_fabric_to_standard(fabric_type)
        print(f"🧵 Mapped fabric '{fabric_type}' -> '{standard_fabric}'")
        
        # Extract wash parameters from care instructions using Gemini
        prompt = f"""
        Analyze these clothing care instructions and extract the following parameters:
        Care Instructions: {care_instructions}
        Fabric Type: {fabric_type}
        Color: {color}
        
        Return ONLY a JSON with these exact keys and values from these specific options:
        - wash_temperature: "Cold" or "Warm"
        - wash_cycle_type: "Normal" or "Gentle" or "Hand"
        - detergent_type: "Mild" or "Wool Detergent"
        - drying_method: "Air Dry" or "Tumble"
        
        Base your analysis on standard care label interpretations.
        """
        
        response = gemini_model.generate_content(prompt)
        response_text = response.text.strip()
        
        print(f"🤖 Gemini wash analysis: {response_text}")
        
        # Extract JSON from response
        try:
            wash_params = json.loads(response_text)
        except:
            # Default values if parsing fails
            wash_params = {
                'wash_temperature': 'Cold',
                'wash_cycle_type': 'Normal',
                'detergent_type': 'Mild',
                'drying_method': 'Air Dry'
            }
        
        # Prepare input for ML model with standardized values
        data = pd.DataFrame([{
            'fabric_type': standard_fabric,
            'color': color,
            'wash_temperature': wash_params.get('wash_temperature', 'Cold'),
            'wash_cycle_type': wash_params.get('wash_cycle_type', 'Normal'),
            'detergent_type': wash_params.get('detergent_type', 'Mild'),
            'drying_method': wash_params.get('drying_method', 'Air Dry')
        }])
        
        print(f"🧽 ML Input: {data.iloc[0].to_dict()}")
        
        # Check available categories in encoders
        for col, le in cluster_encoders.items():
            if col in data.columns:
                available_categories = list(le.classes_)
                input_value = data[col].iloc[0]
                print(f"🔍 Encoder '{col}': Available={available_categories}, Input='{input_value}'")
                
                # Handle unseen labels by mapping to closest available
                if input_value not in available_categories:
                    print(f"⚠️  Unseen label '{input_value}' in '{col}', using first available: '{available_categories[0]}'")
                    data[col] = available_categories[0]
                else:
                    data[col] = le.transform(data[col])
        
        # Predict cluster
        cluster = kmeans.predict(data)[0]
        print(f"✅ Predicted wash cluster: {cluster}")
        return cluster
        
    except Exception as e:
        print(f"❌ Error in wash cluster prediction: {e}")
        return 0

def get_ai_recommendations(user_cloth_info, dataset_df, available_clothes, num_recommendations=5):
    """Use Gemini AI to get intelligent recommendations based on user's cloth"""
    try:
        # Get actual cloth_ids from dataset
        available_cloth_ids = dataset_df['cloth_id'].tolist()
        
        prompt = f"""
        User just uploaded a clothing item with these details:
        - Name: {user_cloth_info.get('cloth_name', 'Unknown')}
        - Category: {user_cloth_info.get('category', 'Unknown')}
        - Fabric: {user_cloth_info.get('fabric_type', 'Unknown')}
        - Color: {user_cloth_info.get('color', 'Unknown')}
        - Weather: {user_cloth_info.get('weather_conditions', 'Unknown')}
        - Care Instructions: {user_cloth_info.get('care_instructions', 'Unknown')}
        
        Available clothing items in dataset (cloth_ids):
        {available_cloth_ids}
        
        Recommend exactly {num_recommendations} cloth_ids from the available list above that would complement the user's uploaded item.
        Consider:
        1. Matching fabric types for similar care requirements
        2. Complementary colors
        3. Suitable for similar weather conditions
        4. Different categories to create complete outfits
        
        Return ONLY a JSON array of actual cloth_ids like: ["C001", "C002", "C003"]
        """
        
        response = gemini_model.generate_content(prompt)
        response_text = response.text.strip()
        
        print(f"🤖 AI Recommendation Response: {response_text}")
        
        # Extract JSON array from response
        try:
            # Clean the response text
            cleaned_text = re.sub(r'```json|```', '', response_text).strip()
            cloth_ids = json.loads(cleaned_text)
            if isinstance(cloth_ids, list):
                # Filter to only include cloth_ids that exist in our dataset
                valid_cloth_ids = [cid for cid in cloth_ids if cid in available_cloth_ids]
                print(f"✅ Valid recommendations: {valid_cloth_ids}")
                return valid_cloth_ids[:num_recommendations]
        except Exception as e:
            print(f"❌ Error parsing AI response: {e}")
        
        # Fallback: extract cloth_ids using regex
        cloth_ids = re.findall(r'C\d+', response_text)
        valid_cloth_ids = [cid for cid in cloth_ids if cid in available_cloth_ids]
        if valid_cloth_ids:
            print(f"✅ Regex fallback recommendations: {valid_cloth_ids}")
            return valid_cloth_ids[:num_recommendations]
            
    except Exception as e:
        print(f"❌ Error in AI recommendation: {e}")
    
    # Final fallback: return first available clothes from dataset
    fallback_ids = dataset_df['cloth_id'].head(num_recommendations).tolist()
    print(f"🔄 Final fallback recommendations: {fallback_ids}")
    return fallback_ids

def process_image_with_genai(image_data):
    try:
        # For base64 images, we need to handle them differently
        if isinstance(image_data, str) and image_data.startswith('data:image'):
            # Extract base64 data
            image_data = image_data.split(',')[1]
        
        # Create a temporary file or process directly
        response = gemini_model.generate_content([
            "Extract all text from this care label image. Identify fabric type and washing instructions. Return in JSON format: {fabric_type: '', washing_instructions: '', full_text: ''}",
            image_data
        ])
        
        # Parse the response to extract structured data
        response_text = response.text
        print("Gemini Response:", response_text)
        
        # Try to extract JSON from response
        try:
            # Look for JSON pattern in response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                extracted_data = json.loads(json_match.group())
            else:
                # Fallback: create structured data from text
                extracted_data = {
                    'fabric_type': 'Unknown',
                    'washing_instructions': response_text,
                    'full_text': response_text
                }
        except:
            extracted_data = {
                'fabric_type': 'Unknown',
                'washing_instructions': response_text,
                'full_text': response_text
            }
            
        return extracted_data
    except Exception as e:
        print(f"Error in OCR processing: {str(e)}")
        return {
            'fabric_type': 'Unknown',
            'washing_instructions': f'Error: {str(e)}',
            'full_text': f'Error processing image: {str(e)}'
        }

def calculate_sustainability_score(user_id):
    conn = get_db_connection()
    
    # Get user's clothing data
    clothes = conn.execute('''
        SELECT * FROM clothes WHERE user_id = ?
    ''', (user_id,)).fetchall()
    
    total_items = len(clothes)
    if total_items == 0:
        return 0
    
    # Calculate score based on various factors
    score = 0
    
    for cloth in clothes:
        # Fabric type scoring
        fabric = cloth['fabric_type'].lower()
        if 'organic' in fabric or 'recycled' in fabric:
            score += 10
        elif 'cotton' in fabric:
            score += 7
        elif 'linen' in fabric:
            score += 8
        elif 'wool' in fabric:
            score += 6
        else:
            score += 5
            
        # Care instructions scoring
        care = cloth['care_instructions'].lower()
        if 'cold wash' in care or 'hand wash' in care:
            score += 3
        if 'air dry' in care or 'line dry' in care:
            score += 2
            
        # Usage frequency scoring
        if cloth['wear_count'] > 5:
            score += 2
    
    # Normalize score to 0-100
    max_possible_score = total_items * 15
    sustainability_score = min(100, (score / max_possible_score) * 100) if max_possible_score > 0 else 0
    
    conn.close()
    return round(sustainability_score)
def get_time_based_greeting():
    """Get greeting based on current time"""
    from datetime import datetime
    current_hour = datetime.now().hour
    
    if 5 <= current_hour < 12:
        return "Good morning"
    elif 12 <= current_hour < 17:
        return "Good afternoon"
    elif 17 <= current_hour < 21:
        return "Good evening"
    else:
        return "Good night"
def recommend_laundry_time(user_id):
    conn = get_db_connection()
    
    clothes = conn.execute('''
        SELECT c.*, 
               CASE 
                   WHEN c.last_worn IS NULL THEN 999
                   ELSE julianday('now') - julianday(c.last_worn) 
               END as days_since_worn,
               l.last_washed
        FROM clothes c
        LEFT JOIN laundry_tracker l ON c.id = l.cloth_id
        WHERE c.user_id = ?
    ''', (user_id,)).fetchall()
    
    recommendations = []
    
    for cloth in clothes:
        days_since_worn = cloth['days_since_worn']
        fabric = cloth['fabric_type'].lower()
        
        # Determine wash priority based on fabric and usage
        if days_since_worn > 14 or days_since_worn == 999:
            priority = "High"
        elif days_since_worn > 7:
            priority = "Medium"
        else:
            priority = "Low"
            
        # Adjust based on fabric type
        if 'silk' in fabric or 'wool' in fabric:
            priority = "Low"  # Delicate fabrics washed less frequently
        elif 'cotton' in fabric and days_since_worn > 10:
            priority = "High"
            
        recommendations.append({
            'id': cloth['id'],
            'cloth_name': cloth['cloth_name'],
            'days_since_worn': int(days_since_worn) if days_since_worn != 999 else 'Never',
            'wash_priority': priority,
            'fabric_type': cloth['fabric_type']
        })
    
    conn.close()
    return recommendations
def get_outfit_recommendations(user_id, weather_conditions):
    """Get AI-powered outfit recommendations from dataset based on user's uploaded clothes"""
    conn = get_db_connection()
    
    # Get user's most recently uploaded cloth
    user_clothes = conn.execute('''
        SELECT * FROM clothes WHERE user_id = ? ORDER BY created_at DESC LIMIT 1
    ''', (user_id,)).fetchall()
    
    # Load dataset and available clothes
    dataset_df = load_clothes_dataset()
    available_clothes = get_clothes_from_data_folder()
    
    recommendations = []
    
    if user_clothes and not dataset_df.empty:
        # Use the most recently uploaded cloth as reference
        user_cloth = user_clothes[0]
        user_cloth_info = {
            'cloth_name': user_cloth['cloth_name'],
            'category': user_cloth['category'],
            'fabric_type': user_cloth['fabric_type'],
            'color': user_cloth['color'],
            'weather_conditions': weather_conditions,
            'care_instructions': user_cloth['care_instructions']
        }
        
        print(f"🎯 Getting recommendations based on: {user_cloth_info}")
        
        # Get AI recommendations
        recommended_ids = get_ai_recommendations(user_cloth_info, dataset_df, available_clothes, 5)
        
        # Also predict wash cluster
        wash_cluster = predict_wash_cluster(
            user_cloth['fabric_type'],
            user_cloth['color'],
            user_cloth['care_instructions']
        )
        
        for cloth_id in recommended_ids:
            # Find cloth in dataset
            cloth_data = dataset_df[dataset_df['cloth_id'] == cloth_id]
            if not cloth_data.empty:
                cloth_info = cloth_data.iloc[0]
                
                # Find matching image - try multiple approaches
                image_path = None
                
                # Approach 1: Exact match with cloth_id
                for cloth in available_clothes:
                    if cloth['id'].lower() == cloth_id.lower():
                        image_path = cloth['image_path']
                        print(f"✅ Found exact image match: {cloth_id} -> {image_path}")
                        break
                
                # Approach 2: Partial match (if cloth_id is C001 and image is cloth_001.jpg)
                if not image_path:
                    for cloth in available_clothes:
                        # Try to match C001 with cloth_001, item_001, etc.
                        if cloth_id.lower() in cloth['id'].lower() or cloth['id'].lower() in cloth_id.lower():
                            image_path = cloth['image_path']
                            print(f"✅ Found partial image match: {cloth_id} -> {image_path}")
                            break
                
                # Approach 3: Use first available image as fallback
                if not image_path and available_clothes:
                    image_path = available_clothes[0]['image_path']
                    print(f"🔄 Using fallback image for {cloth_id}: {image_path}")
                
                # If still no image, create a data URI placeholder
                if not image_path:
                    # Create a simple colored placeholder
                    image_path = f'data:image/svg+xml;base64,{base64.b64encode(f"""<svg width="200" height="200" xmlns="http://www.w3.org/2000/svg"><rect width="100%" height="100%" fill="#2c5282"/><text x="50%" y="50%" font-family="Arial" font-size="14" fill="white" text-anchor="middle" dy=".3em">{cloth_id}</text></svg>""".encode()).decode()}'
                    print(f"🎨 Created SVG placeholder for {cloth_id}")
                
                recommendations.append({
                    'id': cloth_id,
                    'name': cloth_info.get('cloth_name', f'Item {cloth_id}'),
                    'category': cloth_info.get('category', 'Unknown'),
                    'fabric': cloth_info.get('fabric_type', 'Unknown'),
                    'color': cloth_info.get('color', 'Unknown'),
                    'weather': cloth_info.get('season', 'All Season'),
                    'care_instructions': cloth_info.get('care_label', 'Machine Wash'),
                    'image': image_path,
                    'wash_cluster': wash_cluster,
                    'has_real_image': not image_path.startswith('data:image')
                })
                print(f"✅ Added recommendation: {cloth_id} with image: {image_path}")
    else:
        # If no user clothes or dataset, show sample recommendations
        print("🔄 Using fallback recommendations")
        if not dataset_df.empty:
            sample_data = dataset_df.head(5)
            for _, cloth_info in sample_data.iterrows():
                cloth_id = cloth_info['cloth_id']
                
                # Find matching image
                image_path = None
                for cloth in available_clothes:
                    if cloth['id'].lower() == cloth_id.lower():
                        image_path = cloth['image_path']
                        break
                
                # Fallback to first available image or create placeholder
                if not image_path and available_clothes:
                    image_path = available_clothes[0]['image_path']
                elif not image_path:
                    image_path = f'data:image/svg+xml;base64,{base64.b64encode(f"""<svg width="200" height="200" xmlns="http://www.w3.org/2000/svg"><rect width="100%" height="100%" fill="#2c5282"/><text x="50%" y="50%" font-family="Arial" font-size="14" fill="white" text-anchor="middle" dy=".3em">{cloth_id}</text></svg>""".encode()).decode()}'
                
                recommendations.append({
                    'id': cloth_id,
                    'name': cloth_info.get('cloth_name', f'Item {cloth_id}'),
                    'category': cloth_info.get('category', 'Unknown'),
                    'fabric': cloth_info.get('fabric_type', 'Unknown'),
                    'color': cloth_info.get('color', 'Unknown'),
                    'weather': cloth_info.get('season', 'All Season'),
                    'care_instructions': cloth_info.get('care_label', 'Machine Wash'),
                    'image': image_path,
                    'wash_cluster': 0,
                    'has_real_image': not image_path.startswith('data:image')
                })
    
    print(f"🎁 Final recommendations count: {len(recommendations)}")
    
    # Store recommendation
    if recommendations:
        conn.execute('''
            INSERT INTO outfit_recommendations (user_id, recommended_items, weather_conditions)
            VALUES (?, ?, ?)
        ''', (user_id, json.dumps(recommendations), weather_conditions))
        
        conn.commit()
    
    conn.close()
    return recommendations
# Register template globals
@app.context_processor
def utility_processor():
    def get_unread_notifications_count_template(user_id):
        return get_unread_notifications_count(user_id)
    
    def get_user_notifications_template(user_id, limit=10):
        return get_user_notifications(user_id, limit)
    
    return dict(
        get_unread_notifications_count=get_unread_notifications_count_template,
        get_user_notifications=get_user_notifications_template
    )
@app.route('/api/mark_worn', methods=['POST'])
def mark_worn():
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    cloth_id = request.json.get('cloth_id')
    user_id = session['user_id']
    
    print(f"🎯 Marking cloth {cloth_id} as worn for user {user_id}")
    
    conn = get_db_connection()
    
    # First, check if the cloth exists
    cloth = conn.execute('SELECT * FROM clothes WHERE id = ? AND user_id = ?', (cloth_id, user_id)).fetchone()
    
    if not cloth:
        conn.close()
        print(f"❌ Cloth {cloth_id} not found for user {user_id}")
        return jsonify({'success': False, 'error': 'Item not found'})
    
    print(f"✅ Found cloth: {cloth['cloth_name']}")
    
    # Update the last_worn date and increment wear_count
    try:
        result = conn.execute('''
            UPDATE clothes 
            SET last_worn = DATE('now'), wear_count = wear_count + 1 
            WHERE id = ? AND user_id = ?
        ''', (cloth_id, user_id))
        
        conn.commit()
        
        # Check if update was successful
        if conn.total_changes > 0:
            print(f"✅ Successfully marked cloth {cloth_id} as worn")
            
            # Create laundry notification if needed
            create_laundry_notification(user_id, cloth_id, cloth['cloth_name'])
            
            # Get updated cloth data to verify
            updated_cloth = conn.execute('SELECT * FROM clothes WHERE id = ?', (cloth_id,)).fetchone()
            print(f"📊 Updated cloth: last_worn={updated_cloth['last_worn']}, wear_count={updated_cloth['wear_count']}")
            
            conn.close()
            return jsonify({
                'success': True, 
                'message': 'Item marked as worn successfully!',
                'last_worn': updated_cloth['last_worn'],
                'wear_count': updated_cloth['wear_count']
            })
        else:
            print(f"❌ No changes made to cloth {cloth_id}")
            conn.close()
            return jsonify({'success': False, 'error': 'No changes made'})
            
    except Exception as e:
        print(f"❌ Database error: {e}")
        conn.close()
        return jsonify({'success': False, 'error': f'Database error: {str(e)}'})

def create_laundry_notification(user_id, cloth_id, cloth_name):
    """Create a notification for laundry if this item needs washing soon"""
    conn = get_db_connection()
    
    # Get the cloth details
    cloth = conn.execute('SELECT * FROM clothes WHERE id = ?', (cloth_id,)).fetchone()
    
    if cloth:
        # Predict wash time
        wash_due_days = predict_wash_time_ai(
            fabric_type=cloth['fabric_type'],
            usage_count=cloth['wear_count'] + 1,  # Include this wear
            days_since_last_wash=0,  # Just worn today
            temperature_avg=25,
            humidity_avg=60,
            care_instructions=cloth['care_instructions']
        )
        
        # If wash is due soon (within 2 days), create notification
        if wash_due_days <= 2:
            notification_message = f"🚨 {cloth_name} needs washing soon (recommended in {wash_due_days} days)"
            
            conn.execute('''
                INSERT INTO notifications (user_id, cloth_id, message, type, is_read, created_at)
                VALUES (?, ?, ?, ?, ?, datetime('now'))
            ''', (user_id, cloth_id, notification_message, 'laundry_reminder', 0))
            
            conn.commit()
            print(f"🔔 Created laundry notification: {notification_message}")
    
    conn.close()

@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        
        conn = get_db_connection()
        try:
            conn.execute('INSERT INTO users (username, email, password) VALUES (?, ?, ?)',
                        (username, email, password))
            conn.commit()
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username or email already exists!', 'error')
        finally:
            conn.close()
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        conn = get_db_connection()
        user = conn.execute('SELECT * FROM users WHERE username = ? AND password = ?',
                           (username, password)).fetchone()
        conn.close()
        
        if user:
            session['user_id'] = user['id']
            session['username'] = user['username']
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid credentials!', 'error')
    
    return render_template('login.html')
@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user_id = session['user_id']
    conn = get_db_connection()
    
    # Get stats
    clothes_count = conn.execute('SELECT COUNT(*) FROM clothes WHERE user_id = ?', (user_id,)).fetchone()[0]
    
    sustainability_score = calculate_sustainability_score(user_id)
    laundry_recommendations = get_laundry_recommendations_with_predictions(user_id)
    
    # Count high priority laundry items (only "High" priority)
    high_priority_count = len([item for item in laundry_recommendations if item['wash_priority'] == 'High'])
    
    # Get weather and recommendations
    current_weather = get_current_weather()
    weather_recommendations = get_weather_recommendations(user_id)
    
    # Get time-based greeting
    greeting = get_time_based_greeting()
    
    conn.close()
    
    return render_template('dashboard.html',
                         clothes_count=clothes_count,
                         sustainability_score=sustainability_score,
                         laundry_recommendations=laundry_recommendations,
                         high_priority_count=high_priority_count,
                         weather_recommendations=weather_recommendations,
                         current_weather=current_weather,
                         greeting=greeting)
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        cloth_name = request.form['cloth_name']
        category = request.form['category']
        color = request.form['color']
        weather_conditions = request.form['weather_conditions']
        
        # Get captured images from hidden fields
        cloth_image_data = request.form.get('cloth_image_data')
        care_label_image_data = request.form.get('care_label_image_data')
        
        # Process care label with Gemini OCR
        care_label_data = None
        if care_label_image_data:
            care_label_data = process_image_with_genai(care_label_image_data)
        
        # Save images
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        cloth_filename = f"cloth_{session['user_id']}_{timestamp}.jpg"
        care_filename = f"care_{session['user_id']}_{timestamp}.jpg"
        
        # Save cloth image
        if cloth_image_data:
            cloth_image_path = os.path.join(app.config['UPLOAD_FOLDER'], cloth_filename)
            with open(cloth_image_path, 'wb') as f:
                f.write(base64.b64decode(cloth_image_data.split(',')[1]))
        
        # Save care label image
        if care_label_image_data:
            care_image_path = os.path.join(app.config['UPLOAD_FOLDER'], care_filename)
            with open(care_image_path, 'wb') as f:
                f.write(base64.b64decode(care_label_image_data.split(',')[1]))
        
        # Save to database
        conn = get_db_connection()
        conn.execute('''
            INSERT INTO clothes (user_id, cloth_name, category, fabric_type, color, weather_conditions, 
                               care_instructions, image_path, care_label_image_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (session['user_id'], cloth_name, category, 
              care_label_data['fabric_type'] if care_label_data else 'Unknown',
              color, weather_conditions,
              care_label_data['washing_instructions'] if care_label_data else 'Unknown',
              f'uploads/{cloth_filename}', f'uploads/{care_filename}'))
        
        conn.commit()
        conn.close()
        
        # Get current weather for context
        current_weather = get_current_weather()
        
        flash('Clothing item added successfully!', 'success')
        return render_template('upload_success.html', 
                             cloth_name=cloth_name,
                             fabric_type=care_label_data['fabric_type'] if care_label_data else 'Unknown',
                             weather_suitable=weather_conditions,
                             current_weather=current_weather)
    
    return render_template('upload.html')
def get_smart_suggestions(user_id):
    """Generate smart suggestions based on user's uploaded wardrobe"""
    conn = get_db_connection()
    
    # Get user's current wardrobe
    user_clothes = conn.execute('SELECT * FROM clothes WHERE user_id = ?', (user_id,)).fetchall()
    conn.close()
    
    if len(user_clothes) <= 1:
        return [
            "Add more items to create complete outfits",
            "Consider adding bottoms that match your tops",
            "Think about different weather conditions"
        ]
    
    # Analyze wardrobe composition
    categories = {}
    fabrics = {}
    colors = {}
    
    for cloth in user_clothes:
        cat = cloth['category']
        fabric = cloth['fabric_type']
        color = cloth['color']
        
        categories[cat] = categories.get(cat, 0) + 1
        fabrics[fabric] = fabrics.get(fabric, 0) + 1
        colors[color] = colors.get(color, 0) + 1
    
    suggestions = []
    
    # Category-based suggestions
    if categories.get('Top', 0) > categories.get('Bottom', 0):
        suggestions.append("Consider adding more bottoms to balance your wardrobe")
    if categories.get('Bottom', 0) > categories.get('Top', 0):
        suggestions.append("You might want more tops to go with your bottoms")
    
    # Fabric-based suggestions
    if len(fabrics) < 3:
        suggestions.append("Try adding different fabric types for variety")
    
    # Color-based suggestions
    if len(colors) < 4:
        suggestions.append("Adding more color variety can create more outfit options")
    
    # Care-based suggestions
    delicate_fabrics = ['silk', 'wool', 'lace']
    has_delicate = any(fabric.lower() in str(cloth['fabric_type']).lower() for cloth in user_clothes for fabric in delicate_fabrics)
    if has_delicate:
        suggestions.append("You have delicate fabrics - consider hand wash options")
    
    # Ensure we have some suggestions
    if not suggestions:
        suggestions = [
            "Your wardrobe is well-balanced!",
            "Consider seasonal items for upcoming weather changes",
            "Think about adding accessories to complete outfits"
        ]
    
    return suggestions[:5]  # Return top 5 suggestions
@app.route('/wardrobe')
def wardrobe():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user_id = session['user_id']
    conn = get_db_connection()
    
    clothes = conn.execute('SELECT * FROM clothes WHERE user_id = ? ORDER BY created_at DESC', (user_id,)).fetchall()
    conn.close()
    
    return render_template('wardrobe.html', clothes=clothes)

@app.route('/recommendations')
def recommendations():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    weather = request.args.get('weather', 'Moderate')
    recommendations = get_outfit_recommendations(session['user_id'], weather)
    
    return render_template('recommendations.html', recommendations=recommendations, weather=weather)
@app.route('/notifications')
def notifications():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user_id = session['user_id']
    notifications_list = get_user_notifications(user_id)
    unread_count = get_unread_notifications_count(user_id)
    
    return render_template('notifications.html', 
                         notifications=notifications_list,
                         unread_count=unread_count)

@app.route('/api/mark_notification_read', methods=['POST'])
def mark_notification_read():
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    notification_id = request.json.get('notification_id')
    user_id = session['user_id']
    
    mark_notification_read(notification_id, user_id)
    return jsonify({'success': True})

@app.route('/api/mark_all_notifications_read', methods=['POST'])
def mark_all_notifications_read():
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    user_id = session['user_id']
    mark_all_notifications_read(user_id)
    return jsonify({'success': True})
@app.route('/laundry')
def laundry():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    recommendations = get_laundry_recommendations_with_predictions(session['user_id'])
    return render_template('laundry.html', recommendations=recommendations)

@app.route('/sustainability')
def sustainability():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    score = calculate_sustainability_score(session['user_id'])
    return render_template('sustainability.html', sustainability_score=score)

@app.route('/profile')
def profile():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user_id = session['user_id']
    conn = get_db_connection()
    
    user = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
    clothes_count = conn.execute('SELECT COUNT(*) FROM clothes WHERE user_id = ?', (user_id,)).fetchone()[0]
    
    # Get recent activity
    recent_activity = conn.execute('''
        SELECT 'cloth_added' as type, cloth_name as description, created_at as timestamp 
        FROM clothes WHERE user_id = ?
        UNION ALL
        SELECT 'outfit_recommended' as type, weather_conditions as description, recommendation_date as timestamp
        FROM outfit_recommendations WHERE user_id = ?
        ORDER BY timestamp DESC LIMIT 10
    ''', (user_id, user_id)).fetchall()
    
    conn.close()
    
    return render_template('profile.html', 
                         user=user, 
                         clothes_count=clothes_count,
                         recent_activity=recent_activity)
@app.route('/api/chat', methods=['POST'])
def chat():
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    user_message = request.json.get('message')
    user_id = session['user_id']
    
    try:
        # Get user's wardrobe data
        conn = get_db_connection()
        user_clothes = conn.execute('SELECT * FROM clothes WHERE user_id = ?', (user_id,)).fetchall()
        conn.close()
        
        # Get current weather for context
        current_weather = get_current_weather()
        
        # Prepare wardrobe context
        wardrobe_context = []
        for cloth in user_clothes:
            wardrobe_context.append({
                'id': cloth['id'],
                'name': cloth['cloth_name'],
                'category': cloth['category'],
                'fabric': cloth['fabric_type'],
                'color': cloth['color'],
                'weather_suitable': cloth['weather_conditions'],
                'care_instructions': cloth['care_instructions']
            })
        
        # Create concise context for the chatbot
        context = f"""
        You are a fashion assistant. Be VERY concise - maximum 2 sentences. No explanations unless asked.

        USER'S WARDROBE (refer by ID only):
        {json.dumps(wardrobe_context, indent=2)}

        CURRENT BANGALORE WEATHER:
        {current_weather['temperature']}°C, {current_weather['condition']}

        RULES:
        1. Answer in 1-2 short sentences MAX
        2. Only recommend clothes user actually owns
        3. Refer to items by ID like "ID 1", "ID 2"
        4. No long explanations or descriptions
        5. Be direct and to the point

        User: {user_message}
        """
        
        chat = gemini_model.start_chat(history=[])
        gemini_response = chat.send_message(context)
        response_text = gemini_response.text.strip()
        
        # Make response even more concise if it's too long
        if len(response_text.split('. ')) > 2:
            sentences = response_text.split('. ')
            response_text = '. '.join(sentences[:2]) + '.'
        
        # Save chat history
        conn = get_db_connection()
        conn.execute('INSERT INTO chat_history (user_id, message, response) VALUES (?, ?, ?)',
                    (user_id, user_message, response_text))
        conn.commit()
        conn.close()
        
    except Exception as e:
        response_text = "Sorry, I can't help right now."
    
    return jsonify({'response': response_text})
@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out successfully!', 'success')
    return redirect(url_for('index'))
if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    if not os.path.exists(app.config['DATA_FOLDER']):
        os.makedirs(app.config['DATA_FOLDER'])
    if not os.path.exists('static/images'):
        os.makedirs('static/images')
    app.run(debug=True)