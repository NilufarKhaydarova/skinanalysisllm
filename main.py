import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import traceback
from io import BytesIO
from PIL import Image
import base64
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Configuration
class Config:
    UPLOAD_FOLDER = 'uploads'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
    PRODUCTS_PKL_PATH = 'data.pkl'
    SECRET_KEY = 'skin-analysis-secret-key-2024'

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(Config)
CORS(app)

# Create upload directory
os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)

# Load API keys from environment variables
openai_api_key = "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"
hf_token = "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"

if not openai_api_key or not hf_token:
    logger.error("API keys not found in environment variables")
    sys.exit(1)

# Global variables for models (loaded once)
derm_analyzer = None
product_recommender = None

# Initialize LLM with updated syntax
try:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, SystemMessage
    
    llm = ChatOpenAI(
        temperature=0.4,
        model="gpt-3.5",
        api_key=openai_api_key
    )
except ImportError as e:
    logger.error(f"Failed to import LangChain modules: {e}")
    sys.exit(1)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

@app.route('/')
def index():
    try:
        return render_template('chat.html')
    except Exception as e:
        logger.error(f"Template rendering error: {e}")
        return f"Error loading page: {str(e)}", 500

@app.route('/ask', methods=['POST'])
def ask_question():
    try:
        data = request.get_json()
        question = data.get("question")

        if not question:
            return jsonify({"error": "No question provided"}), 400

        messages = [
            SystemMessage(content="You are a helpful skincare expert AI assistant."),
            HumanMessage(content=question)
        ]

        response = llm.invoke(messages)
        return jsonify({"answer": response.content})

    except Exception as e:
        logger.error(f"LLM error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": "LLM processing failed"}), 500

def initialize_models():
    global derm_analyzer, product_recommender
    try:
        logger.info("🔄 Initializing AI models...")

        logger.info("Loading Derm Foundation model...")
        from face_processing import DermFoundationAnalyzer, analyze_uploaded_image
        derm_analyzer = DermFoundationAnalyzer(token=hf_token)
        logger.info("✅ Derm Foundation model loaded")

        if os.path.exists(Config.PRODUCTS_PKL_PATH):
            logger.info("Loading product recommender...")
            from recommender import SkinProductRecommender
            product_recommender = SkinProductRecommender(Config.PRODUCTS_PKL_PATH)
            logger.info("✅ Product recommender loaded")
        else:
            logger.warning(f"⚠️ Products file not found: {Config.PRODUCTS_PKL_PATH}")
            product_recommender = None

        logger.info("🎉 All models initialized successfully!")

    except Exception as e:
        logger.error(f"❌ Error initializing models: {e}")
        logger.error(traceback.format_exc())
        raise

def process_image_base64(base64_data: str) -> Dict:
    try:
        if ',' in base64_data:
            base64_data = base64_data.split(',')[1]

        image_data = base64.b64decode(base64_data)
        image = Image.open(BytesIO(image_data))

        if derm_analyzer:
            from face_processing import analyze_uploaded_image
            analysis_result = analyze_uploaded_image(image, token=hf_token)
        else:
            return {"error": "Derm analyzer model not initialized"}

        return analysis_result

    except Exception as e:
        logger.error(f"Error processing base64 image: {e}")
        logger.error(traceback.format_exc())
        return {"error": "Failed to process image"}

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        if request.is_json:
            data = request.get_json()
            image_base64 = data.get("image_base64")
            if not image_base64:
                return jsonify({"error": "No image data provided"}), 400

            results = process_image_base64(image_base64)

            if product_recommender and 'concerns' in results:
                skin_analysis = {
                    'concerns': results['concerns'],
                    'skin_type': results.get('skin_type', 'normal'),
                    'severity': results.get('severity', 'mild')
                }
                recommendations = product_recommender.get_recommendations(skin_analysis)
                results['recommendations'] = [rec['product_name'] for rec in recommendations]

            return jsonify(results)

        return jsonify({"error": "Invalid request format"}), 400

    except Exception as e:
        logger.error(f"API error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": "Internal server error"}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    try:
        initialize_models()
        app.run(host='0.0.0.0', port=5000, debug=True)
    except Exception as e:
        logger.error(f"Failed to start application: {e}")

        sys.exit(1)