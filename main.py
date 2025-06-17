import sys
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import traceback
from io import BytesIO
from PIL import Image
import os
import base64
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

# Load environment variables from .env file
load_dotenv()

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

# Global variables for models
product_recommender = None
llm = None

def initialize_llm():
    """Initialize Groq LLM"""
    global llm
    try:
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            logger.error("GROQ_API_KEY not found in environment variables")
            return False
        
        llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name="llama3-70b-8192",
            temperature=0.7,
            max_tokens=1024
        )
        
        logger.info("✅ Groq LLM initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize Groq LLM: {e}")
        return False

def initialize_models():
    """Initialize all AI models"""
    global product_recommender
    
    try:
        logger.info("🔄 Initializing AI models...")

        # Initialize LLM
        if not initialize_llm():
            logger.warning("⚠️ LLM initialization failed")

        # Initialize product recommender
        if os.path.exists(Config.PRODUCTS_PKL_PATH):
            logger.info("Loading product recommender...")
            from recommender import SkinProductRecommender
            product_recommender = SkinProductRecommender(Config.PRODUCTS_PKL_PATH)
            logger.info("✅ Product recommender loaded")
        else:
            logger.warning(f"⚠️ Products file not found: {Config.PRODUCTS_PKL_PATH}")
            product_recommender = None

        logger.info("🎉 Models initialization completed!")

    except Exception as e:
        logger.error(f"❌ Error initializing models: {e}")
        logger.error(traceback.format_exc())
        raise

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

def process_image_base64(base64_data: str) -> Dict:
    """Process base64 image and return analysis results"""
    try:
        if ',' in base64_data:
            base64_data = base64_data.split(',')[1]

        image_data = base64.b64decode(base64_data)
        image = Image.open(BytesIO(image_data))

        # Use the face processing system
        hf_token = os.getenv("HF_TOKEN", "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")
        from face_processing import analyze_uploaded_image
        analysis_result = analyze_uploaded_image(image, token=hf_token)

        return analysis_result

    except Exception as e:
        logger.error(f"Error processing base64 image: {e}")
        logger.error(traceback.format_exc())
        return {"error": "Failed to process image"}

# Routes
@app.route('/')
def index():
    try:
        return render_template('chat.html')
    except Exception as e:
        logger.error(f"Template rendering error: {e}")
        return f"Error loading page: {str(e)}", 500

@app.route('/ask', methods=['POST'])
def ask_question():
    """Handle general skincare questions with Groq LLM"""
    try:
        data = request.get_json()
        question = data.get("question")

        if not question:
            return jsonify({"error": "No question provided"}), 400

        if not llm:
            return jsonify({"error": "LLM service not available. Please set GROQ_API_KEY."}), 503

        # Detect language and create appropriate prompt
        question_lower = question.lower()
        
        # Language detection
        is_russian = any(char in question for char in 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя')
        is_uzbek = any(word in question_lower for word in ['salom', 'tushunmadim', 'nima', 'qanday', 'rahmat', 'yordam', 'teri', 'mahsulot'])
        system_prompt="Based on the language, provide thorough response with specific dermatologist knowledge in the given language based in the recommendations"

        messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=question)
            ]
        
        response = llm.invoke(messages)
        
        return jsonify({"answer": response.content})

    except Exception as e:
        logger.error(f"Question processing error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": "Failed to process question"}), 500

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze uploaded image and provide skincare recommendations"""
    try:
        if request.is_json:
            data = request.get_json()
            image_base64 = data.get("image_base64")
            if not image_base64:
                return jsonify({"error": "No image data provided"}), 400

            # Process the image
            results = process_image_base64(image_base64)
            
            if "error" in results:
                return jsonify(results), 400

            # Get product recommendations
            results['product_catalog'] = []
            recommendations = []
            
            if product_recommender:
                skin_analysis = {
                    'concerns': results.get('concerns', []),
                    'skin_type': results.get('skin_type', 'normal'),
                    'severity': results.get('severity', 'mild')
                }
                
                if not skin_analysis['concerns']:
                    skin_analysis['concerns'] = ['general', 'moisturizing']
                
                try:
                    detailed_recs = product_recommender.get_recommendations_with_details(skin_analysis, top_k=4)
                    recommendations = detailed_recs.get('recommendations', [])
                    
                    # Convert each recommendation to product card format
                    for i, rec in enumerate(recommendations):
                        product = {
                            'id': f"product_{i}",
                            'name': rec.get('product_name', f'Product {i+1}'),
                            'price': rec.get('price', 'Price not available'),
                            'brand': rec.get('manufacturer', 'Unknown Brand'),
                            'description': rec.get('description', 'No description available'),
                            'purpose': rec.get('purpose', 'General skincare'),
                            'url': rec.get('url', ''),
                            'score': rec.get('score', 0.7),
                            'reasons': rec.get('reasons', ['Recommended for your skin'])
                        }
                        results['product_catalog'].append(product)
                    
                    results['recommendations'] = [rec.get('product_name', 'Unknown') for rec in recommendations]
                    
                except Exception as e:
                    logger.error(f"Error getting recommendations: {e}")

            # Generate interpretation
            if llm:
                try:
                    concerns = results.get('concerns', [])
                    skin_type = results.get('skin_type', 'normal')
                    confidence = results.get('confidence', 0)
                    product_names = []
                    if recommendations:  # This is the list of recommendation objects
                        product_names = [rec.get('product_name', 'Unknown Product') for rec in recommendations]
                    
                    recommendations_text = ""
                    if product_names:
                        recommendations_text = f"- Recommended Products: {', '.join(product_names)}"
                    
                    prompt = f"""Analyze this skin info
- Skin Type: {skin_type}
- Detected Concerns: {', '.join(concerns) if concerns else 'None detected'}
- Analysis Confidence: {confidence:.1%}
{recommendations_text}
"""
                    
                    messages = [
                        SystemMessage(content="You are a dermatologist. Provide thorough explanation as normal text with no formatting"),
                        HumanMessage(content=prompt)
                    ]
                    
                    response = llm.invoke(messages)
                    results['interpretation'] = response.content
                    
                except Exception as e:
                    logger.error(f"LLM error: {e}")
                    results['interpretation'] = """SORRY ISSUES WITH LLM"""

            logger.info(f"Returning {len(results.get('product_catalog', []))} products")
            return jsonify(results)

        return jsonify({"error": "Invalid request format"}), 400

    except Exception as e:
        logger.error(f"API error: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    status = {
        "status": "healthy",
        "llm_available": llm is not None,
        "product_recommender": product_recommender is not None,
        "products_file_exists": os.path.exists(Config.PRODUCTS_PKL_PATH)
    }
    return jsonify(status)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    try:
        initialize_models()
        logger.info("🚀 Starting Flask application...")
        app.run(host='0.0.0.0', port=5000, debug=True)
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        sys.exit(1)