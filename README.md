# AI Skin Analysis & Product Recommendations

An intelligent skincare analysis system that uses computer vision to analyze skin conditions and provides personalized product recommendations. Built with Flask, Groq LLM, and Hugging Face models.

By: Khaydarova Nilufar

## Features

### AI-Powered Skin Analysis
- Computer Vision: Analyzes uploaded photos to detect skin type and concerns
- Multi-concern Detection: Identifies enlarged pores, uneven skin tone, acne, dryness, and more
- Confidence Scoring: Provides accuracy percentage for analysis results

### Smart Product Recommendations
- Personalized Matching: Recommends products based on your specific skin analysis
- Product Catalog: Beautiful e-commerce style product cards with prices and links
- Compatibility Scoring: Shows how well each product matches your skin needs
- Real Product Database: Uses actual skincare products with detailed information

### Intelligent Chatbot
- Multilingual Support: Responds in English, Russian, and Uzbek
- Context Awareness: Remembers your skin analysis for follow-up questions
- Expert Advice: Powered by Groq LLM for dermatologist-level skincare guidance
- Natural Conversations: Ask questions about products, routines, and skin concerns

### Modern Interface
- Responsive Design: Works seamlessly on desktop and mobile
- Beautiful UI: Gradient backgrounds, smooth animations, and modern styling
- Product Showcase: Professional product cards with images and detailed information
- Real-time Chat: Instant responses with typing indicators

## Quick Start

### Prerequisites
- Python 3.8+
- Virtual environment (recommended)
- Groq API Key
- Hugging Face Token (optional)

### Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/ai-skincare-analysis.git
cd ai-skincare-analysis
```

2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Set up environment variables
Create a .env file in the root directory:
```env
GROQ_API_KEY=your_groq_api_key_here
HF_TOKEN=your_huggingface_token_here
```

5. Prepare product database
Place your product database file as data.pkl in the root directory.

6. Run the application
```bash
python main.py
```

Visit http://localhost:5000 to start analyzing your skin.

## Requirements

Create a requirements.txt file with:

```text
Flask==2.3.3
Flask-CORS==4.0.0
langchain-groq==0.1.5
langchain-core==0.2.5
python-dotenv==1.0.0
Pillow==10.0.0
pandas==2.1.1
numpy==1.24.3
scikit-learn==1.3.0
requests==2.31.0
transformers==4.33.2
torch==2.0.1
```

## Project Structure

```
ai-skincare-analysis/
├── main.py                 # Main Flask application
├── recommender.py          # Product recommendation system
├── face_processing.py      # Computer vision for skin analysis
├── templates/
│   └── chat.html          # Frontend interface
├── static/                # CSS, JS, and assets
├── uploads/               # Temporary image storage
├── data.pkl              # Product database
├── .env                  # Environment variables
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| GROQ_API_KEY | Your Groq API key for LLM functionality | Yes |
| HF_TOKEN | Hugging Face token for computer vision models | Optional |

### Application Settings

Edit main.py to configure:
- MAX_CONTENT_LENGTH: Maximum file upload size (default: 16MB)
- ALLOWED_EXTENSIONS: Supported image formats
- PRODUCTS_PKL_PATH: Path to product database file

## API Endpoints

### POST /analyze
Analyzes uploaded image and returns skin analysis + product recommendations.

Request:
```json
{
  "image_base64": "data:image/jpeg;base64,/9j/4AAQSkZJRgABA..."
}
```

Response:
```json
{
  "skin_type": "normal",
  "concerns": ["enlarged_pores", "uneven_skin_tone"],
  "confidence": 0.85,
  "severity": "mild",
  "product_catalog": [
    {
      "id": "product_0",
      "name": "Gentle Cleanser",
      "price": "$15.99",
      "brand": "SkinCare Pro",
      "description": "A gentle cleanser for daily use",
      "url": "https://example.com/product",
      "score": 0.92,
      "reasons": ["Perfect for normal skin", "Addresses pore concerns"]
    }
  ],
  "interpretation": "Based on your skin analysis..."
}
```

### POST /ask
Handles natural language questions about skincare.

Request:
```json
{
  "question": "What should I do about enlarged pores?"
}
```

Response:
```json
{
  "answer": "For enlarged pores, I recommend..."
}
```

### GET /health
Health check endpoint for monitoring.

## Usage Examples

### Basic Skin Analysis
```python
import requests
import base64

# Read and encode image
with open('selfie.jpg', 'rb') as f:
    image_data = base64.b64encode(f.read()).decode()

# Send to API
response = requests.post('http://localhost:5000/analyze', 
                        json={'image_base64': f'data:image/jpeg;base64,{image_data}'})
result = response.json()

print(f"Skin Type: {result['skin_type']}")
print(f"Concerns: {result['concerns']}")
print(f"Recommended Products: {len(result['product_catalog'])}")
```

### Chat with AI Dermatologist
```python
import requests

response = requests.post('http://localhost:5000/ask',
                        json={'question': 'How often should I exfoliate?'})
print(response.json()['answer'])
```

## How It Works

### Image Processing
- User uploads a facial photo
- Computer vision models analyze skin features
- AI detects skin type, concerns, and calculates confidence scores

### Product Matching
- Recommendation engine processes skin analysis
- Machine learning algorithms match products to skin needs
- Compatibility scores calculated for each product

### Intelligent Responses
- Groq LLM generates personalized skincare advice
- Context-aware conversations remember previous analysis
- Multilingual support with automatic language detection

### User Interface
- Real-time chat interface with typing indicators
- Beautiful product catalog with e-commerce styling
- Responsive design for all devices

## Security & Privacy

- No Data Storage: Images are processed in memory and not stored
- Secure API Keys: Environment variables protect sensitive credentials
- Input Validation: All uploads validated for type and size
- Error Handling: Comprehensive error handling prevents crashes

## Multilingual Support

The system automatically detects and responds in:
- English: Full feature support
- Russian: Complete translations and responses
- Uzbek: Native language support

Language detection is automatic based on user input patterns.

## Deployment

### Local Development
```bash
python main.py
```

### Production Deployment
For production, consider using:
- Gunicorn: WSGI server for better performance
- Nginx: Reverse proxy for static files
- Docker: Containerization for easy deployment

Example Docker deployment:
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "main:app"]
```

## Troubleshooting

### Common Issues

1. "Product recommender not available"
- Ensure data.pkl file exists in root directory
- Check file permissions and format

2. "LLM service not available"
- Verify GROQ_API_KEY is set in .env file
- Check API key validity and quota

3. "Failed to process image"
- Verify image format is supported (PNG, JPG, JPEG, GIF, BMP, WebP)
- Check image file size (max 16MB)

4. Computer vision errors
- Set HF_TOKEN in environment variables
- Check internet connection for model downloads

### Debug Mode
Enable debug logging by setting:
```python
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

Contributions are welcome. Please fork the repository and submit pull requests.

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Acknowledgments

- Groq: For powerful LLM API
- Hugging Face: For computer vision models
- Flask Community: For the excellent web framework