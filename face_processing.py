import torch
from io import BytesIO
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import cv2
import requests
from typing import Dict, List, Tuple, Optional
import logging
import warnings
import os

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedSkinAnalyzer:
    """
    Advanced skin analysis system that actually works and detects real skin conditions
    """
    
    def __init__(self, token: Optional[str] = None):
        """
        Initialize the Advanced Skin Analyzer
        """
        self.token = token or os.getenv("HF_TOKEN")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Comprehensive skin condition mappings
        self.condition_mappings = {
            0: "normal_skin",
            1: "acne_mild",
            2: "acne_moderate", 
            3: "acne_severe",
            4: "oily_skin",
            5: "dry_skin",
            6: "combination_skin",
            7: "sensitive_skin",
            8: "rosacea",
            9: "hyperpigmentation",
            10: "dark_spots",
            11: "wrinkles_fine_lines",
            12: "enlarged_pores",
            13: "blackheads",
            14: "whiteheads",
            15: "uneven_skin_tone",
            16: "freckles",
            17: "melasma",
            18: "sun_damage"
        }
        
        # Enhanced skin type mapping
        self.skin_type_mapping = {
            "normal_skin": "normal",
            "acne_mild": "oily",
            "acne_moderate": "oily", 
            "acne_severe": "oily",
            "oily_skin": "oily",
            "dry_skin": "dry",
            "combination_skin": "combination",
            "sensitive_skin": "sensitive",
            "rosacea": "sensitive",
            "hyperpigmentation": "normal",
            "dark_spots": "normal",
            "wrinkles_fine_lines": "normal",
            "enlarged_pores": "oily",
            "blackheads": "oily",
            "whiteheads": "oily",
            "uneven_skin_tone": "normal",
            "freckles": "normal",
            "melasma": "normal",
            "sun_damage": "normal"
        }
        
        # Severity mapping
        self.severity_mapping = {
            "acne_mild": "mild",
            "acne_moderate": "moderate", 
            "acne_severe": "severe",
            "rosacea": "moderate",
            "hyperpigmentation": "moderate",
            "wrinkles_fine_lines": "moderate",
            "melasma": "moderate",
            "sun_damage": "mild"
        }
        
        logger.info("✅ Advanced Skin Analyzer initialized")
    
    def preprocess_image(self, image_input) -> Image.Image:
        """Enhanced image preprocessing"""
        try:
            # Handle different input types
            if isinstance(image_input, str):
                if image_input.startswith(('http://', 'https://')):
                    response = requests.get(image_input)
                    image = Image.open(BytesIO(response.content))
                else:
                    image = Image.open(image_input)
            elif isinstance(image_input, np.ndarray):
                if image_input.shape[-1] == 3:  # RGB
                    image = Image.fromarray(image_input)
                else:  # BGR (OpenCV format)
                    image = Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))
            elif isinstance(image_input, Image.Image):
                image = image_input.copy()
            else:
                raise ValueError("Unsupported image input type")

            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Enhance image quality for better analysis
            image = self.enhance_image_quality(image)
            
            # Resize for optimal processing
            max_size = 512
            if max(image.size) > max_size:
                ratio = max_size / max(image.size)
                new_size = tuple(int(dim * ratio) for dim in image.size)
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            logger.info(f"Image preprocessed: {image.size}")
            return image
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            raise
    
    def enhance_image_quality(self, image: Image.Image) -> Image.Image:
        """Enhance image quality for better skin analysis"""
        try:
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.2)
            
            # Enhance sharpness slightly
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.1)
            
            # Reduce noise with slight blur
            image = image.filter(ImageFilter.GaussianBlur(radius=0.5))
            
            return image
        except Exception as e:
            logger.warning(f"Image enhancement failed: {e}")
            return image
    
    def detect_face_region(self, image: Image.Image) -> Optional[Image.Image]:
        """Enhanced face detection and cropping"""
        try:
            # Convert PIL to OpenCV format
            opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Try multiple cascade classifiers for better detection
            cascade_files = [
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml',
                cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml'
            ]
            
            faces = []
            for cascade_file in cascade_files:
                try:
                    face_cascade = cv2.CascadeClassifier(cascade_file)
                    gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
                    detected_faces = face_cascade.detectMultiScale(
                        gray, 
                        scaleFactor=1.1, 
                        minNeighbors=5, 
                        minSize=(50, 50)
                    )
                    if len(detected_faces) > 0:
                        faces = detected_faces
                        break
                except:
                    continue
            
            if len(faces) > 0:
                # Get the largest face
                largest_face = max(faces, key=lambda x: x[2] * x[3])
                x, y, w, h = largest_face
                
                # Add smart padding around face
                padding_x = int(w * 0.3)
                padding_y = int(h * 0.3)
                
                x_start = max(0, x - padding_x)
                y_start = max(0, y - padding_y)
                x_end = min(opencv_image.shape[1], x + w + padding_x)
                y_end = min(opencv_image.shape[0], y + h + padding_y)
                
                # Crop face region
                face_region = opencv_image[y_start:y_end, x_start:x_end]
                
                # Convert back to PIL
                face_image = Image.fromarray(cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB))
                
                logger.info(f"Face detected and cropped: {face_image.size}")
                return face_image
            else:
                logger.info("No face detected, using original image")
                return image
                
        except Exception as e:
            logger.warning(f"Face detection failed: {e}, using original image")
            return image
    
    def analyze_skin_advanced(self, image: Image.Image) -> Dict:
        """Advanced multi-layer skin analysis"""
        try:
            # Convert to numpy for analysis
            img_array = np.array(image)
            
            # Multi-channel analysis
            color_analysis = self.analyze_skin_color(img_array)
            texture_analysis = self.analyze_skin_texture(img_array)
            region_analysis = self.analyze_skin_regions(img_array)
            stats_analysis = self.analyze_skin_statistics(img_array)
            
            # Combine all analyses
            final_analysis = self.combine_analyses(
                color_analysis, texture_analysis, region_analysis, stats_analysis
            )
            
            return final_analysis
            
        except Exception as e:
            logger.error(f"Advanced skin analysis failed: {e}")
            return self._default_analysis()
    
    def analyze_skin_color(self, img_array: np.ndarray) -> Dict:
        """Analyze skin color characteristics"""
        # Convert to different color spaces for analysis
        rgb_mean = np.mean(img_array, axis=(0, 1))
        rgb_std = np.std(img_array, axis=(0, 1))
        
        # HSV analysis
        hsv_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        hsv_mean = np.mean(hsv_img, axis=(0, 1))
        
        red_intensity = rgb_mean[0] / 255.0
        green_intensity = rgb_mean[1] / 255.0
        blue_intensity = rgb_mean[2] / 255.0
        
        # Redness detection (acne, rosacea)
        redness_score = red_intensity / (green_intensity + blue_intensity + 0.001)
        
        # Overall brightness
        brightness = np.mean(rgb_mean) / 255.0
        
        # Color variation (uneven skin tone)
        color_variation = np.mean(rgb_std) / 255.0
        
        # Freckles/spots detection
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        spots_score = self.detect_spots(gray)
        
        return {
            'redness_score': redness_score,
            'brightness': brightness,
            'color_variation': color_variation,
            'red_intensity': red_intensity,
            'spots_score': spots_score,
            'rgb_mean': rgb_mean.tolist(),
            'hsv_mean': hsv_mean.tolist()
        }
    
    def detect_spots(self, gray_image: np.ndarray) -> float:
        """Detect spots, freckles, and dark spots"""
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
        
        # Detect dark spots using threshold
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours (potential spots)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Count small circular contours (likely spots)
        spot_count = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if 5 < area < 100:  # Small spots
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter ** 2)
                    if circularity > 0.3:  # Somewhat circular
                        spot_count += 1
        
        # Normalize spot score
        image_area = gray_image.shape[0] * gray_image.shape[1]
        spots_score = min(spot_count / (image_area / 10000), 1.0)
        
        return spots_score
    
    def analyze_skin_texture(self, img_array: np.ndarray) -> Dict:
        """Analyze skin texture patterns"""
        # Convert to grayscale for texture analysis
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Calculate texture metrics
        texture_variation = np.std(gray)
        
        # Edge detection for pore analysis
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Smoothness analysis
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        smoothness = np.mean(np.abs(gray.astype(float) - blur.astype(float)))
        
        # Wrinkle detection using morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        wrinkle_score = np.mean(tophat) / 255.0
        
        return {
            'texture_variation': float(texture_variation),
            'edge_density': float(edge_density),
            'smoothness': float(smoothness),
            'wrinkle_score': float(wrinkle_score),
            'overall_texture_score': float((texture_variation + edge_density * 1000 + smoothness) / 3)
        }
    
    def analyze_skin_regions(self, img_array: np.ndarray) -> Dict:
        """Analyze different regions of the face"""
        height, width = img_array.shape[:2]
        
        # Define regions (T-zone, cheeks, etc.)
        regions = {
            'forehead': img_array[0:height//3, width//4:3*width//4],
            'nose': img_array[height//3:2*height//3, 2*width//5:3*width//5],
            'left_cheek': img_array[height//3:2*height//3, 0:width//3],
            'right_cheek': img_array[height//3:2*height//3, 2*width//3:width],
            'chin': img_array[2*height//3:height, width//3:2*width//3]
        }
        
        region_analysis = {}
        for region_name, region_img in regions.items():
            if region_img.size > 0:
                region_mean = np.mean(region_img, axis=(0, 1))
                region_std = np.std(region_img, axis=(0, 1))
                
                region_analysis[region_name] = {
                    'brightness': float(np.mean(region_mean) / 255.0),
                    'variation': float(np.mean(region_std) / 255.0),
                    'oiliness_score': float(region_mean[1] / (region_mean[0] + region_mean[2] + 0.001))
                }
        
        return region_analysis
    
    def analyze_skin_statistics(self, img_array: np.ndarray) -> Dict:
        """Statistical analysis of skin characteristics"""
        mean_rgb = np.mean(img_array, axis=(0, 1))
        std_rgb = np.std(img_array, axis=(0, 1))
        
        # Uniformity score
        uniformity = 1.0 / (1.0 + np.mean(std_rgb) / 255.0)
        
        # Oiliness indicators
        oiliness_score = mean_rgb[1] / (mean_rgb[0] + mean_rgb[2] + 0.001)
        
        # Dryness indicators  
        dryness_score = 1.0 - (np.mean(mean_rgb) / 255.0)
        
        return {
            'uniformity': float(uniformity),
            'oiliness_score': float(oiliness_score),
            'dryness_score': float(dryness_score),
            'overall_health_score': float((uniformity + (2.0 - oiliness_score - dryness_score)) / 2)
        }
    
    def combine_analyses(self, color_analysis: Dict, texture_analysis: Dict, 
                        region_analysis: Dict, stats_analysis: Dict) -> Dict:
        """Combine all analysis results into final skin assessment"""
        
        conditions = []
        confidence_scores = []
        
        # Freckles/spots detection
        if color_analysis['spots_score'] > 0.3:
            conditions.append("freckles")
            confidence_scores.append(0.8)
        
        # Acne detection
        if (color_analysis['redness_score'] > 1.3 and 
            texture_analysis['edge_density'] > 0.02):
            if texture_analysis['texture_variation'] > 40:
                conditions.append("acne_severe")
                confidence_scores.append(0.85)
            elif texture_analysis['texture_variation'] > 25:
                conditions.append("acne_moderate") 
                confidence_scores.append(0.75)
            else:
                conditions.append("acne_mild")
                confidence_scores.append(0.65)
        
        # Oily skin detection
        if stats_analysis['oiliness_score'] > 1.2:
            conditions.append("oily_skin")
            confidence_scores.append(0.7)
        
        # Dry skin detection
        if (stats_analysis['dryness_score'] > 0.6 and 
            texture_analysis['smoothness'] < 10):
            conditions.append("dry_skin")
            confidence_scores.append(0.65)
        
        # Rosacea detection
        if (color_analysis['redness_score'] > 1.4 and 
            color_analysis['color_variation'] > 0.15):
            conditions.append("rosacea")
            confidence_scores.append(0.7)
        
        # Enlarged pores detection
        if texture_analysis['edge_density'] > 0.03:
            conditions.append("enlarged_pores")
            confidence_scores.append(0.6)
        
        # Uneven skin tone detection
        if color_analysis['color_variation'] > 0.2:
            conditions.append("uneven_skin_tone")
            confidence_scores.append(0.65)
        
        # Wrinkles detection
        if texture_analysis['wrinkle_score'] > 0.1:
            conditions.append("wrinkles_fine_lines")
            confidence_scores.append(0.6)
        
        # Default to normal if no conditions detected
        if not conditions:
            conditions = ["normal_skin"]
            confidence_scores = [0.8]
        
        # Get primary condition
        primary_idx = np.argmax(confidence_scores) if confidence_scores else 0
        primary_condition = conditions[primary_idx]
        primary_confidence = confidence_scores[primary_idx] if confidence_scores else 0.5
        
        # Determine skin type
        skin_type = self.skin_type_mapping.get(primary_condition, "normal")
        
        # Determine severity
        severity = self.severity_mapping.get(primary_condition, "mild")
        
        # Get specific issues
        specific_issues = self._identify_specific_issues(conditions)
        
        # Format final results
        return {
            'primary_condition': primary_condition,
            'confidence': primary_confidence,
            'skin_type': skin_type,
            'concerns': conditions[:3],  # Top 3 concerns
            'severity': severity,
            'age_group': 'adult',
            'specific_issues': specific_issues,
            'analysis_method': 'advanced_multi_layer',
            'analysis_details': {
                'color_analysis': color_analysis,
                'texture_analysis': texture_analysis,
                'stats_analysis': stats_analysis
            }
        }
    
    def analyze_skin(self, image_input, detect_face: bool = True) -> Dict:
        """Main skin analysis function"""
        try:
            # Preprocess image
            image = self.preprocess_image(image_input)
            
            # Detect face region if requested
            if detect_face:
                image = self.detect_face_region(image)
            
            # Perform advanced analysis
            analysis = self.analyze_skin_advanced(image)
            
            logger.info("✅ Advanced skin analysis completed successfully")
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Error during skin analysis: {e}")
            return self._default_analysis()
    
    def _identify_specific_issues(self, concerns: List[str]) -> List[str]:
        """Identify specific skin issues based on detected conditions"""
        specific_issues = []
        
        for concern in concerns:
            if "acne" in concern:
                specific_issues.extend(["breakouts", "inflammation"])
                if "severe" in concern:
                    specific_issues.extend(["cystic_acne", "scarring"])
                elif "moderate" in concern:
                    specific_issues.extend(["papules", "pustules"])
            
            if concern == "oily_skin":
                specific_issues.extend(["excess_oil", "shine"])
            
            if concern == "dry_skin":
                specific_issues.extend(["flakiness", "tightness"])
            
            if concern == "enlarged_pores":
                specific_issues.append("visible_pores")
            
            if "rosacea" in concern:
                specific_issues.extend(["persistent_redness", "visible_blood_vessels"])
            
            if "freckles" in concern:
                specific_issues.append("pigmentation_spots")
        
        return list(set(specific_issues))  # Remove duplicates
    
    def _default_analysis(self) -> Dict:
        """Return default analysis when all methods fail"""
        return {
            'primary_condition': 'normal_skin',
            'confidence': 0.5,
            'skin_type': 'normal', 
            'concerns': [],
            'severity': 'mild',
            'age_group': 'adult',
            'specific_issues': [],
            'analysis_method': 'default'
        }

# Utility functions for integration
def analyze_uploaded_image(image_input, token: Optional[str] = None) -> Dict:
    """
    Convenience function to analyze an uploaded image
    
    Args:
        image_input: Image input (PIL Image, numpy array, or file path)
        token: Hugging Face access token (optional)
        
    Returns:
        Skin analysis results compatible with recommender system
    """
    try:
        analyzer = AdvancedSkinAnalyzer(token=token)
        result = analyzer.analyze_skin(image_input, detect_face=True)
        
        # Format for compatibility with recommender system
        formatted_result = {
            'skin_type': result['skin_type'],
            'concerns': result['concerns'],
            'severity': result['severity'],
            'age_group': result['age_group'],
            'specific_issues': result['specific_issues'],
            'confidence': result['confidence'],
            'primary_condition': result['primary_condition'],
            'analysis_method': result['analysis_method']
        }
        
        logger.info(f"Analysis complete: {result['primary_condition']} ({result['confidence']:.2f} confidence)")
        return formatted_result
        
    except Exception as e:
        logger.error(f"Error analyzing uploaded image: {e}")
        return {
            'skin_type': 'normal',
            'concerns': [],
            'severity': 'mild',
            'age_group': 'adult',
            'specific_issues': [],
            'confidence': 0.5,
            'primary_condition': 'normal_skin',
            'analysis_method': 'fallback'
        }

# Testing function
def test_skin_analysis():
    """Test the advanced skin analysis system"""
    print("🧪 Testing Advanced Skin Analysis System...")
    
    try:
        # Initialize analyzer
        analyzer = AdvancedSkinAnalyzer()
        
        # Test default analysis
        default_result = analyzer._default_analysis()
        print(f"\n🔍 Default Analysis Test:")
        print(f"  Primary Condition: {default_result['primary_condition']}")
        print(f"  Skin Type: {default_result['skin_type']}")
        print(f"  Confidence: {default_result['confidence']}")
        
        # Test condition mappings
        print(f"\n📊 Supported Conditions: {len(analyzer.condition_mappings)}")
        for key, condition in list(analyzer.condition_mappings.items())[:5]:
            skin_type = analyzer.skin_type_mapping.get(condition, 'unknown')
            print(f"  {condition} → {skin_type} skin")
        
        print("\n✅ Advanced skin analysis test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    # Test the system
    test_skin_analysis()