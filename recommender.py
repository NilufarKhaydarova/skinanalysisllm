import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import re
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

class EnhancedSkinProductRecommender:
    def __init__(self, pkl_file_path: str):
        """
        Enhanced recommender with multilingual support and better matching
        
        Args:
            pkl_file_path: Path to the pickle file containing product dataframe
        """
        self.products_df = None
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words=None)
        self.content_matrix = None
        self.scaler = StandardScaler()
        self.load_data(pkl_file_path)
        self.preprocess_data()
        self.build_recommendation_models()
    
    def load_data(self, pkl_file_path: str):
        """Load product data from pickle file"""
        try:
            with open(pkl_file_path, 'rb') as file:
                self.products_df = pickle.load(file)
            print(f"✅ Loaded {len(self.products_df)} products from {pkl_file_path}")
        except Exception as e:
            print(f"❌ Error loading pickle file: {e}")
            raise
    
    def preprocess_data(self):
        """Clean and preprocess the product data with enhanced multilingual support"""
        # Fill missing values
        self.products_df = self.products_df.fillna("Unknown")
        
        # Clean price column
        self.products_df['Price_Numeric'] = self.products_df['Price'].apply(self.extract_price)
        
        # Create combined text features for content-based filtering
        self.products_df['Combined_Features'] = (
            self.products_df['Description'].astype(str) + " " +
            self.products_df['Ingredients'].astype(str) + " " +
            self.products_df['Тип кожи'].astype(str) + " " +
            self.products_df['По назначению'].astype(str)
        )
        
        # Enhanced multilingual cleaning
        self.products_df['Skin_Types_Clean'] = self.products_df['Тип кожи'].apply(self.clean_skin_type)
        self.products_df['Product_Purpose_Clean'] = self.products_df['По назначению'].apply(self.clean_purpose)
        
        # Add product category mapping
        self.products_df['Product_Category'] = self.products_df['Тип средства'].apply(self.categorize_product)
        
        print("✅ Enhanced data preprocessing completed")
    
    def extract_price(self, price_str: str) -> float:
        """Extract numeric price from price string (handles multiple currencies)"""
        if pd.isna(price_str) or price_str == "❌":
            return 0.0
        
        # Remove currency symbols and extract numbers
        price_clean = re.sub(r'[^\d.,]', '', str(price_str))
        numbers = re.findall(r'\d+\.?\d*', price_clean)
        if numbers:
            return float(numbers[0])
        return 0.0
    
    def clean_skin_type(self, skin_type: str) -> List[str]:
        """Enhanced multilingual skin type cleaning (Russian, Uzbek, English)"""
        if pd.isna(skin_type) or skin_type == "❌":
            return ["universal"]
        
        skin_type = str(skin_type).lower()
        types = []
        
        # Russian, Uzbek, and English mappings
        if any(word in skin_type for word in ["сухая", "сухой", "quruq", "dry"]):
            types.append("dry")
        if any(word in skin_type for word in ["жирная", "жирный", "yog'li", "yoğli", "oily"]):
            types.append("oily")
        if any(word in skin_type for word in ["комбинированная", "комбинированный", "kombinatsiyali", "combination", "смешанная"]):
            types.append("combination") 
        if any(word in skin_type for word in ["чувствительная", "чувствительный", "sezgir", "sensitive"]):
            types.append("sensitive")
        if any(word in skin_type for word in ["нормальная", "нормальный", "oddiy", "normal"]):
            types.append("normal")
        if any(word in skin_type for word in ["проблемная", "проблемный", "проблематичная", "muammoli", "problem"]):
            types.append("problem")
        if any(word in skin_type for word in ["все", "всех", "универсальная", "универсальный", "barcha", "all", "universal"]):
            types.append("universal")
        
        return types if types else ["universal"]
    
    def clean_purpose(self, purpose: str) -> List[str]:
        """Enhanced multilingual product purpose cleaning"""
        if pd.isna(purpose) or purpose == "❌":
            return ["general"]
        
        purpose = str(purpose).lower()
        purposes = []
        
        # Russian, Uzbek, and English mappings
        if any(word in purpose for word in ["очищение", "очищающ", "tozalash", "tozalovchi", "cleansing", "умывание", "очистка"]):
            purposes.append("cleansing")
        if any(word in purpose for word in ["увлажнение", "увлажняющ", "namlantirish", "namlantiruvchi", "moisturizing", "гидрат"]):
            purposes.append("moisturizing")
        if any(word in purpose for word in ["питание", "питательн", "oziqlanish", "oziqlanuvchi", "nourishing", "питающ"]):
            purposes.append("nourishing")
        if any(word in purpose for word in ["против старения", "антивозрастн", "anti-aging", "омолажив", "возраст", "qarshilash", "yoshartirish"]):
            purposes.append("anti-aging")
        if any(word in purpose for word in ["акне", "прыщи", "akne", "acne", "угри", "воспаления"]):
            purposes.append("acne")
        if any(word in purpose for word in ["отбеливание", "осветляющ", "oqartirish", "yoruglashtirish", "whitening", "brightening", "пигмент"]):
            purposes.append("brightening")
        if any(word in purpose for word in ["защита", "защитн", "himoya", "himoyalovchi", "protection", "spf", "солнцезащит"]):
            purposes.append("protection")
        if any(word in purpose for word in ["лечение", "лечебн", "davolash", "davolovchi", "treatment", "терапия"]):
            purposes.append("treatment")
        if any(word in purpose for word in ["тонизирование", "тонизирующ", "tonizatsiya", "toning", "тоник"]):
            purposes.append("toning")
        if any(word in purpose for word in ["отшелушивание", "пилинг", "tozalash", "exfoliating", "скраб"]):
            purposes.append("exfoliating")
        if any(word in purpose for word in ["поры", "пор", "teshik", "pore", "pores"]):
            purposes.append("pore-minimizing")
        
        return purposes if purposes else ["general"]
    
    def categorize_product(self, product_type: str) -> str:
        """Categorize product type for better matching"""
        if pd.isna(product_type):
            return "other"
        
        product_type = str(product_type).lower()
        
        if any(word in product_type for word in ["крем", "krem", "cream"]):
            return "moisturizer"
        elif any(word in product_type for word in ["гель", "gel", "очищение", "tozalash", "cleanser"]):
            return "cleanser"
        elif any(word in product_type for word in ["сыворотка", "serum", "эссенция"]):
            return "serum"
        elif any(word in product_type for word in ["маска", "mask", "niqob"]):
            return "mask"
        elif any(word in product_type for word in ["тоник", "tonic", "tonik"]):
            return "toner"
        elif any(word in product_type for word in ["лосьон", "lotion"]):
            return "lotion"
        else:
            return "other"
    
    def build_recommendation_models(self):
        """Build enhanced TF-IDF matrix for content-based recommendations"""
        try:
            # Create TF-IDF matrix from combined features
            self.content_matrix = self.tfidf_vectorizer.fit_transform(self.products_df['Combined_Features'])
            print("✅ Enhanced recommendation models built successfully")
        except Exception as e:
            print(f"❌ Error building recommendation models: {e}")
            raise
    
    def analyze_skin_condition(self, skin_analysis: Dict) -> Dict:
        """Enhanced skin condition analysis with better product need mapping"""
        analysis = {
            'skin_type': skin_analysis.get('skin_type', 'normal'),
            'concerns': skin_analysis.get('concerns', []),
            'severity': skin_analysis.get('severity', 'mild'),
            'age_group': skin_analysis.get('age_group', 'adult'),
            'specific_issues': skin_analysis.get('specific_issues', []),
            'primary_condition': skin_analysis.get('primary_condition', 'normal_skin')
        }
        
        # Enhanced product need mapping based on detected conditions
        product_needs = []
        
        # Map specific conditions to product needs
        for concern in analysis['concerns']:
            if 'acne' in concern:
                product_needs.extend(['acne', 'cleansing', 'treatment'])
                if 'severe' in concern:
                    product_needs.append('medical')
            elif concern == 'oily_skin':
                product_needs.extend(['cleansing', 'toning', 'oil-control'])
            elif concern == 'dry_skin':
                product_needs.extend(['moisturizing', 'nourishing', 'hydrating'])
            elif concern == 'sensitive_skin':
                product_needs.extend(['gentle', 'soothing', 'hypoallergenic'])
            elif concern == 'rosacea':
                product_needs.extend(['anti-inflammatory', 'gentle', 'soothing'])
            elif concern == 'enlarged_pores':
                product_needs.extend(['pore-minimizing', 'toning', 'exfoliating'])
            elif concern == 'uneven_skin_tone':
                product_needs.extend(['brightening', 'exfoliating', 'vitamin-c'])
            elif concern == 'freckles':
                product_needs.extend(['brightening', 'protection', 'vitamin-c'])
            elif concern == 'wrinkles_fine_lines':
                product_needs.extend(['anti-aging', 'moisturizing', 'treatment'])
            elif 'hyperpigmentation' in concern or 'dark_spots' in concern:
                product_needs.extend(['brightening', 'treatment', 'vitamin-c'])
        
        # Add general needs based on skin type
        if analysis['skin_type'] == 'oily':
            product_needs.extend(['oil-control', 'cleansing'])
        elif analysis['skin_type'] == 'dry':
            product_needs.extend(['moisturizing', 'nourishing'])
        elif analysis['skin_type'] == 'combination':
            product_needs.extend(['balancing', 'cleansing'])
        elif analysis['skin_type'] == 'sensitive':
            product_needs.extend(['gentle', 'hypoallergenic'])
        
        # Remove duplicates and add fallback
        analysis['product_needs'] = list(set(product_needs)) if product_needs else ['general', 'moisturizing']
        
        return analysis
    
    def get_recommendations(self, skin_analysis: Dict, top_k: int = 5) -> List[Dict]:
        """Get enhanced product recommendations with better scoring"""
        analysis = self.analyze_skin_condition(skin_analysis)
        
        # Multi-stage filtering and scoring
        # Stage 1: Filter by skin type compatibility
        filtered_df = self.filter_by_skin_type(analysis['skin_type'])
        
        # Stage 2: Filter by product needs/concerns
        filtered_df = self.filter_by_concerns(filtered_df, analysis['product_needs'])
        
        # Stage 3: Calculate comprehensive recommendation scores
        recommendations = self.calculate_enhanced_scores(filtered_df, analysis)
        
        # Sort by score and return top k
        recommendations = sorted(recommendations, key=lambda x: x['score'], reverse=True)
        
        return recommendations[:top_k]
    
    def get_recommendations_with_details(self, skin_analysis: Dict, top_k: int = 5) -> Dict:
        """Get detailed product recommendations with explanatory text for LLM"""
        analysis = self.analyze_skin_condition(skin_analysis)
        recommendations = self.get_recommendations(skin_analysis, top_k)
        
        # Generate explanatory text
        explanation = self.generate_recommendation_explanation(analysis, recommendations)
        
        return {
            'analysis': analysis,
            'recommendations': recommendations,
            'explanation': explanation,
            'recommendation_text': self.format_recommendations_for_display(analysis, recommendations)
        }
    
    def generate_recommendation_explanation(self, analysis: Dict, recommendations: List[Dict]) -> str:
        """Generate multilingual explanation for recommendations"""
        skin_type = analysis['skin_type']
        concerns = analysis['concerns']
        product_needs = analysis['product_needs']
        
        explanation = f"Based on your {skin_type} skin type"
        
        if concerns:
            concern_names = []
            for concern in concerns:
                if concern == 'enlarged_pores':
                    concern_names.append('enlarged pores')
                elif concern == 'uneven_skin_tone':
                    concern_names.append('uneven skin tone')
                elif 'acne' in concern:
                    concern_names.append('acne')
                elif concern == 'freckles':
                    concern_names.append('freckles/pigmentation')
                else:
                    concern_names.append(concern.replace('_', ' '))
            
            explanation += f" and detected concerns ({', '.join(concern_names[:3])})"
        
        explanation += f", I've selected {len(recommendations)} products specifically targeting: {', '.join(product_needs[:4])}."
        
        return explanation
    
    def format_recommendations_for_display(self, analysis: Dict, recommendations: List[Dict]) -> str:
        """Format recommendations for beautiful display"""
        if not recommendations:
            return "I couldn't find specific product recommendations, but I'd be happy to suggest general skincare tips!"
        
        # This will be handled by the multilingual LLM in main.py
        return ""  # Let the LLM format this properly
    
    def filter_by_skin_type(self, skin_type: str) -> pd.DataFrame:
        """Enhanced skin type filtering with better logic"""
        skin_type = skin_type.lower()
        
        # Create mask for skin type compatibility
        mask = self.products_df['Skin_Types_Clean'].apply(
            lambda x: (
                skin_type in x or 
                'universal' in x or 
                'all' in str(x).lower() or
                (skin_type == 'combination' and ('normal' in x or 'oily' in x)) or
                (skin_type == 'sensitive' and 'gentle' in str(x).lower())
            )
        )
        
        filtered_df = self.products_df[mask].copy()
        
        # If no specific matches, return universal products
        if len(filtered_df) == 0:
            mask = self.products_df['Skin_Types_Clean'].apply(
                lambda x: 'universal' in x or 'all' in str(x).lower()
            )
            filtered_df = self.products_df[mask].copy()
        
        print(f"Filtered {len(filtered_df)} products for {skin_type} skin type")
        return filtered_df
    
    def filter_by_concerns(self, df: pd.DataFrame, concerns: List[str]) -> pd.DataFrame:
        """Enhanced filtering by skin concerns with fuzzy matching"""
        if not concerns:
            return df
        
        # Create flexible matching for concerns
        mask = df['Product_Purpose_Clean'].apply(
            lambda x: any(
                concern in x or 
                any(synonym in x for synonym in self.get_concern_synonyms(concern))
                for concern in concerns
            )
        )
        
        # Also check product categories
        category_mask = df['Product_Category'].apply(
            lambda x: any(self.matches_concern_category(x, concern) for concern in concerns)
        )
        
        # Combine masks
        combined_mask = mask | category_mask
        filtered_df = df[combined_mask].copy()
        
        # If no matches, return original dataframe
        if len(filtered_df) == 0:
            print(f"No products found for concerns {concerns}, returning all products")
            return df
        
        print(f"Filtered {len(filtered_df)} products for concerns: {concerns}")
        return filtered_df
    
    def get_concern_synonyms(self, concern: str) -> List[str]:
        """Get synonyms for better matching"""
        synonyms = {
            'acne': ['угри', 'прыщи', 'воспаления', 'breakouts'],
            'cleansing': ['очищение', 'умывание', 'tozalash'],
            'moisturizing': ['увлажнение', 'namlantirish', 'hydrating'],
            'anti-aging': ['антивозрастной', 'омолаживающий', 'yoshartirish'],
            'brightening': ['отбеливание', 'осветление', 'oqartirish'],
            'pore-minimizing': ['поры', 'teshiklar', 'pores'],
            'oil-control': ['контроль жирности', 'yog nazorati'],
            'gentle': ['мягкий', 'деликатный', 'yumshoq'],
            'soothing': ['успокаивающий', 'тинчлантирувчи'],
            'protection': ['защита', 'himoya', 'spf']
        }
        return synonyms.get(concern, [concern])
    
    def matches_concern_category(self, category: str, concern: str) -> bool:
        """Check if product category matches concern"""
        category_matches = {
            'acne': ['cleanser', 'treatment', 'serum'],
            'cleansing': ['cleanser', 'toner'],
            'moisturizing': ['moisturizer', 'cream', 'lotion'],
            'anti-aging': ['serum', 'cream', 'treatment'],
            'brightening': ['serum', 'treatment', 'mask'],
            'pore-minimizing': ['toner', 'serum', 'mask'],
            'gentle': ['moisturizer', 'cleanser'],
            'protection': ['moisturizer', 'cream']
        }
        
        return category in category_matches.get(concern, [])
    
    def calculate_enhanced_scores(self, filtered_df: pd.DataFrame, analysis: Dict) -> List[Dict]:
        """Calculate enhanced recommendation scores with multiple factors"""
        recommendations = []
        
        for idx, row in filtered_df.iterrows():
            score = 0.0
            reasons = []
            
            # Base score for skin type match (30%)
            skin_type_match = self.calculate_skin_type_score(row['Skin_Types_Clean'], analysis['skin_type'])
            score += skin_type_match * 0.3
            if skin_type_match > 0.5:
                reasons.append(f"Perfect for {analysis['skin_type']} skin")
            
            # Concern/need match score (40%)
            concern_score = self.calculate_concern_score(row, analysis['product_needs'])
            score += concern_score * 0.4
            if concern_score > 0.3:
                matching_needs = [need for need in analysis['product_needs'] 
                                if need in row['Product_Purpose_Clean']]
                if matching_needs:
                    reasons.append(f"Targets {', '.join(matching_needs[:2])}")
            
            # Ingredient quality bonus (15%)
            ingredient_score = self.calculate_ingredient_score(row['Ingredients'])
            score += ingredient_score * 0.15
            if ingredient_score > 0.5:
                reasons.append("Contains quality active ingredients")
            
            # Price consideration (10%)
            price_score = self.calculate_price_score(row['Price_Numeric'])
            score += price_score * 0.1
            
            # Brand/country bonus (5%)
            brand_score = self.calculate_brand_score(row['Производитель'], row['Страна'])
            score += brand_score * 0.05
            if brand_score > 0.5:
                reasons.append("Trusted brand")
            
            # Ensure minimum reasons
            if not reasons:
                reasons.append("Suitable for your skin type")
            
            recommendations.append({
                'product_name': row['Product Name'],
                'url': self.clean_product_url(row['URL']),
                'price': row['Price'],
                'description': row['Description'],
                'ingredients': row['Ingredients'],
                'skin_type': row['Тип кожи'],
                'purpose': row['По назначению'],
                'manufacturer': row['Производитель'],
                'country': row['Страна'],
                'volume': row['Объём (мл)'],
                'score': min(score, 1.0),  # Cap at 1.0
                'reasons': reasons[:3]  # Limit to top 3 reasons
            })
        
        return recommendations
    
    def calculate_skin_type_score(self, product_skin_types: List[str], user_skin_type: str) -> float:
        """Calculate skin type compatibility score"""
        if user_skin_type in product_skin_types:
            return 1.0
        elif 'universal' in product_skin_types or 'all' in product_skin_types:
            return 0.8
        elif user_skin_type == 'combination' and ('normal' in product_skin_types or 'oily' in product_skin_types):
            return 0.7
        elif user_skin_type == 'sensitive' and any(t in ['normal', 'dry'] for t in product_skin_types):
            return 0.6
        else:
            return 0.3
    
    def calculate_concern_score(self, row: pd.Series, user_needs: List[str]) -> float:
        """Calculate how well product addresses user concerns"""
        product_purposes = row['Product_Purpose_Clean']
        product_category = row['Product_Category']
        
        matches = 0
        total_needs = len(user_needs)
        
        for need in user_needs:
            # Direct purpose match
            if need in product_purposes:
                matches += 1
            # Category match
            elif self.matches_concern_category(product_category, need):
                matches += 0.5
            # Synonym match
            elif any(synonym in product_purposes for synonym in self.get_concern_synonyms(need)):
                matches += 0.7
        
        return min(matches / max(total_needs, 1), 1.0)
    
    def calculate_ingredient_score(self, ingredients: str) -> float:
        """Score based on ingredient quality"""
        if pd.isna(ingredients) or ingredients == "❌" or len(str(ingredients)) < 10:
            return 0.2
        
        ingredients = str(ingredients).lower()
        
        # High-value ingredients
        premium_ingredients = [
            'hyaluronic acid', 'retinol', 'vitamin c', 'niacinamide', 'peptides',
            'ceramides', 'salicylic acid', 'glycolic acid', 'vitamin e',
            'гиалуроновая кислота', 'ретинол', 'витамин c', 'ниацинамид',
            'салициловая кислота', 'витамин е'
        ]
        
        score = 0.3  # Base score
        for ingredient in premium_ingredients:
            if ingredient in ingredients:
                score += 0.1
        
        return min(score, 1.0)
    
    def calculate_price_score(self, price: float) -> float:
        """Score based on price (moderate prices get higher scores)"""
        if price <= 0:
            return 0.5
        
        # Optimal price range (adjust based on your market)
        if 15 <= price <= 50:
            return 1.0
        elif 10 <= price <= 70:
            return 0.8
        elif price < 10:
            return 0.6  # Too cheap might be low quality
        else:
            return 0.4  # Too expensive
    
    def calculate_brand_score(self, manufacturer: str, country: str) -> float:
        """Score based on brand reputation and country"""
        if pd.isna(manufacturer):
            return 0.3
        
        # Known quality brands (add more as needed)
        premium_brands = [
            'la roche-posay', 'cerave', 'neutrogena', 'olay', 'nivea',
            'vichy', 'eucerin', 'avene', 'bioderma', 'pharmaceris'
        ]
        
        manufacturer_lower = str(manufacturer).lower()
        if any(brand in manufacturer_lower for brand in premium_brands):
            return 1.0
        
        # Country-based scoring
        quality_countries = ['usa', 'france', 'germany', 'south korea', 'japan', 'switzerland']
        if any(country_name in str(country).lower() for country_name in quality_countries):
            return 0.7
        
        return 0.5
    
    def clean_product_url(self, url: str) -> str:
        """Clean and validate product URL"""
        if not url or url == "❌" or pd.isna(url):
            return ""
        
        url = str(url).strip()
        
        # Basic URL validation and cleaning
        if url.startswith(('http://', 'https://')):
            return url
        elif url.startswith('www.'):
            return f"https://{url}"
        elif '.' in url and not url.startswith(('mailto:', 'tel:')):
            return f"https://{url}"
        else:
            return ""  # Invalid URL
    
    def get_product_by_category(self, category: str, skin_type: str = None, limit: int = 10) -> List[Dict]:
        """Get products by category with enhanced filtering"""
        category_mapping = {
            'cleanser': ['очищение', 'cleansing', 'гель для умывания', 'tozalash'],
            'moisturizer': ['увлажнение', 'moisturizing', 'крем', 'namlantirish'],
            'serum': ['сыворотка', 'serum', 'эссенция'],
            'treatment': ['лечение', 'treatment', 'акне', 'davolash'],
            'sunscreen': ['защита', 'spf', 'солнцезащитный', 'himoya'],
            'mask': ['маска', 'mask', 'niqob'],
            'toner': ['тоник', 'tonic', 'тонер']
        }
        
        category_terms = category_mapping.get(category.lower(), [category])
        
        # Filter by category
        mask = self.products_df['Product_Purpose_Clean'].apply(
            lambda x: any(term.lower() in str(x).lower() for term in category_terms)
        ) | self.products_df['Тип средства'].apply(
            lambda x: any(term.lower() in str(x).lower() for term in category_terms)
        )
        
        filtered_df = self.products_df[mask].copy()
        
        # Further filter by skin type if specified
        if skin_type:
            filtered_df = self.filter_by_skin_type_simple(filtered_df, skin_type)
        
        # Convert to list of dictionaries
        products = []
        for idx, row in filtered_df.head(limit).iterrows():
            products.append({
                'product_name': row['Product Name'],
                'url': self.clean_product_url(row['URL']),
                'price': row['Price'],
                'description': row['Description'],
                'ingredients': row['Ingredients'],
                'skin_type': row['Тип кожи'],
                'purpose': row['По назначению'],
                'manufacturer': row['Производитель'],
                'volume': row['Объём (мл)']
            })
        
        return products
    
    def filter_by_skin_type_simple(self, df: pd.DataFrame, skin_type: str) -> pd.DataFrame:
        """Simple skin type filtering for category searches"""
        skin_mask = df['Skin_Types_Clean'].apply(
            lambda x: skin_type.lower() in x or 'universal' in x or 'all' in x
        )
        return df[skin_mask]
    
    def get_statistics(self) -> Dict:
        """Get enhanced dataset statistics"""
        return {
            'total_products': len(self.products_df),
            'manufacturers': self.products_df['Производитель'].nunique(),
            'countries': self.products_df['Страна'].nunique(),
            'categories': self.products_df['Product_Category'].value_counts().to_dict(),
            'price_range': {
                'min': self.products_df['Price_Numeric'].min(),
                'max': self.products_df['Price_Numeric'].max(),
                'avg': self.products_df['Price_Numeric'].mean(),
                'median': self.products_df['Price_Numeric'].median()
            },
            'skin_types_coverage': self.products_df['Тип кожи'].value_counts().to_dict(),
            'purposes_coverage': self.products_df['По назначению'].value_counts().head(10).to_dict()
        }

# Alias for backward compatibility
SkinProductRecommender = EnhancedSkinProductRecommender

# Testing function
def test_recommender(pkl_file_path: str):
    """Test the enhanced recommender system"""
    print("🧪 Testing Enhanced Skin Product Recommender System...")
    
    try:
        # Initialize recommender
        recommender = EnhancedSkinProductRecommender(pkl_file_path)
        
        # Print statistics
        stats = recommender.get_statistics()
        print(f"\n📊 Dataset Statistics:")
        print(f"Total Products: {stats['total_products']}")
        print(f"Manufacturers: {stats['manufacturers']}")
        print(f"Countries: {stats['countries']}")
        print(f"Categories: {list(stats['categories'].keys())}")
        print(f"Price Range: ${stats['price_range']['min']:.2f} - ${stats['price_range']['max']:.2f}")
        
        # Test recommendation with sample skin analysis
        sample_analysis = {
            'skin_type': 'oily',
            'concerns': ['enlarged_pores', 'acne_mild'],
            'severity': 'mild',
            'age_group': 'young_adult',
            'specific_issues': ['visible_pores', 'breakouts']
        }
        
        print(f"\n🔍 Testing recommendations for: {sample_analysis}")
        detailed_recommendations = recommender.get_recommendations_with_details(sample_analysis, top_k=3)
        
        print(f"\n💡 Explanation: {detailed_recommendations['explanation']}")
        print(f"\n📝 Top Recommendations:")
        for i, rec in enumerate(detailed_recommendations['recommendations'], 1):
            print(f"{i}. {rec['product_name']} - {rec['price']}")
            print(f"   Score: {rec['score']:.2f} | Reasons: {', '.join(rec['reasons'])}")
            print(f"   URL: {rec['url'][:50]}..." if rec['url'] else "   No URL available")
        
        print("\n✅ Enhanced recommender system test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise

if __name__ == "__main__":
    # Example usage
    pkl_file_path = "data.pkl"  # Replace with your actual pkl file path
    
    # Test the system
    test_recommender(pkl_file_path)