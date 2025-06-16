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

class SkinProductRecommender:
    def __init__(self, pkl_file_path: str):
        """
        Initialize the recommender with product data from pickle file
        
        Args:
            pkl_file_path: Path to the pickle file containing product dataframe
        """
        self.products_df = None
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
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
        """Clean and preprocess the product data"""
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
        
        # Clean and standardize skin types
        self.products_df['Skin_Types_Clean'] = self.products_df['Тип кожи'].apply(self.clean_skin_type)
        self.products_df['Product_Purpose_Clean'] = self.products_df['По назначению'].apply(self.clean_purpose)
        
        print("✅ Data preprocessing completed")
    
    def extract_price(self, price_str: str) -> float:
        """Extract numeric price from price string"""
        if pd.isna(price_str) or price_str == "❌":
            return 0.0
        
        # Extract numbers from price string
        numbers = re.findall(r'\d+\.?\d*', str(price_str))
        if numbers:
            return float(numbers[0])
        return 0.0
    
    def clean_skin_type(self, skin_type: str) -> List[str]:
        """Clean and standardize skin type information"""
        if pd.isna(skin_type) or skin_type == "❌":
            return ["Universal"]
        
        skin_type = str(skin_type).lower()
        types = []
        
        if "сухая" in skin_type or "dry" in skin_type:
            types.append("dry")
        if "жирная" in skin_type or "oily" in skin_type:
            types.append("oily")
        if "комбинированная" in skin_type or "combination" in skin_type:
            types.append("combination") 
        if "чувствительная" in skin_type or "sensitive" in skin_type:
            types.append("sensitive")
        if "нормальная" in skin_type or "normal" in skin_type:
            types.append("normal")
        if "проблемная" in skin_type or "problem" in skin_type:
            types.append("problem")
        
        return types if types else ["universal"]
    
    def clean_purpose(self, purpose: str) -> List[str]:
        """Clean and standardize product purpose"""
        if pd.isna(purpose) or purpose == "❌":
            return ["general"]
        
        purpose = str(purpose).lower()
        purposes = []
        
        if "очищение" in purpose or "cleansing" in purpose:
            purposes.append("cleansing")
        if "увлажнение" in purpose or "moisturizing" in purpose:
            purposes.append("moisturizing")
        if "питание" in purpose or "nourishing" in purpose:
            purposes.append("nourishing")
        if "против старения" in purpose or "anti-aging" in purpose:
            purposes.append("anti-aging")
        if "акне" in purpose or "acne" in purpose:
            purposes.append("acne")
        if "отбеливание" in purpose or "whitening" in purpose:
            purposes.append("brightening")
        if "защита" in purpose or "protection" in purpose:
            purposes.append("protection")
        
        return purposes if purposes else ["general"]
    
    def build_recommendation_models(self):
        """Build TF-IDF matrix for content-based recommendations"""
        try:
            # Create TF-IDF matrix from combined features
            self.content_matrix = self.tfidf_vectorizer.fit_transform(self.products_df['Combined_Features'])
            print("✅ Recommendation models built successfully")
        except Exception as e:
            print(f"❌ Error building recommendation models: {e}")
            raise
    
    def analyze_skin_condition(self, skin_analysis: Dict) -> Dict:
        """
        Analyze skin condition and return structured analysis
        
        Args:
            skin_analysis: Dictionary with skin analysis results from Derm Foundation
        
        Returns:
            Dictionary with structured skin analysis
        """
        analysis = {
            'skin_type': skin_analysis.get('skin_type', 'normal'),
            'concerns': skin_analysis.get('concerns', []),
            'severity': skin_analysis.get('severity', 'mild'),
            'age_group': skin_analysis.get('age_group', 'adult'),
            'specific_issues': skin_analysis.get('specific_issues', [])
        }
        
        # Map skin concerns to product needs
        if 'acne' in analysis['concerns']:
            analysis['product_needs'] = ['cleansing', 'acne', 'oil-control']
        elif 'dryness' in analysis['concerns']:
            analysis['product_needs'] = ['moisturizing', 'nourishing', 'hydrating']
        elif 'aging' in analysis['concerns']:
            analysis['product_needs'] = ['anti-aging', 'firming', 'moisturizing']
        else:
            analysis['product_needs'] = ['general', 'moisturizing']
        
        return analysis
    
    def get_recommendations(self, skin_analysis: Dict, top_k: int = 5) -> List[Dict]:
        """
        Get product recommendations based on skin analysis
        
        Args:
            skin_analysis: Skin analysis results
            top_k: Number of recommendations to return
        
        Returns:
            List of recommended products with scores
        """
        analysis = self.analyze_skin_condition(skin_analysis)
        
        # Filter products by skin type
        filtered_df = self.filter_by_skin_type(analysis['skin_type'])
        
        # Filter by product needs/concerns
        filtered_df = self.filter_by_concerns(filtered_df, analysis['product_needs'])
        
        # Calculate similarity scores
        recommendations = self.calculate_recommendation_scores(filtered_df, analysis)
        
        # Sort by score and return top k
        recommendations = sorted(recommendations, key=lambda x: x['score'], reverse=True)
        
        return recommendations[:top_k]
    
    def filter_by_skin_type(self, skin_type: str) -> pd.DataFrame:
        """Filter products suitable for specific skin type"""
        skin_type = skin_type.lower()
        
        mask = (
            self.products_df['Skin_Types_Clean'].apply(
                lambda x: skin_type in x or 'universal' in x or 'all' in str(x).lower()
            )
        )
        
        filtered_df = self.products_df[mask].copy()
        
        if len(filtered_df) == 0:
            # If no specific matches, return products for all skin types
            mask = self.products_df['Skin_Types_Clean'].apply(
                lambda x: 'universal' in x or 'all' in str(x).lower()
            )
            filtered_df = self.products_df[mask].copy()
        
        return filtered_df
    
    def filter_by_concerns(self, df: pd.DataFrame, concerns: List[str]) -> pd.DataFrame:
        """Filter products by skin concerns/needs"""
        if not concerns:
            return df
        
        mask = df['Product_Purpose_Clean'].apply(
            lambda x: any(concern in x for concern in concerns)
        )
        
        filtered_df = df[mask].copy()
        
        if len(filtered_df) == 0:
            return df  # Return original if no matches
        
        return filtered_df
    
    def calculate_recommendation_scores(self, filtered_df: pd.DataFrame, analysis: Dict) -> List[Dict]:
        """Calculate recommendation scores for filtered products"""
        recommendations = []
        
        for idx, row in filtered_df.iterrows():
            score = 0.0
            reasons = []
            
            # Skin type match bonus
            if analysis['skin_type'] in row['Skin_Types_Clean']:
                score += 0.3
                reasons.append(f"Perfect for {analysis['skin_type']} skin")
            
            # Concern/need match bonus
            purpose_match = any(need in row['Product_Purpose_Clean'] for need in analysis['product_needs'])
            if purpose_match:
                score += 0.4
                matching_purposes = [need for need in analysis['product_needs'] if need in row['Product_Purpose_Clean']]
                reasons.append(f"Addresses {', '.join(matching_purposes)}")
            
            # Price consideration (lower price gets slight bonus)
            if row['Price_Numeric'] > 0:
                price_score = max(0, (100 - row['Price_Numeric']) / 100 * 0.1)
                score += price_score
            
            # Ingredient quality bonus (if ingredients are specified)
            if row['Ingredients'] != "❌" and len(str(row['Ingredients'])) > 10:
                score += 0.2
                reasons.append("Contains quality ingredients")
            
            recommendations.append({
                'product_name': row['Product Name'],
                'url': row['URL'],
                'price': row['Price'],
                'description': row['Description'],
                'ingredients': row['Ingredients'],
                'skin_type': row['Тип кожи'],
                'purpose': row['По назначению'],
                'manufacturer': row['Производитель'],
                'country': row['Страна'],
                'volume': row['Объём (мл)'],
                'score': score,
                'reasons': reasons
            })
        
        return recommendations
    
    def get_product_by_category(self, category: str, skin_type: str = None, limit: int = 10) -> List[Dict]:
        """Get products by category (cleanser, moisturizer, etc.)"""
        category_mapping = {
            'cleanser': ['очищение', 'cleansing', 'гель для умывания'],
            'moisturizer': ['увлажнение', 'moisturizing', 'крем'],
            'serum': ['сыворотка', 'serum', 'эссенция'],
            'treatment': ['лечение', 'treatment', 'акне'],
            'sunscreen': ['защита', 'spf', 'солнцезащитный']
        }
        
        category_terms = category_mapping.get(category.lower(), [category])
        
        # Filter by category
        mask = self.products_df['Product_Purpose_Clean'].apply(
            lambda x: any(term in str(x).lower() for term in category_terms)
        ) | self.products_df['Тип средства'].apply(
            lambda x: any(term in str(x).lower() for term in category_terms)
        )
        
        filtered_df = self.products_df[mask].copy()
        
        # Further filter by skin type if specified
        if skin_type:
            skin_mask = filtered_df['Skin_Types_Clean'].apply(
                lambda x: skin_type.lower() in x or 'universal' in x
            )
            filtered_df = filtered_df[skin_mask]
        
        # Convert to list of dictionaries
        products = []
        for idx, row in filtered_df.head(limit).iterrows():
            products.append({
                'product_name': row['Product Name'],
                'url': row['URL'],
                'price': row['Price'],
                'description': row['Description'],
                'ingredients': row['Ingredients'],
                'skin_type': row['Тип кожи'],
                'purpose': row['По назначению'],
                'manufacturer': row['Производитель'],
                'volume': row['Объём (мл)']
            })
        
        return products
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics"""
        return {
            'total_products': len(self.products_df),
            'manufacturers': self.products_df['Производитель'].nunique(),
            'countries': self.products_df['Страна'].nunique(),
            'price_range': {
                'min': self.products_df['Price_Numeric'].min(),
                'max': self.products_df['Price_Numeric'].max(),
                'avg': self.products_df['Price_Numeric'].mean()
            },
            'skin_types_coverage': self.products_df['Тип кожи'].value_counts().to_dict()
        }

# Example usage and testing functions
def test_recommender(pkl_file_path: str):
    """Test the recommender system"""
    print("🧪 Testing Skin Product Recommender System...")
    
    try:
        # Initialize recommender
        recommender = SkinProductRecommender(pkl_file_path)
        
        # Print statistics
        stats = recommender.get_statistics()
        print(f"\n📊 Dataset Statistics:")
        print(f"Total Products: {stats['total_products']}")
        print(f"Manufacturers: {stats['manufacturers']}")
        print(f"Countries: {stats['countries']}")
        print(f"Price Range: ${stats['price_range']['min']:.2f} - ${stats['price_range']['max']:.2f}")
        
        # Test recommendation with sample skin analysis
        sample_analysis = {
            'skin_type': 'oily',
            'concerns': ['acne', 'oil-control'],
            'severity': 'moderate',
            'age_group': 'young_adult',
            'specific_issues': ['blackheads', 'enlarged_pores']
        }
        
        print(f"\n🔍 Testing recommendations for: {sample_analysis}")
        recommendations = recommender.get_recommendations(sample_analysis, top_k=3)
        
        print(f"\n💡 Top Recommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"\n{i}. {rec['product_name']}")
            print(f"   Price: {rec['price']}")
            print(f"   Score: {rec['score']:.2f}")
            print(f"   Reasons: {', '.join(rec['reasons'])}")
            print(f"   Purpose: {rec['purpose']}")
        
        # Test category filtering
        print(f"\n🧴 Testing category filtering (cleansers for oily skin):")
        cleansers = recommender.get_product_by_category('cleanser', 'oily', limit=3)
        for cleanser in cleansers:
            print(f"- {cleanser['product_name']} ({cleanser['price']})")
        
        print("\n✅ Recommender system test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise

if __name__ == "__main__":
    # Example usage
    pkl_file_path = "data.pkl"  # Replace with your actual pkl file path
    
    # Test the system
    test_recommender(pkl_file_path)
    
    # Example of how to use in your application
    recommender = SkinProductRecommender(pkl_file_path)
    
    # Sample skin analysis from Derm Foundation model
    skin_analysis = {
        'skin_type': 'combination',
        'concerns': ['dryness', 'sensitivity'],
        'severity': 'mild',
        'age_group': 'adult'
    }
    
    # Get recommendations
    recommendations = recommender.get_recommendations(skin_analysis, top_k=5)
    
    # Print results
    print("\n🎯 Final Recommendations:")
    for rec in recommendations:
        print(f"• {rec['product_name']} - {rec['price']}")
        print(f"  {rec['description'][:100]}...")
        print(f"  Reasons: {', '.join(rec['reasons'])}")
        print()