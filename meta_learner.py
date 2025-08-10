import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from knowledge_base import ALGORITHM_KNOWLEDGE_BASE, DATASET_PROFILES
import warnings
warnings.filterwarnings('ignore')

class MetaLearner:
    """
    Meta-learning system that recommends algorithms based on dataset characteristics
    using similarity matching with historical performance data.
    """
    
    def __init__(self):
        self.knowledge_base = ALGORITHM_KNOWLEDGE_BASE
        self.dataset_profiles = DATASET_PROFILES
        
    def recommend_algorithms(self, meta_features, task_type, top_k=5):
        """
        Recommend top-k algorithms based on meta-features and task type.
        
        Args:
            meta_features (dict): Extracted meta-features from the dataset
            task_type (str): 'classification' or 'regression'
            top_k (int): Number of recommendations to return
            
        Returns:
            list: List of algorithm recommendations with confidence scores
        """
        # Filter algorithms by task type
        compatible_algorithms = {
            name: info for name, info in self.knowledge_base.items()
            if task_type in info['task_types']
        }
        
        if not compatible_algorithms:
            return []
        
        # Calculate similarities and scores
        recommendations = []
        
        for algo_name, algo_info in compatible_algorithms.items():
            # Calculate similarity score
            similarity_score = self._calculate_similarity(meta_features, algo_info)
            
            # Calculate base performance score
            performance_score = self._calculate_performance_score(meta_features, algo_info)
            
            # Combine scores
            confidence = (similarity_score * 0.6 + performance_score * 0.4)
            
            # Generate explanation
            explanation = self._generate_explanation(meta_features, algo_info)
            
            recommendation = {
                'algorithm': algo_name,
                'confidence': confidence,
                'expected_performance': performance_score,
                'similarity_score': similarity_score,
                'explanation': explanation,
                'description': algo_info['description'],
                'strengths': algo_info['strengths'],
                'use_cases': algo_info['use_cases'],
                'complexity': algo_info['complexity']
            }
            
            recommendations.append(recommendation)
        
        # Sort by confidence and return top-k
        recommendations.sort(key=lambda x: x['confidence'], reverse=True)
        return recommendations[:top_k]
    
    def _calculate_similarity(self, meta_features, algo_info):
        """Calculate similarity between dataset meta-features and algorithm preferences."""
        similarity_score = 0.0
        total_weight = 0.0
        
        preferences = algo_info['preferences']
        
        # Dataset size preference
        if 'dataset_size' in preferences:
            n_samples = meta_features.get('n_samples', 0)
            size_pref = preferences['dataset_size']
            
            if size_pref == 'small' and n_samples < 1000:
                similarity_score += 0.2
            elif size_pref == 'medium' and 1000 <= n_samples <= 10000:
                similarity_score += 0.2
            elif size_pref == 'large' and n_samples > 10000:
                similarity_score += 0.2
            elif size_pref == 'any':
                similarity_score += 0.1
            
            total_weight += 0.2
        
        # Feature count preference
        if 'n_features' in preferences:
            n_features = meta_features.get('n_features', 0)
            feature_pref = preferences['n_features']
            
            if feature_pref == 'low' and n_features < 10:
                similarity_score += 0.15
            elif feature_pref == 'medium' and 10 <= n_features <= 100:
                similarity_score += 0.15
            elif feature_pref == 'high' and n_features > 100:
                similarity_score += 0.15
            elif feature_pref == 'any':
                similarity_score += 0.1
            
            total_weight += 0.15
        
        # Missing data tolerance
        if 'missing_data' in preferences:
            missing_ratio = meta_features.get('missing_ratio', 0)
            missing_pref = preferences['missing_data']
            
            if missing_pref == 'low' and missing_ratio < 0.05:
                similarity_score += 0.1
            elif missing_pref == 'medium' and missing_ratio < 0.2:
                similarity_score += 0.1
            elif missing_pref == 'high':
                similarity_score += 0.1
            
            total_weight += 0.1
        
        # Categorical data handling
        if 'categorical_features' in preferences:
            categorical_ratio = meta_features.get('categorical_ratio', 0)
            cat_pref = preferences['categorical_features']
            
            if cat_pref == 'good' and categorical_ratio > 0:
                similarity_score += 0.1
            elif cat_pref == 'poor' and categorical_ratio == 0:
                similarity_score += 0.1
            
            total_weight += 0.1
        
        # Class imbalance handling
        if 'class_imbalance' in preferences:
            class_entropy = meta_features.get('class_entropy', 1.0)
            imbalance_pref = preferences['class_imbalance']
            
            # Lower entropy indicates imbalance
            if imbalance_pref == 'good' and class_entropy < 0.8:
                similarity_score += 0.1
            elif imbalance_pref == 'poor' and class_entropy > 0.8:
                similarity_score += 0.05
            
            total_weight += 0.1
        
        # Normalize by total weight
        if total_weight > 0:
            similarity_score = similarity_score / total_weight
        else:
            similarity_score = 0.5  # Default neutral score
        
        return min(1.0, max(0.0, similarity_score))
    
    def _calculate_performance_score(self, meta_features, algo_info):
        """Calculate expected performance based on historical data and meta-features."""
        base_performance = algo_info.get('base_performance', 0.75)
        
        # Adjust based on dataset characteristics
        performance_modifier = 0.0
        
        # Dataset size factor
        n_samples = meta_features.get('n_samples', 0)
        if n_samples < 100:
            performance_modifier -= 0.05  # Small datasets are harder
        elif n_samples > 10000:
            performance_modifier += 0.02  # Large datasets generally perform better
        
        # Feature dimensionality
        n_features = meta_features.get('n_features', 0)
        if n_features > n_samples and n_samples > 0:
            performance_modifier -= 0.1  # Curse of dimensionality
        
        # Missing data penalty
        missing_ratio = meta_features.get('missing_ratio', 0)
        performance_modifier -= missing_ratio * 0.1
        
        # Class imbalance penalty (for classification)
        class_entropy = meta_features.get('class_entropy', 1.0)
        if class_entropy < 0.5:
            performance_modifier -= 0.05
        
        # Algorithm-specific adjustments
        complexity = algo_info.get('complexity', 'medium')
        if complexity == 'high' and n_samples < 1000:
            performance_modifier -= 0.05  # Complex algorithms need more data
        elif complexity == 'low' and n_features > 100:
            performance_modifier -= 0.03  # Simple algorithms struggle with high dimensions
        
        final_performance = base_performance + performance_modifier
        return min(0.99, max(0.5, final_performance))
    
    def _generate_explanation(self, meta_features, algo_info):
        """Generate human-readable explanation for why the algorithm was recommended."""
        explanations = []
        
        # Dataset size explanation
        n_samples = meta_features.get('n_samples', 0)
        dataset_size_pref = algo_info['preferences'].get('dataset_size', 'any')
        
        if dataset_size_pref == 'small' and n_samples < 1000:
            explanations.append("works well with small datasets")
        elif dataset_size_pref == 'large' and n_samples > 10000:
            explanations.append("excels with large datasets")
        elif dataset_size_pref == 'any':
            explanations.append("adapts to various dataset sizes")
        
        # Feature handling
        categorical_ratio = meta_features.get('categorical_ratio', 0)
        if algo_info['preferences'].get('categorical_features') == 'good' and categorical_ratio > 0.3:
            explanations.append("handles categorical features effectively")
        
        # Missing data
        missing_ratio = meta_features.get('missing_ratio', 0)
        if algo_info['preferences'].get('missing_data') in ['medium', 'high'] and missing_ratio > 0.1:
            explanations.append("robust to missing data")
        
        # Class imbalance
        class_entropy = meta_features.get('class_entropy', 1.0)
        if algo_info['preferences'].get('class_imbalance') == 'good' and class_entropy < 0.8:
            explanations.append("performs well with imbalanced classes")
        
        # Default explanation
        if not explanations:
            explanations.append("suitable for your dataset characteristics")
        
        return ", ".join(explanations[:3])  # Limit to 3 explanations
    
    def get_similar_datasets(self, meta_features, top_k=3):
        """Find similar datasets from the knowledge base."""
        similarities = []
        
        for dataset_name, profile in self.dataset_profiles.items():
            # Calculate cosine similarity between meta-features
            current_features = np.array([meta_features.get(key, 0) for key in profile['meta_features'].keys()])
            profile_features = np.array(list(profile['meta_features'].values()))
            
            # Handle zero vectors
            if np.linalg.norm(current_features) == 0 or np.linalg.norm(profile_features) == 0:
                similarity = 0.0
            else:
                similarity = cosine_similarity([current_features], [profile_features])[0][0]
            
            similarities.append({
                'dataset': dataset_name,
                'similarity': similarity,
                'best_algorithms': profile['best_algorithms']
            })
        
        # Sort by similarity and return top-k
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:top_k]
