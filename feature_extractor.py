import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class FeatureExtractor:
    """
    Extracts meta-features from datasets for algorithm recommendation.
    """
    
    def __init__(self):
        self.label_encoders = {}
    
    def extract_features(self, dataset, target_column, task_type):
        """
        Extract comprehensive meta-features from the dataset.
        
        Args:
            dataset (pd.DataFrame): The input dataset
            target_column (str): Name of the target column
            task_type (str): 'classification' or 'regression'
            
        Returns:
            dict: Dictionary containing extracted meta-features
        """
        # Separate features and target
        X = dataset.drop(columns=[target_column])
        y = dataset[target_column]
        
        meta_features = {}
        
        # Basic dataset statistics
        meta_features.update(self._extract_basic_stats(X, y))
        
        # Feature type statistics
        meta_features.update(self._extract_feature_type_stats(X))
        
        # Statistical properties
        meta_features.update(self._extract_statistical_properties(X))
        
        # Data quality metrics
        meta_features.update(self._extract_data_quality_metrics(X, y))
        
        # Task-specific features
        if task_type == 'classification':
            meta_features.update(self._extract_classification_features(y))
        else:
            meta_features.update(self._extract_regression_features(y))
        
        # Complexity metrics
        meta_features.update(self._extract_complexity_metrics(X, y))
        
        return meta_features
    
    def _extract_basic_stats(self, X, y):
        """Extract basic dataset statistics."""
        return {
            'n_samples': len(X),
            'n_features': len(X.columns),
            'samples_to_features_ratio': len(X) / len(X.columns) if len(X.columns) > 0 else 0,
            'dataset_ratio': len(X) / (len(X.columns) * len(X)) if len(X.columns) > 0 else 0
        }
    
    def _extract_feature_type_stats(self, X):
        """Extract statistics about feature types."""
        numerical_features = X.select_dtypes(include=[np.number]).columns
        categorical_features = X.select_dtypes(include=['object', 'category']).columns
        
        total_features = len(X.columns)
        
        return {
            'numerical_features': len(numerical_features),
            'categorical_features': len(categorical_features),
            'numerical_ratio': len(numerical_features) / total_features if total_features > 0 else 0,
            'categorical_ratio': len(categorical_features) / total_features if total_features > 0 else 0
        }
    
    def _extract_statistical_properties(self, X):
        """Extract statistical properties of numerical features."""
        numerical_features = X.select_dtypes(include=[np.number])
        
        if numerical_features.empty:
            return {
                'mean_skewness': 0.0,
                'mean_kurtosis': 0.0,
                'mean_std': 0.0,
                'mean_correlation': 0.0
            }
        
        try:
            # Calculate skewness and kurtosis
            skewness_values = numerical_features.skew()
            kurtosis_values = numerical_features.kurtosis()
            
            # Calculate standard deviations (normalized)
            std_values = numerical_features.std()
            
            # Calculate mean correlation
            corr_matrix = numerical_features.corr()
            # Get upper triangle (excluding diagonal)
            upper_triangle = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            mean_correlation = upper_triangle.stack().mean()
            
            return {
                'mean_skewness': abs(skewness_values.mean()) if not skewness_values.empty else 0.0,
                'mean_kurtosis': abs(kurtosis_values.mean()) if not kurtosis_values.empty else 0.0,
                'mean_std': std_values.mean() if not std_values.empty else 0.0,
                'mean_correlation': abs(mean_correlation) if not pd.isna(mean_correlation) else 0.0
            }
        except:
            return {
                'mean_skewness': 0.0,
                'mean_kurtosis': 0.0,
                'mean_std': 0.0,
                'mean_correlation': 0.0
            }
    
    def _extract_data_quality_metrics(self, X, y):
        """Extract data quality metrics."""
        total_values = X.size + len(y)
        missing_values = X.isnull().sum().sum() + y.isnull().sum()
        
        # Calculate outlier ratio for numerical features
        numerical_features = X.select_dtypes(include=[np.number])
        outlier_count = 0
        
        if not numerical_features.empty:
            for column in numerical_features.columns:
                Q1 = numerical_features[column].quantile(0.25)
                Q3 = numerical_features[column].quantile(0.75)
                IQR = Q3 - Q1
                outlier_count += ((numerical_features[column] < (Q1 - 1.5 * IQR)) | 
                                (numerical_features[column] > (Q3 + 1.5 * IQR))).sum()
        
        return {
            'missing_ratio': missing_values / total_values if total_values > 0 else 0,
            'outlier_ratio': outlier_count / len(X) if len(X) > 0 else 0,
            'duplicate_ratio': X.duplicated().sum() / len(X) if len(X) > 0 else 0
        }
    
    def _extract_classification_features(self, y):
        """Extract classification-specific features."""
        try:
            # Handle missing values
            y_clean = y.dropna()
            
            if len(y_clean) == 0:
                return {
                    'n_classes': 0,
                    'class_entropy': 0.0,
                    'class_imbalance_ratio': 1.0,
                    'minority_class_ratio': 0.0
                }
            
            # Get class distribution
            class_counts = y_clean.value_counts()
            n_classes = len(class_counts)
            
            # Calculate class entropy
            class_probs = class_counts / len(y_clean)
            class_entropy = -np.sum(class_probs * np.log2(class_probs + 1e-10))
            
            # Normalize entropy by maximum possible entropy
            max_entropy = np.log2(n_classes) if n_classes > 1 else 1
            normalized_entropy = class_entropy / max_entropy if max_entropy > 0 else 0
            
            # Class imbalance metrics
            majority_class_ratio = class_counts.max() / len(y_clean)
            minority_class_ratio = class_counts.min() / len(y_clean)
            imbalance_ratio = minority_class_ratio / majority_class_ratio if majority_class_ratio > 0 else 1.0
            
            return {
                'n_classes': n_classes,
                'class_entropy': normalized_entropy,
                'class_imbalance_ratio': imbalance_ratio,
                'minority_class_ratio': minority_class_ratio
            }
        except:
            return {
                'n_classes': 2,
                'class_entropy': 1.0,
                'class_imbalance_ratio': 1.0,
                'minority_class_ratio': 0.5
            }
    
    def _extract_regression_features(self, y):
        """Extract regression-specific features."""
        try:
            # Handle missing values
            y_clean = y.dropna()
            
            if len(y_clean) == 0:
                return {
                    'target_skewness': 0.0,
                    'target_kurtosis': 0.0,
                    'target_range': 0.0,
                    'target_std': 0.0
                }
            
            # Convert to numeric if needed
            y_numeric = pd.to_numeric(y_clean, errors='coerce')
            y_numeric = y_numeric.dropna()
            
            if len(y_numeric) == 0:
                return {
                    'target_skewness': 0.0,
                    'target_kurtosis': 0.0,
                    'target_range': 0.0,
                    'target_std': 0.0
                }
            
            # Calculate target statistics
            target_skewness = abs(float(stats.skew(y_numeric)))
            target_kurtosis = abs(float(stats.kurtosis(y_numeric)))
            target_range = float(y_numeric.max() - y_numeric.min())
            target_std = float(y_numeric.std())
            
            # Normalize range by mean to get coefficient of variation
            target_mean = float(y_numeric.mean())
            normalized_range = target_range / abs(target_mean) if target_mean != 0 else target_range
            
            return {
                'target_skewness': target_skewness,
                'target_kurtosis': target_kurtosis,
                'target_range': normalized_range,
                'target_std': target_std / abs(target_mean) if target_mean != 0 else target_std
            }
        except:
            return {
                'target_skewness': 0.0,
                'target_kurtosis': 0.0,
                'target_range': 1.0,
                'target_std': 1.0
            }
    
    def _extract_complexity_metrics(self, X, y):
        """Extract dataset complexity metrics."""
        try:
            # Feature correlation complexity
            numerical_features = X.select_dtypes(include=[np.number])
            
            if numerical_features.empty:
                feature_correlation = 0.0
            else:
                corr_matrix = numerical_features.corr()
                # Average absolute correlation (excluding diagonal)
                mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
                feature_correlation = np.abs(corr_matrix.where(mask)).stack().mean()
                feature_correlation = feature_correlation if not pd.isna(feature_correlation) else 0.0
            
            # Dimensionality ratio
            dimensionality_ratio = len(X.columns) / len(X) if len(X) > 0 else 0
            
            # Sparsity (for datasets with many zeros)
            sparsity_ratio = 0.0
            if not numerical_features.empty:
                total_numeric_values = numerical_features.size
                zero_values = (numerical_features == 0).sum().sum()
                sparsity_ratio = zero_values / total_numeric_values if total_numeric_values > 0 else 0
            
            return {
                'feature_correlation': feature_correlation,
                'dimensionality_ratio': dimensionality_ratio,
                'sparsity_ratio': sparsity_ratio
            }
        except:
            return {
                'feature_correlation': 0.0,
                'dimensionality_ratio': 0.1,
                'sparsity_ratio': 0.0
            }
