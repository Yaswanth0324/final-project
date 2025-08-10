import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple

def detect_task_type(target_series: pd.Series) -> str:
    """
    Automatically detect if the task is classification or regression.
    
    Args:
        target_series (pd.Series): The target variable
        
    Returns:
        str: 'classification' or 'regression'
    """
    # Remove missing values for analysis
    target_clean = target_series.dropna()
    
    if len(target_clean) == 0:
        return 'classification'  # Default fallback
    
    # Check if target is numeric
    if pd.api.types.is_numeric_dtype(target_clean):
        # For numeric targets, use heuristics to determine task type
        n_unique = target_clean.nunique()
        n_samples = len(target_clean)
        
        # If very few unique values relative to sample size, likely classification
        unique_ratio = n_unique / n_samples
        
        # Classification heuristics:
        # 1. Less than 20 unique values AND unique ratio < 0.05
        # 2. All values are integers AND unique ratio < 0.1
        # 3. Binary target (only 2 unique values)
        
        if n_unique <= 2:
            return 'classification'
        elif n_unique <= 20 and unique_ratio < 0.05:
            return 'classification'
        elif all(target_clean == target_clean.astype(int)) and unique_ratio < 0.1:
            return 'classification'
        else:
            return 'regression'
    
    else:
        # Non-numeric targets are always classification
        return 'classification'

def validate_dataset(dataset: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate the uploaded dataset for common issues.
    
    Args:
        dataset (pd.DataFrame): The dataset to validate
        
    Returns:
        dict: Validation result with 'valid' flag and 'message'
    """
    if dataset is None:
        return {'valid': False, 'message': 'Dataset is None'}
    
    if dataset.empty:
        return {'valid': False, 'message': 'Dataset is empty'}
    
    if len(dataset.columns) < 2:
        return {'valid': False, 'message': 'Dataset must have at least 2 columns (features + target)'}
    
    if len(dataset) < 5:
        return {'valid': False, 'message': 'Dataset must have at least 5 rows'}
    
    # Check for completely empty columns
    empty_columns = dataset.columns[dataset.isnull().all()].tolist()
    if empty_columns:
        return {'valid': False, 'message': f'Columns with all missing values: {empty_columns}'}
    
    # Check for excessive missing data (>90%)
    missing_ratios = dataset.isnull().sum() / len(dataset)
    high_missing_cols = missing_ratios[missing_ratios > 0.9].index.tolist()
    if high_missing_cols:
        return {'valid': False, 'message': f'Columns with >90% missing data: {high_missing_cols}'}
    
    # Check for duplicate column names
    if len(dataset.columns) != len(set(dataset.columns)):
        return {'valid': False, 'message': 'Dataset contains duplicate column names'}
    
    return {'valid': True, 'message': 'Dataset validation passed'}

def preprocess_features(dataset: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Basic preprocessing of features and target.
    
    Args:
        dataset (pd.DataFrame): Raw dataset
        target_column (str): Name of target column
        
    Returns:
        tuple: (processed_features, processed_target)
    """
    # Separate features and target
    X = dataset.drop(columns=[target_column]).copy()
    y = dataset[target_column].copy()
    
    # Handle missing values in features (simple imputation)
    for column in X.columns:
        if X[column].dtype in ['object', 'category']:
            # Categorical: fill with mode
            mode_value = X[column].mode()
            if len(mode_value) > 0:
                X[column] = X[column].fillna(mode_value.iloc[0])
            else:
                X[column] = X[column].fillna('Unknown')
        else:
            # Numerical: fill with median
            median_value = X[column].median()
            if pd.notna(median_value):
                X[column] = X[column].fillna(median_value)
            else:
                X[column] = X[column].fillna(0)
    
    # Handle missing values in target
    if y.dtype in ['object', 'category']:
        mode_value = y.mode()
        if len(mode_value) > 0:
            y = y.fillna(mode_value.iloc[0])
    else:
        median_value = y.median()
        if pd.notna(median_value):
            y = y.fillna(median_value)
    
    return X, y

def get_feature_importance_explanation(meta_features: Dict[str, Any]) -> str:
    """
    Generate explanation based on key meta-features.
    
    Args:
        meta_features (dict): Extracted meta-features
        
    Returns:
        str: Human-readable explanation
    """
    explanations = []
    
    # Dataset size
    n_samples = meta_features.get('n_samples', 0)
    if n_samples < 1000:
        explanations.append("small dataset size")
    elif n_samples > 10000:
        explanations.append("large dataset size")
    
    # Feature dimensionality
    n_features = meta_features.get('n_features', 0)
    if n_features > 50:
        explanations.append("high-dimensional feature space")
    elif n_features < 10:
        explanations.append("low-dimensional feature space")
    
    # Missing data
    missing_ratio = meta_features.get('missing_ratio', 0)
    if missing_ratio > 0.1:
        explanations.append("significant missing data")
    
    # Feature types
    categorical_ratio = meta_features.get('categorical_ratio', 0)
    if categorical_ratio > 0.5:
        explanations.append("predominantly categorical features")
    elif categorical_ratio == 0:
        explanations.append("only numerical features")
    
    # Class imbalance (for classification)
    if 'class_entropy' in meta_features:
        class_entropy = meta_features['class_entropy']
        if class_entropy < 0.7:
            explanations.append("imbalanced classes")
    
    if not explanations:
        return "standard dataset characteristics"
    
    return ", ".join(explanations[:3])

def format_algorithm_name(algorithm_name: str) -> str:
    """
    Format algorithm names for display.
    
    Args:
        algorithm_name (str): Raw algorithm name
        
    Returns:
        str: Formatted algorithm name
    """
    name_mapping = {
        'Random Forest': '🌲 Random Forest',
        'XGBoost': '🚀 XGBoost',
        'Logistic Regression': '📈 Logistic Regression',
        'Linear Regression': '📉 Linear Regression',
        'Support Vector Machine': '🎯 Support Vector Machine',
        'Decision Tree': '🌳 Decision Tree',
        'K-Nearest Neighbors': '👥 K-Nearest Neighbors',
        'Naive Bayes': '🎲 Naive Bayes',
        'Neural Network': '🧠 Neural Network',
        'AdaBoost': '⚡ AdaBoost'
    }
    
    return name_mapping.get(algorithm_name, algorithm_name)

def calculate_dataset_complexity_score(meta_features: Dict[str, Any]) -> float:
    """
    Calculate a complexity score for the dataset.
    
    Args:
        meta_features (dict): Extracted meta-features
        
    Returns:
        float: Complexity score between 0 and 1
    """
    complexity_score = 0.0
    
    # High dimensionality increases complexity
    dimensionality_ratio = meta_features.get('dimensionality_ratio', 0)
    complexity_score += min(dimensionality_ratio * 2, 0.3)
    
    # Missing data increases complexity
    missing_ratio = meta_features.get('missing_ratio', 0)
    complexity_score += missing_ratio * 0.2
    
    # High correlation increases complexity
    mean_correlation = meta_features.get('mean_correlation', 0)
    complexity_score += mean_correlation * 0.2
    
    # Class imbalance increases complexity (for classification)
    if 'class_entropy' in meta_features:
        class_entropy = meta_features['class_entropy']
        complexity_score += (1 - class_entropy) * 0.2
    
    # Feature skewness increases complexity
    mean_skewness = meta_features.get('mean_skewness', 0)
    complexity_score += min(mean_skewness / 5, 0.1)
    
    return min(complexity_score, 1.0)

def get_recommendation_confidence_level(confidence: float) -> str:
    """
    Convert confidence score to descriptive level.
    
    Args:
        confidence (float): Confidence score between 0 and 1
        
    Returns:
        str: Confidence level description
    """
    if confidence >= 0.9:
        return "Very High"
    elif confidence >= 0.8:
        return "High"
    elif confidence >= 0.7:
        return "Good"
    elif confidence >= 0.6:
        return "Medium"
    elif confidence >= 0.5:
        return "Fair"
    else:
        return "Low"
