"""
Knowledge base containing algorithm information and historical performance data.
This serves as the meta-learning foundation for algorithm recommendations.
"""

# Comprehensive algorithm knowledge base
ALGORITHM_KNOWLEDGE_BASE = {
    'Random Forest': {
        'description': 'Ensemble method using multiple decision trees with voting',
        'task_types': ['classification', 'regression'],
        'complexity': 'medium',
        'base_performance': 0.85,
        'strengths': [
            'Handles missing values well',
            'Feature importance ranking',
            'Resistant to overfitting',
            'Works with mixed data types'
        ],
        'use_cases': [
            'Mixed feature types',
            'Medium to large datasets',
            'When interpretability is needed',
            'Baseline model for comparison'
        ],
        'preferences': {
            'dataset_size': 'any',
            'n_features': 'any',
            'missing_data': 'high',
            'categorical_features': 'good',
            'class_imbalance': 'medium'
        }
    },
    
    'XGBoost': {
        'description': 'Gradient boosting framework optimized for speed and performance',
        'task_types': ['classification', 'regression'],
        'complexity': 'high',
        'base_performance': 0.88,
        'strengths': [
            'Excellent performance on structured data',
            'Built-in regularization',
            'Handles missing values',
            'Feature importance'
        ],
        'use_cases': [
            'Structured/tabular data',
            'Competitions and benchmarks',
            'When high accuracy is priority',
            'Large datasets'
        ],
        'preferences': {
            'dataset_size': 'medium',
            'n_features': 'medium',
            'missing_data': 'medium',
            'categorical_features': 'medium',
            'class_imbalance': 'good'
        }
    },
    
    'Logistic Regression': {
        'description': 'Linear model for classification with probabilistic output',
        'task_types': ['classification'],
        'complexity': 'low',
        'base_performance': 0.78,
        'strengths': [
            'Fast training and prediction',
            'Probabilistic output',
            'No hyperparameter tuning needed',
            'Interpretable coefficients'
        ],
        'use_cases': [
            'Linear relationships',
            'Small to medium datasets',
            'When speed is important',
            'Baseline classification model'
        ],
        'preferences': {
            'dataset_size': 'small',
            'n_features': 'low',
            'missing_data': 'low',
            'categorical_features': 'medium',
            'class_imbalance': 'medium'
        }
    },
    
    'Support Vector Machine': {
        'description': 'Finds optimal hyperplane for classification/regression',
        'task_types': ['classification', 'regression'],
        'complexity': 'high',
        'base_performance': 0.82,
        'strengths': [
            'Effective in high dimensions',
            'Memory efficient',
            'Kernel trick for non-linearity',
            'Works well with small datasets'
        ],
        'use_cases': [
            'High-dimensional data',
            'Small to medium datasets',
            'Text classification',
            'Non-linear patterns with kernels'
        ],
        'preferences': {
            'dataset_size': 'small',
            'n_features': 'high',
            'missing_data': 'low',
            'categorical_features': 'poor',
            'class_imbalance': 'medium'
        }
    },
    
    'Decision Tree': {
        'description': 'Tree-based model with simple if-else rules',
        'task_types': ['classification', 'regression'],
        'complexity': 'low',
        'base_performance': 0.75,
        'strengths': [
            'Highly interpretable',
            'No data preprocessing needed',
            'Handles non-linear relationships',
            'Fast prediction'
        ],
        'use_cases': [
            'When interpretability is crucial',
            'Mixed data types',
            'Rule extraction needed',
            'Small datasets'
        ],
        'preferences': {
            'dataset_size': 'small',
            'n_features': 'low',
            'missing_data': 'medium',
            'categorical_features': 'good',
            'class_imbalance': 'poor'
        }
    },
    
    'K-Nearest Neighbors': {
        'description': 'Instance-based learning using k nearest neighbors',
        'task_types': ['classification', 'regression'],
        'complexity': 'low',
        'base_performance': 0.76,
        'strengths': [
            'Simple and intuitive',
            'No assumptions about data',
            'Naturally handles multi-class',
            'Good for irregular decision boundaries'
        ],
        'use_cases': [
            'Small datasets',
            'Irregular decision boundaries',
            'Multi-class problems',
            'When local patterns matter'
        ],
        'preferences': {
            'dataset_size': 'small',
            'n_features': 'low',
            'missing_data': 'low',
            'categorical_features': 'poor',
            'class_imbalance': 'medium'
        }
    },
    
    'Naive Bayes': {
        'description': 'Probabilistic classifier based on Bayes theorem',
        'task_types': ['classification'],
        'complexity': 'low',
        'base_performance': 0.74,
        'strengths': [
            'Fast training and prediction',
            'Good with small datasets',
            'Handles multiple classes well',
            'Not sensitive to irrelevant features'
        ],
        'use_cases': [
            'Text classification',
            'Small datasets',
            'Multi-class problems',
            'When features are independent'
        ],
        'preferences': {
            'dataset_size': 'small',
            'n_features': 'high',
            'missing_data': 'medium',
            'categorical_features': 'good',
            'class_imbalance': 'medium'
        }
    },
    
    'Linear Regression': {
        'description': 'Linear model for regression tasks',
        'task_types': ['regression'],
        'complexity': 'low',
        'base_performance': 0.72,
        'strengths': [
            'Fast and simple',
            'Interpretable coefficients',
            'No hyperparameters',
            'Good baseline model'
        ],
        'use_cases': [
            'Linear relationships',
            'Small datasets',
            'When interpretability is key',
            'Baseline regression model'
        ],
        'preferences': {
            'dataset_size': 'any',
            'n_features': 'low',
            'missing_data': 'low',
            'categorical_features': 'medium',
            'class_imbalance': 'any'
        }
    },
    
    'Neural Network': {
        'description': 'Multi-layer perceptron with backpropagation',
        'task_types': ['classification', 'regression'],
        'complexity': 'high',
        'base_performance': 0.84,
        'strengths': [
            'Handles complex non-linear patterns',
            'Universal function approximator',
            'Good with large datasets',
            'Flexible architecture'
        ],
        'use_cases': [
            'Complex non-linear patterns',
            'Large datasets',
            'When high accuracy is needed',
            'Image and text data'
        ],
        'preferences': {
            'dataset_size': 'large',
            'n_features': 'medium',
            'missing_data': 'low',
            'categorical_features': 'medium',
            'class_imbalance': 'medium'
        }
    },
    
    'AdaBoost': {
        'description': 'Adaptive boosting ensemble method',
        'task_types': ['classification', 'regression'],
        'complexity': 'medium',
        'base_performance': 0.81,
        'strengths': [
            'Good generalization',
            'Reduces bias and variance',
            'Works well with weak learners',
            'Less prone to overfitting than single trees'
        ],
        'use_cases': [
            'Binary classification',
            'When weak learners available',
            'Medium-sized datasets',
            'Ensemble learning'
        ],
        'preferences': {
            'dataset_size': 'medium',
            'n_features': 'medium',
            'missing_data': 'low',
            'categorical_features': 'medium',
            'class_imbalance': 'medium'
        }
    },
    
    'Gradient Boosting': {
        'description': 'Sequential boosting method that builds models iteratively',
        'task_types': ['classification', 'regression'],
        'complexity': 'high',
        'base_performance': 0.86,
        'strengths': [
            'High predictive accuracy',
            'Handles mixed data types',
            'Built-in feature selection',
            'Robust to outliers'
        ],
        'use_cases': [
            'Insurance claim prediction',
            'Fraud detection',
            'Structured data problems',
            'When accuracy is priority'
        ],
        'preferences': {
            'dataset_size': 'medium',
            'n_features': 'medium',
            'missing_data': 'medium',
            'categorical_features': 'good',
            'class_imbalance': 'good'
        }
    },
    
    'LightGBM': {
        'description': 'Fast gradient boosting framework by Microsoft',
        'task_types': ['classification', 'regression'],
        'complexity': 'high',
        'base_performance': 0.87,
        'strengths': [
            'Very fast training',
            'Memory efficient',
            'High accuracy',
            'Handles categorical features'
        ],
        'use_cases': [
            'Credit scoring',
            'Energy demand forecasting',
            'Large datasets',
            'Time-sensitive applications'
        ],
        'preferences': {
            'dataset_size': 'large',
            'n_features': 'high',
            'missing_data': 'medium',
            'categorical_features': 'excellent',
            'class_imbalance': 'good'
        }
    },
    
    'K-Means': {
        'description': 'Centroid-based clustering algorithm',
        'task_types': ['clustering'],
        'complexity': 'low',
        'base_performance': 0.73,
        'strengths': [
            'Simple and fast',
            'Works well with spherical clusters',
            'Scales well to large datasets',
            'Easy to interpret'
        ],
        'use_cases': [
            'Customer segmentation',
            'Market research',
            'Data exploration',
            'Feature engineering'
        ],
        'preferences': {
            'dataset_size': 'any',
            'n_features': 'medium',
            'missing_data': 'low',
            'categorical_features': 'poor',
            'class_imbalance': 'any'
        }
    },
    
    'DBSCAN': {
        'description': 'Density-based clustering algorithm',
        'task_types': ['clustering'],
        'complexity': 'medium',
        'base_performance': 0.78,
        'strengths': [
            'Finds arbitrary shaped clusters',
            'Handles noise and outliers',
            'No need to specify cluster number',
            'Robust to initialization'
        ],
        'use_cases': [
            'Anomaly detection',
            'Event clustering',
            'Irregular cluster shapes',
            'Noisy data'
        ],
        'preferences': {
            'dataset_size': 'medium',
            'n_features': 'low',
            'missing_data': 'low',
            'categorical_features': 'poor',
            'class_imbalance': 'any'
        }
    },
    
    'Hierarchical Clustering': {
        'description': 'Tree-based clustering creating hierarchy of clusters',
        'task_types': ['clustering'],
        'complexity': 'medium',
        'base_performance': 0.75,
        'strengths': [
            'Creates cluster hierarchy',
            'No need to specify cluster number',
            'Deterministic results',
            'Good for small datasets'
        ],
        'use_cases': [
            'Gene sequence grouping',
            'Taxonomy creation',
            'Small dataset clustering',
            'Exploratory analysis'
        ],
        'preferences': {
            'dataset_size': 'small',
            'n_features': 'medium',
            'missing_data': 'low',
            'categorical_features': 'medium',
            'class_imbalance': 'any'
        }
    },
    
    'PCA': {
        'description': 'Principal Component Analysis for dimensionality reduction',
        'task_types': ['dimensionality_reduction'],
        'complexity': 'medium',
        'base_performance': 0.80,
        'strengths': [
            'Reduces dimensionality',
            'Removes correlation',
            'Data compression',
            'Noise reduction'
        ],
        'use_cases': [
            'Feature reduction',
            'Data compression',
            'Visualization',
            'Preprocessing step'
        ],
        'preferences': {
            'dataset_size': 'any',
            'n_features': 'high',
            'missing_data': 'low',
            'categorical_features': 'poor',
            'class_imbalance': 'any'
        }
    },
    
    't-SNE': {
        'description': 't-distributed Stochastic Neighbor Embedding for visualization',
        'task_types': ['dimensionality_reduction'],
        'complexity': 'high',
        'base_performance': 0.85,
        'strengths': [
            'Excellent for visualization',
            'Preserves local structure',
            'Reveals hidden patterns',
            'Good for non-linear data'
        ],
        'use_cases': [
            'High-dimensional data visualization',
            'Exploratory data analysis',
            'Pattern discovery',
            'Cluster visualization'
        ],
        'preferences': {
            'dataset_size': 'medium',
            'n_features': 'high',
            'missing_data': 'low',
            'categorical_features': 'poor',
            'class_imbalance': 'any'
        }
    },
    
    'Apriori': {
        'description': 'Association rule mining algorithm for frequent itemsets',
        'task_types': ['association_rule_mining'],
        'complexity': 'medium',
        'base_performance': 0.82,
        'strengths': [
            'Finds frequent patterns',
            'Generates association rules',
            'Interpretable results',
            'Good for market analysis'
        ],
        'use_cases': [
            'Market basket analysis',
            'Product recommendations',
            'Web usage patterns',
            'Frequent pattern discovery'
        ],
        'preferences': {
            'dataset_size': 'large',
            'n_features': 'medium',
            'missing_data': 'low',
            'categorical_features': 'excellent',
            'class_imbalance': 'any'
        }
    },
    
    'FP-Growth': {
        'description': 'Frequent Pattern Growth for association rule mining',
        'task_types': ['association_rule_mining'],
        'complexity': 'medium',
        'base_performance': 0.84,
        'strengths': [
            'Faster than Apriori',
            'Memory efficient',
            'No candidate generation',
            'Scalable to large datasets'
        ],
        'use_cases': [
            'Market basket analysis',
            'Product recommendations',
            'Large-scale pattern mining',
            'E-commerce analytics'
        ],
        'preferences': {
            'dataset_size': 'large',
            'n_features': 'medium',
            'missing_data': 'low',
            'categorical_features': 'excellent',
            'class_imbalance': 'any'
        }
    },
    
    'Q-Learning': {
        'description': 'Model-free reinforcement learning algorithm',
        'task_types': ['reinforcement_learning'],
        'complexity': 'high',
        'base_performance': 0.75,
        'strengths': [
            'Learns optimal policies',
            'Model-free approach',
            'Handles sequential decisions',
            'Good for discrete actions'
        ],
        'use_cases': [
            'Maze solving',
            'Grid games',
            'Route optimization',
            'Resource allocation'
        ],
        'preferences': {
            'dataset_size': 'any',
            'n_features': 'low',
            'missing_data': 'low',
            'categorical_features': 'good',
            'class_imbalance': 'any'
        }
    },
    
    'Deep Q Network': {
        'description': 'Deep reinforcement learning with neural networks',
        'task_types': ['reinforcement_learning'],
        'complexity': 'very_high',
        'base_performance': 0.88,
        'strengths': [
            'Handles complex state spaces',
            'Deep learning integration',
            'Excellent for games',
            'Scalable to high dimensions'
        ],
        'use_cases': [
            'Video game AI',
            'Self-driving simulation',
            'Complex decision making',
            'Multi-agent systems'
        ],
        'preferences': {
            'dataset_size': 'large',
            'n_features': 'high',
            'missing_data': 'low',
            'categorical_features': 'poor',
            'class_imbalance': 'any'
        }
    },
    
    'Policy Gradient': {
        'description': 'Direct policy optimization for reinforcement learning',
        'task_types': ['reinforcement_learning'],
        'complexity': 'very_high',
        'base_performance': 0.83,
        'strengths': [
            'Continuous action spaces',
            'Stochastic policies',
            'Good for control tasks',
            'Policy-based learning'
        ],
        'use_cases': [
            'Continuous control tasks',
            'Robotics control',
            'Portfolio optimization',
            'Dynamic pricing'
        ],
        'preferences': {
            'dataset_size': 'medium',
            'n_features': 'medium',
            'missing_data': 'low',
            'categorical_features': 'poor',
            'class_imbalance': 'any'
        }
    },
    
    'Actor-Critic': {
        'description': 'Advanced reinforcement learning combining value and policy methods',
        'task_types': ['reinforcement_learning'],
        'complexity': 'very_high',
        'base_performance': 0.90,
        'strengths': [
            'Combines best of both worlds',
            'Stable training',
            'Efficient learning',
            'Advanced optimization'
        ],
        'use_cases': [
            'Robotics applications',
            'Strategy optimization',
            'Complex control systems',
            'Multi-objective optimization'
        ],
        'preferences': {
            'dataset_size': 'large',
            'n_features': 'high',
            'missing_data': 'low',
            'categorical_features': 'poor',
            'class_imbalance': 'any'
        }
    },
    
    'Semi-Supervised Learning': {
        'description': 'Learning with both labeled and unlabeled data',
        'task_types': ['semi_supervised'],
        'complexity': 'high',
        'base_performance': 0.86,
        'strengths': [
            'Uses unlabeled data',
            'Reduces labeling cost',
            'Improves performance',
            'Good for limited labels'
        ],
        'use_cases': [
            'Image labeling with few labels',
            'Text classification',
            'Medical diagnosis',
            'Speech recognition'
        ],
        'preferences': {
            'dataset_size': 'large',
            'n_features': 'high',
            'missing_data': 'medium',
            'categorical_features': 'good',
            'class_imbalance': 'good'
        }
    },
    
    'Self-Supervised Learning': {
        'description': 'Learning representations from data itself without labels',
        'task_types': ['self_supervised'],
        'complexity': 'very_high',
        'base_performance': 0.87,
        'strengths': [
            'No labels required',
            'Learn rich representations',
            'Transfer learning ready',
            'Scalable to large data'
        ],
        'use_cases': [
            'Pre-training models',
            'Language modeling',
            'Image representation learning',
            'Feature extraction'
        ],
        'preferences': {
            'dataset_size': 'very_large',
            'n_features': 'very_high',
            'missing_data': 'low',
            'categorical_features': 'poor',
            'class_imbalance': 'any'
        }
    }
}

# Task-based algorithm recommendations
TASK_ALGORITHM_MAPPING = {
    'Spam Detection': {
        'description': 'Email/message spam classification',
        'task_type': 'classification',
        'recommended_algorithms': ['Naive Bayes', 'Support Vector Machine', 'Logistic Regression'],
        'data_characteristics': 'Text-heavy, high-dimensional, sparse features'
    },
    'Disease Diagnosis': {
        'description': 'Medical diagnosis based on symptoms/tests',
        'task_type': 'classification',
        'recommended_algorithms': ['Random Forest', 'XGBoost', 'Support Vector Machine'],
        'data_characteristics': 'Mixed features, interpretability important'
    },
    'House Price Prediction': {
        'description': 'Predict real estate prices',
        'task_type': 'regression',
        'recommended_algorithms': ['Random Forest', 'XGBoost', 'Linear Regression'],
        'data_characteristics': 'Mixed numerical/categorical, location features'
    },
    'Sales Forecasting': {
        'description': 'Predict future sales volume/revenue',
        'task_type': 'regression',
        'recommended_algorithms': ['Linear Regression', 'Random Forest', 'Gradient Boosting'],
        'data_characteristics': 'Time series, seasonal patterns'
    },
    'Customer Churn': {
        'description': 'Predict customer retention/departure',
        'task_type': 'classification',
        'recommended_algorithms': ['Random Forest', 'XGBoost', 'Logistic Regression'],
        'data_characteristics': 'Behavioral data, class imbalance'
    },
    'Fraud Detection': {
        'description': 'Identify fraudulent transactions/activities',
        'task_type': 'classification',
        'recommended_algorithms': ['XGBoost', 'Random Forest', 'Support Vector Machine'],
        'data_characteristics': 'Imbalanced classes, anomaly patterns'
    },
    'Face Recognition': {
        'description': 'Identify persons from facial images',
        'task_type': 'classification',
        'recommended_algorithms': ['Support Vector Machine', 'Neural Network', 'K-Nearest Neighbors'],
        'data_characteristics': 'High-dimensional image data'
    },
    'Sentiment Analysis': {
        'description': 'Analyze text sentiment (positive/negative)',
        'task_type': 'classification',
        'recommended_algorithms': ['Naive Bayes', 'Support Vector Machine', 'Logistic Regression'],
        'data_characteristics': 'Text data, bag-of-words features'
    },
    'Customer Segmentation': {
        'description': 'Group customers by behavior/characteristics',
        'task_type': 'clustering',
        'recommended_algorithms': ['K-Means', 'Hierarchical Clustering', 'DBSCAN'],
        'data_characteristics': 'Customer behavioral data'
    },
    'Stock Price Prediction': {
        'description': 'Predict stock market prices',
        'task_type': 'regression',
        'recommended_algorithms': ['Random Forest', 'XGBoost', 'Neural Network'],
        'data_characteristics': 'Time series, volatile patterns'
    },
    'Product Recommendation': {
        'description': 'Recommend products to users',
        'task_type': 'classification',
        'recommended_algorithms': ['K-Nearest Neighbors', 'Random Forest', 'Neural Network'],
        'data_characteristics': 'User-item interactions, sparse matrix'
    },
    'Weather Prediction': {
        'description': 'Predict weather conditions',
        'task_type': 'classification',
        'recommended_algorithms': ['Decision Tree', 'Random Forest', 'Gradient Boosting'],
        'data_characteristics': 'Meteorological sensors, temporal patterns'
    },
    'Loan Approval': {
        'description': 'Approve/reject loan applications',
        'task_type': 'classification',
        'recommended_algorithms': ['Decision Tree', 'Random Forest', 'Logistic Regression'],
        'data_characteristics': 'Financial data, interpretability needed'
    },
    'Email Filtering': {
        'description': 'Categorize emails by content/importance',
        'task_type': 'classification',
        'recommended_algorithms': ['Naive Bayes', 'Support Vector Machine', 'Random Forest'],
        'data_characteristics': 'Text content, multiple categories'
    },
    'Pattern Recognition': {
        'description': 'Identify patterns in data/images',
        'task_type': 'classification',
        'recommended_algorithms': ['K-Nearest Neighbors', 'Support Vector Machine', 'Neural Network'],
        'data_characteristics': 'Complex patterns, spatial/temporal features'
    },
    'Anomaly Detection': {
        'description': 'Detect unusual events/outliers',
        'task_type': 'clustering',
        'recommended_algorithms': ['DBSCAN', 'K-Means', 'Support Vector Machine'],
        'data_characteristics': 'Normal vs abnormal patterns'
    },
    'Credit Scoring': {
        'description': 'Assess creditworthiness of applicants',
        'task_type': 'classification',
        'recommended_algorithms': ['XGBoost', 'LightGBM', 'Logistic Regression'],
        'data_characteristics': 'Financial history, mixed features'
    },
    'Energy Demand Forecasting': {
        'description': 'Predict energy consumption patterns',
        'task_type': 'regression',
        'recommended_algorithms': ['LightGBM', 'Random Forest', 'Linear Regression'],
        'data_characteristics': 'Time series, seasonal/weather factors'
    },
    'Insurance Claim Prediction': {
        'description': 'Predict insurance claim amounts/probability',
        'task_type': 'regression',
        'recommended_algorithms': ['Gradient Boosting', 'Random Forest', 'XGBoost'],
        'data_characteristics': 'Policy details, historical claims'
    },
    'Text Classification': {
        'description': 'Categorize documents by content/topic',
        'task_type': 'classification',
        'recommended_algorithms': ['Naive Bayes', 'Support Vector Machine', 'Logistic Regression'],
        'data_characteristics': 'Text documents, bag-of-words/TF-IDF'
    },
    'Market Basket Analysis': {
        'description': 'Find associations between products purchased together',
        'task_type': 'association_rule_mining',
        'recommended_algorithms': ['Apriori', 'FP-Growth'],
        'data_characteristics': 'Transactional data, categorical items'
    },
    'Maze Solving': {
        'description': 'Navigate through environments to reach goals',
        'task_type': 'reinforcement_learning',
        'recommended_algorithms': ['Q-Learning', 'Deep Q Network'],
        'data_characteristics': 'Sequential states, discrete actions'
    },
    'Video Game AI': {
        'description': 'Create intelligent game-playing agents',
        'task_type': 'reinforcement_learning',
        'recommended_algorithms': ['Deep Q Network', 'Actor-Critic'],
        'data_characteristics': 'Complex state spaces, real-time decisions'
    },
    'Robotics Control': {
        'description': 'Control robotic systems and movements',
        'task_type': 'reinforcement_learning',
        'recommended_algorithms': ['Policy Gradient', 'Actor-Critic'],
        'data_characteristics': 'Continuous actions, sensor data'
    },
    'Image Labeling (Few Labels)': {
        'description': 'Classify images with limited labeled data',
        'task_type': 'semi_supervised',
        'recommended_algorithms': ['Semi-Supervised Learning'],
        'data_characteristics': 'Large unlabeled dataset, few labeled samples'
    },
    'Language Modeling': {
        'description': 'Learn language patterns without supervision',
        'task_type': 'self_supervised',
        'recommended_algorithms': ['Self-Supervised Learning'],
        'data_characteristics': 'Large text corpus, no labels needed'
    },
    'Data Visualization': {
        'description': 'Visualize high-dimensional data in 2D/3D',
        'task_type': 'dimensionality_reduction',
        'recommended_algorithms': ['t-SNE', 'PCA'],
        'data_characteristics': 'High-dimensional data, visualization needs'
    },
    'Gene Sequence Analysis': {
        'description': 'Analyze genetic sequences and relationships',
        'task_type': 'clustering',
        'recommended_algorithms': ['Hierarchical Clustering', 'K-Means'],
        'data_characteristics': 'Biological sequences, similarity patterns'
    }
}

# Historical dataset profiles for similarity matching
DATASET_PROFILES = {
    'iris': {
        'meta_features': {
            'n_samples': 150,
            'n_features': 4,
            'samples_to_features_ratio': 37.5,
            'numerical_ratio': 1.0,
            'categorical_ratio': 0.0,
            'missing_ratio': 0.0,
            'n_classes': 3,
            'class_entropy': 0.95,
            'mean_correlation': 0.35
        },
        'best_algorithms': ['SVM', 'Random Forest', 'KNN'],
        'task_type': 'classification'
    },
    
    'titanic': {
        'meta_features': {
            'n_samples': 891,
            'n_features': 8,
            'samples_to_features_ratio': 111.4,
            'numerical_ratio': 0.5,
            'categorical_ratio': 0.5,
            'missing_ratio': 0.15,
            'n_classes': 2,
            'class_entropy': 0.85,
            'mean_correlation': 0.25
        },
        'best_algorithms': ['Random Forest', 'XGBoost', 'Logistic Regression'],
        'task_type': 'classification'
    },
    
    'boston_housing': {
        'meta_features': {
            'n_samples': 506,
            'n_features': 13,
            'samples_to_features_ratio': 38.9,
            'numerical_ratio': 1.0,
            'categorical_ratio': 0.0,
            'missing_ratio': 0.0,
            'target_skewness': 1.1,
            'mean_correlation': 0.45
        },
        'best_algorithms': ['Random Forest', 'XGBoost', 'Linear Regression'],
        'task_type': 'regression'
    },
    
    'wine_quality': {
        'meta_features': {
            'n_samples': 1599,
            'n_features': 11,
            'samples_to_features_ratio': 145.4,
            'numerical_ratio': 1.0,
            'categorical_ratio': 0.0,
            'missing_ratio': 0.0,
            'n_classes': 6,
            'class_entropy': 0.75,
            'mean_correlation': 0.3
        },
        'best_algorithms': ['Random Forest', 'SVM', 'XGBoost'],
        'task_type': 'classification'
    },
    
    'diabetes': {
        'meta_features': {
            'n_samples': 768,
            'n_features': 8,
            'samples_to_features_ratio': 96.0,
            'numerical_ratio': 1.0,
            'categorical_ratio': 0.0,
            'missing_ratio': 0.0,
            'n_classes': 2,
            'class_entropy': 0.88,
            'mean_correlation': 0.2
        },
        'best_algorithms': ['Logistic Regression', 'Random Forest', 'SVM'],
        'task_type': 'classification'
    },
    
    'california_housing': {
        'meta_features': {
            'n_samples': 20640,
            'n_features': 8,
            'samples_to_features_ratio': 2580.0,
            'numerical_ratio': 1.0,
            'categorical_ratio': 0.0,
            'missing_ratio': 0.0,
            'target_skewness': 2.3,
            'mean_correlation': 0.6
        },
        'best_algorithms': ['XGBoost', 'Random Forest', 'Neural Network'],
        'task_type': 'regression'
    },
    
    'breast_cancer': {
        'meta_features': {
            'n_samples': 569,
            'n_features': 30,
            'samples_to_features_ratio': 19.0,
            'numerical_ratio': 1.0,
            'categorical_ratio': 0.0,
            'missing_ratio': 0.0,
            'n_classes': 2,
            'class_entropy': 0.92,
            'mean_correlation': 0.7
        },
        'best_algorithms': ['SVM', 'Logistic Regression', 'Random Forest'],
        'task_type': 'classification'
    },
    
    'mnist_small': {
        'meta_features': {
            'n_samples': 1000,
            'n_features': 784,
            'samples_to_features_ratio': 1.28,
            'numerical_ratio': 1.0,
            'categorical_ratio': 0.0,
            'missing_ratio': 0.0,
            'n_classes': 10,
            'class_entropy': 0.98,
            'mean_correlation': 0.1
        },
        'best_algorithms': ['Neural Network', 'SVM', 'Random Forest'],
        'task_type': 'classification'
    }
}

# Algorithm complexity mapping
COMPLEXITY_MAPPING = {
    'low': ['Logistic Regression', 'Linear Regression', 'Decision Tree', 'Naive Bayes', 'KNN'],
    'medium': ['Random Forest', 'AdaBoost'],
    'high': ['XGBoost', 'SVM', 'Neural Network']
}

# Task type mapping
TASK_TYPE_ALGORITHMS = {
    'classification': [
        'Random Forest', 'XGBoost', 'Logistic Regression', 'SVM', 
        'Decision Tree', 'KNN', 'Naive Bayes', 'Neural Network', 'AdaBoost'
    ],
    'regression': [
        'Random Forest', 'XGBoost', 'Linear Regression', 'SVM',
        'Decision Tree', 'KNN', 'Neural Network', 'AdaBoost'
    ]
}
