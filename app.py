import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from meta_learner import MetaLearner
from feature_extractor import FeatureExtractor
from utils import detect_task_type, validate_dataset
from knowledge_base import TASK_ALGORITHM_MAPPING
import io

# Configure page
st.set_page_config(
    page_title="ML Algorithm Recommendation System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'dataset' not in st.session_state:
    st.session_state.dataset = None
if 'meta_features' not in st.session_state:
    st.session_state.meta_features = None
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = None
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Home"

# Handle navigation via query parameter
query_params = st.query_params
if 'page' in query_params:
    st.session_state.current_page = query_params['page'][0]

def main():
    # Navigation bar CSS
    st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .header-nav {margin-top: -42px;}
    .block-container {padding-top: 1.5rem;}
    .header-nav {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 0;
        margin: -1rem -1rem 0 -1rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    .nav-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 15px 40px;
        max-width: 1200px;
        margin: 0 auto;
    }
    .logo-section {
        display: flex;
        align-items: center;
        color: white;
        font-size: 28px;
        font-weight: bold;
        text-decoration: none;
    }
    .logo-icon {
        background: rgba(255,255,255,0.2);
        border-radius: 12px;
        padding: 8px;
        margin-right: 15px;
        font-size: 24px;
    }
    .nav-menu {
        display: flex;
        gap: 30px;
        align-items: center;
    }
    .nav-item {
        color: white;
        text-decoration: none;
        font-weight: 500;
        font-size: 16px;
        padding: 8px 16px;
        border-radius: 6px;
        transition: all 0.3s ease;
        cursor: pointer;
        border: none;
        background: transparent;
    }
    .nav-item:hover {
        background: rgba(255,255,255,0.15);
        transform: translateY(-1px);
    }
    .nav-item.active {
        background: rgba(255,255,255,0.2);
        border: 1px solid rgba(255,255,255,0.3);
    }
    .title-section {
        text-align: center;
        padding: 40px 20px;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        margin: 0 -1rem;
    }
    </style>
    """, unsafe_allow_html=True)

    home_active = "active" if st.session_state.current_page == "Home" else ""
    analysis_active = "active" if st.session_state.current_page == "Data Analysis" else ""

    st.markdown(f"""
    <div class="header-nav">
        <div class="nav-container">
            <div class="logo-section">
                <span class="logo-icon">🤖</span>
                <span>ISAAR</span>
            </div>
            <div class="nav-menu">
                <a href="/?page=Home" class="nav-item {home_active}">HOME</a>
                <a href="/?page=Data%20Analysis" class="nav-item {analysis_active}">DATA ANALYSIS</a>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="title-section">
        <h1 style="color: #2c3e50; margin-bottom: 15px; font-size: 2.8rem; font-weight: 700;">
            Intelligent System for Automated Algorithm Recommendation
        </h1>
        <p style="font-size: 20px; color: #5a6c7d; margin-bottom: 0; font-weight: 300;">
            AI-powered system that analyzes your dataset characteristics and recommends the best machine learning algorithms
        </p>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.current_page == "Home":
        display_home_page()
    elif st.session_state.current_page == "Data Analysis":
        display_data_analysis_page()

def display_home_page():
    st.markdown("<br>", unsafe_allow_html=True)
    st.header("🏠 Welcome to ISAAR")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## How it Works
        
        Our intelligent system uses **meta-learning** techniques to analyze your dataset characteristics and recommend the most suitable machine learning algorithms. Here's the process:
        
        ### 🔍 **Step 1: Dataset Analysis**
        - Upload your CSV or Excel file
        - Automatic detection of classification vs regression tasks
        - Extraction of statistical meta-features from your data
        
        ### 🧠 **Step 2: Meta-Learning Engine**
        - Compare your dataset with historical performance data
        - Calculate similarity scores with algorithm preferences
        - Apply advanced scoring algorithms for recommendations
        
        ### 🏆 **Step 3: Algorithm Recommendations**
        - Get top 5 algorithm recommendations ranked by confidence
        - Detailed explanations for each recommendation
        - Performance expectations and use case guidance
        
        ### 📈 **Step 4: Insights & Visualization**
        - Comprehensive dataset profiling and statistics
        - Interactive visualizations of your data characteristics
        - Meta-feature analysis and complexity scoring
        """)
        
        st.markdown("---")
        
        st.markdown("""
        ## Supported Features
        
        ✅ **Task Types**
        - Binary and multi-class classification
        - Linear and non-linear regression
        - Automatic task detection
        
        ✅ **Data Formats**
        - CSV files (comma-separated values)
        - Excel files (.xlsx, .xls)
        - Support for mixed data types
        
        ✅ **Algorithm Coverage**
        - Tree-based methods (Random Forest, XGBoost, Decision Trees)
        - Linear models (Logistic/Linear Regression)
        - Instance-based learning (K-Nearest Neighbors)
        - Ensemble methods (AdaBoost)
        - Support Vector Machines
        - Neural Networks
        - Naive Bayes
        
        ✅ **Analysis Features**
        - Statistical meta-feature extraction
        - Data quality assessment
        - Missing value analysis
        - Feature correlation analysis
        - Class balance evaluation
        - Outlier detection
        
        ✅ **Visualization & Insights**
        - Interactive dataset profiling
        - Feature distribution plots
        - Correlation heatmaps
        - Algorithm confidence scoring
        - Detailed recommendation explanations
        """)
    
    with col2:
        st.markdown("""
        ### 🚀 Quick Start
        
        Ready to get algorithm recommendations for your dataset?
        """)
        
        if st.button("📊 Start Data Analysis", type="primary", use_container_width=True):
            st.session_state.current_page = "Data Analysis"
            st.rerun()
        
        st.markdown("---")
        
        st.markdown("""
        ### 📋 Requirements
        
        **Dataset Requirements:**
        - Minimum 5 rows
        - At least 2 columns (features + target)
        - Maximum 200MB file size
        
        **Supported File Types:**
        - `.csv` files
        - `.xlsx` Excel files
        - `.xls` Excel files
        
        **Data Quality:**
        - Handle missing values (< 90% missing per column)
        - Mixed data types supported
        - No duplicate column names
        """)
        
        st.markdown("---")
        
        st.info("""
        💡 **Tip**: For best results, ensure your target variable is clearly defined and your dataset is reasonably clean. The system can handle some missing values and mixed data types.
        """)
        
        # ML Algorithms Overview Table
        st.markdown("---")
        st.subheader("📊 ML Methods and Algorithms Overview")
        
        # Create algorithm overview data
        algorithm_data = []
        
        # Supervised Learning
        supervised_algorithms = [
            ("Logistic Regression", "Classification (Binary/Multiclass)", "Spam detection, disease diagnosis"),
            ("Linear Regression", "Regression", "House prices, sales forecasting"),
            ("Decision Tree", "Classification, Regression", "Loan approval, weather prediction"),
            ("Random Forest", "Classification, Regression", "Customer churn, stock price prediction"),
            ("Support Vector Machine", "Classification, Regression", "Face recognition, sentiment analysis"),
            ("K-Nearest Neighbors", "Classification, Regression", "Product recommendation, pattern recognition"),
            ("Naive Bayes", "Classification (Text-heavy tasks)", "Email filtering, text classification"),
            ("Gradient Boosting", "Classification, Regression", "Insurance claim prediction, fraud detection"),
            ("XGBoost / LightGBM", "Classification, Regression", "Credit scoring, energy demand forecasting"),
            ("Neural Network", "Classification, Regression", "Complex patterns, image/text data"),
            ("AdaBoost", "Classification, Regression", "Binary classification, ensemble learning")
        ]
        
        # Unsupervised Learning
        unsupervised_algorithms = [
            ("K-Means", "Clustering", "Customer segmentation, grouping data"),
            ("DBSCAN", "Clustering (density-based)", "Anomaly detection, event clustering"),
            ("Hierarchical Clustering", "Clustering", "Gene sequence grouping, taxonomy"),
            ("PCA", "Dimensionality Reduction", "Feature reduction, data compression"),
            ("t-SNE", "Dimensionality Reduction (visualization)", "Visualizing high-dimensional data"),
            ("Apriori / FP-Growth", "Association Rule Mining", "Market basket analysis, product recommendations")
        ]
        
        # Reinforcement Learning
        reinforcement_algorithms = [
            ("Q-Learning", "Policy Learning (Sequential decision tasks)", "Maze solving, grid games"),
            ("Deep Q Network (DQN)", "Policy Learning with Deep Learning", "Video game AI, self-driving simulation"),
            ("Policy Gradient (REINFORCE)", "Policy Learning", "Continuous control tasks"),
            ("Actor-Critic (A3C, PPO)", "Policy and Value Optimization (Advanced)", "Robotics, strategy optimization")
        ]
        
        # Semi-Supervised and Self-Supervised Learning
        advanced_algorithms = [
            ("Semi-Supervised Learning", "Classification (on mixed labeled/unlabeled datasets)", "Image labeling with few labeled examples"),
            ("Self-Supervised Learning", "Feature learning from data itself", "Pre-training models, language modeling")
        ]
        
        # Combine all algorithms with ML method labels
        for algo, task_type, examples in supervised_algorithms:
            algorithm_data.append({"ML Method": "Supervised", "Algorithm": algo, "Types of Tasks Supported": task_type, "Examples of User Tasks": examples})
        
        for algo, task_type, examples in unsupervised_algorithms:
            algorithm_data.append({"ML Method": "Unsupervised", "Algorithm": algo, "Types of Tasks Supported": task_type, "Examples of User Tasks": examples})
        
        for algo, task_type, examples in reinforcement_algorithms:
            algorithm_data.append({"ML Method": "Reinforcement", "Algorithm": algo, "Types of Tasks Supported": task_type, "Examples of User Tasks": examples})
        
        for algo, task_type, examples in advanced_algorithms:
            if "Semi-Supervised" in algo:
                algorithm_data.append({"ML Method": "Semi-Supervised", "Algorithm": algo, "Types of Tasks Supported": task_type, "Examples of User Tasks": examples})
            else:
                algorithm_data.append({"ML Method": "Self-Supervised", "Algorithm": algo, "Types of Tasks Supported": task_type, "Examples of User Tasks": examples})
        
        # Create DataFrame and display table
        df_algorithms = pd.DataFrame(algorithm_data)
        
        st.markdown("""
        <style>
        .dataframe {
            border-collapse: collapse;
            margin: 25px 0;
            font-size: 0.9em;
            font-family: sans-serif;
            min-width: 400px;
            border-radius: 5px 5px 0 0;
            overflow: hidden;
        }
        .dataframe thead tr {
            background-color: #667eea;
            color: #ffffff;
            text-align: left;
        }
        .dataframe th,
        .dataframe td {
            padding: 12px 15px;
            border: 1px solid #dddddd;
        }
        .dataframe tbody tr {
            border-bottom: 1px solid #dddddd;
        }
        .dataframe tbody tr:nth-of-type(even) {
            background-color: #f3f3f3;
        }
        </style>
        """, unsafe_allow_html=True)
        
        st.dataframe(df_algorithms, use_container_width=True, hide_index=True)

def display_data_analysis_page():
    st.markdown("<br>", unsafe_allow_html=True)
    st.header("📊 Data Analysis & Algorithm Recommendations")
    
    # Task-based recommendation section
    st.sidebar.header("🎯 Task-Based Recommendations")
    task_options = ["Select a task..."] + list(TASK_ALGORITHM_MAPPING.keys())
    selected_task = st.sidebar.selectbox(
        "What type of problem are you solving?",
        options=task_options,
        help="Select your specific use case to get targeted algorithm recommendations"
    )
    
    if selected_task != "Select a task...":
        task_info = TASK_ALGORITHM_MAPPING[selected_task]
        st.sidebar.markdown(f"**Task:** {task_info['description']}")
        st.sidebar.markdown(f"**Type:** {task_info['task_type'].title()}")
        st.sidebar.markdown("**Recommended Algorithms:**")
        for i, algo in enumerate(task_info['recommended_algorithms'], 1):
            st.sidebar.markdown(f"{i}. {algo}")
        st.sidebar.markdown(f"**Data Characteristics:** {task_info['data_characteristics']}")
    
    st.sidebar.markdown("---")
    
    # Sidebar for file upload
    st.sidebar.header("📁 Dataset Upload")
    uploaded_file = st.sidebar.file_uploader(
        "Choose a dataset file",
        type=['csv', 'xlsx', 'xls'],
        help="Upload CSV or Excel files. Maximum size: 200MB"
    )
    
    if uploaded_file is not None:
        try:
            # Load dataset
            if uploaded_file.name.endswith('.csv'):
                dataset = pd.read_csv(uploaded_file)
            else:
                dataset = pd.read_excel(uploaded_file)
            
            # Validate dataset
            validation_result = validate_dataset(dataset)
            if not validation_result['valid']:
                st.error(f"Dataset validation failed: {validation_result['message']}")
                return
            
            st.session_state.dataset = dataset
            
            # Target selection
            st.sidebar.header("🎯 Target Variable")
            target_column = st.sidebar.selectbox(
                "Select target column:",
                options=dataset.columns.tolist(),
                help="Choose the column you want to predict"
            )
            
            if target_column:
                # Detect task type
                target_series = dataset[target_column]
                task_type = detect_task_type(target_series)
                st.sidebar.info(f"Detected task type: **{task_type.title()}**")
                
                # Extract meta-features
                try:
                    feature_extractor = FeatureExtractor()
                    meta_features = feature_extractor.extract_features(dataset, target_column, task_type)
                    st.session_state.meta_features = meta_features
                    
                    # Get recommendations
                    meta_learner = MetaLearner()
                    recommendations = meta_learner.recommend_algorithms(meta_features, task_type)
                    st.session_state.recommendations = recommendations
                except Exception as meta_error:
                    st.error(f"Error during analysis: {str(meta_error)}")
                    st.info("Please try with a different dataset or check the data format.")
                    return
                
                # Display results
                display_results(dataset, meta_features, recommendations, target_column, task_type)
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.info("Please ensure your file is properly formatted and try again.")
    
    else:
        # Instructions for data analysis page
        st.markdown("""
        ## 📤 Upload Your Dataset
        
        Use the sidebar to upload your CSV or Excel file to get started with algorithm recommendations.
        
        ### What happens next:
        1. **🎯 Select Target Variable**: Choose which column you want to predict
        2. **🔍 Automatic Analysis**: The system will extract meta-features from your dataset
        3. **🤖 Get Recommendations**: View top 5 algorithm recommendations with confidence scores
        4. **📊 Explore Insights**: Analyze dataset profiling, visualizations, and meta-features
        
        ### Ready to start?
        Choose your dataset file from the sidebar! 👈
        """)

def display_results(dataset, meta_features, recommendations, target_column, task_type):
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🏆 Recommendations", "📊 Dataset Profile", "🔍 Meta-Features", "📈 Visualizations"])
    
    with tab1:
        display_recommendations(recommendations, meta_features)
    
    with tab2:
        display_dataset_profile(dataset, target_column, task_type)
    
    with tab3:
        display_meta_features(meta_features)
    
    with tab4:
        display_visualizations(dataset, target_column, task_type)

def display_recommendations(recommendations, meta_features):
    st.header("🏆 Algorithm Recommendations")
    
    if not recommendations:
        st.warning("No recommendations available. Please check your dataset.")
        return
    
    # Display top recommendations
    for i, rec in enumerate(recommendations[:5], 1):
        with st.container():
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.subheader(f"{i}. {rec['algorithm']}")
                st.write(rec['description'])
            
            with col2:
                st.metric("Confidence", f"{rec['confidence']:.1%}")
            
            with col3:
                st.metric("Expected Score", f"{rec['expected_performance']:.3f}")
            
            # Explanation
            st.info(f"💡 **Why this algorithm?** {rec['explanation']}")
            
            # Algorithm characteristics
            with st.expander("View Algorithm Details"):
                col_a, col_b = st.columns(2)
                with col_a:
                    st.write("**Strengths:**")
                    for strength in rec['strengths']:
                        st.write(f"• {strength}")
                
                with col_b:
                    st.write("**Best for:**")
                    for use_case in rec['use_cases']:
                        st.write(f"• {use_case}")
            
            st.divider()

def display_dataset_profile(dataset, target_column, task_type):
    st.header("📊 Dataset Profile")
    
    # Basic statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Samples", f"{len(dataset):,}")
    
    with col2:
        st.metric("Features", len(dataset.columns) - 1)
    
    with col3:
        st.metric("Task Type", task_type.title())
    
    with col4:
        missing_percentage = (dataset.isnull().sum().sum() / (len(dataset) * len(dataset.columns))) * 100
        st.metric("Missing Data", f"{missing_percentage:.1f}%")
    
    # Data types breakdown
    st.subheader("📋 Feature Types")
    feature_types = dataset.dtypes.value_counts()
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        for dtype, count in feature_types.items():
            st.write(f"**{dtype}**: {count} features")
    
    with col2:
        # Convert to simple Python types for Plotly compatibility
        fig = px.pie(
            values=feature_types.values.tolist(), 
            names=[str(dtype) for dtype in feature_types.index], 
            title="Feature Type Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Missing values heatmap
    if dataset.isnull().sum().sum() > 0:
        st.subheader("🕳️ Missing Values Pattern")
        missing_data = dataset.isnull()
        
        if missing_data.sum().sum() > 0:
            fig = px.imshow(
                missing_data.T, 
                title="Missing Values Heatmap (Yellow = Missing)",
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Target distribution
    st.subheader(f"🎯 Target Variable Distribution: {target_column}")
    
    if task_type == 'classification':
        target_counts = dataset[target_column].value_counts()
        fig = px.bar(
            x=[str(x) for x in target_counts.index], 
            y=target_counts.values.tolist(), 
            title="Class Distribution"
        )
        fig.update_layout(xaxis_title="Classes", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)
        
        # Class balance check
        class_balance = target_counts.min() / target_counts.max()
        if class_balance < 0.5:
            st.warning(f"⚠️ Class imbalance detected. Ratio: {class_balance:.2f}")
        else:
            st.success("✅ Classes are reasonably balanced")
    
    else:  # regression
        fig = px.histogram(dataset, x=target_column, title="Target Distribution")
        st.plotly_chart(fig, use_container_width=True)

def display_meta_features(meta_features):
    st.header("🔍 Extracted Meta-Features")
    st.write("These features characterize your dataset and are used for algorithm recommendation:")
    
    # Organize meta-features by category
    categories = {
        "Dataset Size": ['n_samples', 'n_features', 'n_classes'],
        "Statistical": ['mean_skewness', 'mean_kurtosis', 'mean_std'],
        "Feature Types": ['categorical_ratio', 'numerical_ratio'],
        "Data Quality": ['missing_ratio', 'outlier_ratio'],
        "Complexity": ['class_entropy', 'feature_correlation']
    }
    
    for category, features in categories.items():
        st.subheader(category)
        cols = st.columns(min(len(features), 3))
        
        for i, feature in enumerate(features):
            if feature in meta_features:
                with cols[i % 3]:
                    value = meta_features[feature]
                    if isinstance(value, float):
                        st.metric(feature.replace('_', ' ').title(), f"{value:.4f}")
                    else:
                        st.metric(feature.replace('_', ' ').title(), str(value))

def display_visualizations(dataset, target_column, task_type):
    st.header("📈 Dataset Visualizations")
    
    # Correlation heatmap for numerical features
    numerical_cols = dataset.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numerical_cols) > 1:
        st.subheader("🔥 Feature Correlation Heatmap")
        corr_matrix = dataset[numerical_cols].corr()
        
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            title="Feature Correlation Matrix",
            color_continuous_scale='RdBu_r'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Feature distributions
    st.subheader("📊 Feature Distributions")
    
    # Select features to visualize
    features_to_plot = st.multiselect(
        "Select features to visualize:",
        options=[col for col in dataset.columns if col != target_column],
        default=[col for col in dataset.columns if col != target_column][:4]
    )
    
    if features_to_plot:
        n_cols = min(2, len(features_to_plot))
        n_rows = (len(features_to_plot) + n_cols - 1) // n_cols
        
        fig = make_subplots(
            rows=n_rows, 
            cols=n_cols,
            subplot_titles=features_to_plot
        )
        
        for i, feature in enumerate(features_to_plot):
            row = i // n_cols + 1
            col = i % n_cols + 1
            
            if dataset[feature].dtype in ['object', 'category']:
                # Categorical feature
                value_counts = dataset[feature].value_counts().head(10)
                fig.add_trace(
                    go.Bar(x=value_counts.index, y=value_counts.values, name=feature),
                    row=row, col=col
                )
            else:
                # Numerical feature
                fig.add_trace(
                    go.Histogram(x=dataset[feature], name=feature, nbinsx=30),
                    row=row, col=col
                )
        
        fig.update_layout(height=300 * n_rows, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Pairwise relationships (for small datasets)
    if len(numerical_cols) <= 6 and len(dataset) <= 10000:
        st.subheader("🔍 Feature Relationships")
        selected_features = st.multiselect(
            "Select features for pairplot:",
            options=numerical_cols,
            default=numerical_cols[:3]
        )
        
        if len(selected_features) >= 2:
            # Create pairwise scatter plots
            n_features = len(selected_features)
            fig = make_subplots(
                rows=n_features, 
                cols=n_features,
                subplot_titles=[f"{f1} vs {f2}" for f1 in selected_features for f2 in selected_features]
            )
            
            for i, feat1 in enumerate(selected_features):
                for j, feat2 in enumerate(selected_features):
                    if i != j:
                        fig.add_trace(
                            go.Scatter(
                                x=dataset[feat2], 
                                y=dataset[feat1],
                                mode='markers',
                                opacity=0.6,
                                name=f"{feat1} vs {feat2}"
                            ),
                            row=i+1, col=j+1
                        )
                    else:
                        # Diagonal - show distribution
                        fig.add_trace(
                            go.Histogram(x=dataset[feat1], name=feat1),
                            row=i+1, col=j+1
                        )
            
            fig.update_layout(height=200 * n_features, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
