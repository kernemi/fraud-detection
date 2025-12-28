# functions for EDA plots, model evaluation visualizations, and insights

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

class EDAVisualizer:
  
    @staticmethod
    def plot_class_distribution(y: pd.Series, title: str = "Class Distribution") -> None:
        """Plot the distribution of target classes."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Count plot
        y.value_counts().plot(kind='bar', ax=ax1, color=['skyblue', 'lightcoral'])
        ax1.set_title(f'{title} - Counts')
        ax1.set_xlabel('Class')
        ax1.set_ylabel('Count')
        ax1.tick_params(axis='x', rotation=0)
        
        # Pie chart
        y.value_counts().plot(kind='pie', ax=ax2, autopct='%1.1f%%', startangle=90)
        ax2.set_title(f'{title} - Proportions')
        ax2.set_ylabel('')

        plt.show()
    
    @staticmethod
    def plot_numerical_distributions(df: pd.DataFrame, numerical_cols: List[str], 
                                   target_col: str = 'class') -> None:
        """Plot distributions of numerical features by class."""
        n_cols = min(3, len(numerical_cols))
        n_rows = (len(numerical_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = np.array(axes).reshape(-1)

        for i, col in enumerate(numerical_cols):
            for class_val in df[target_col].unique():
                subset = df[df[target_col] == class_val][col]
                axes[i].hist(subset, alpha=0.6, bins=30, label=f'Class {class_val}')

            axes[i].set_title(f'Distribution of {col}')
            axes[i].legend()

        for j in range(len(numerical_cols), len(axes)):
            axes[j].set_visible(False)

        plt.show()
    
    @staticmethod
    def plot_categorical_distributions(df: pd.DataFrame, categorical_cols: List[str], 
                                     target_col: str = 'class', max_unique: int = 20, top_n: int = 10) -> None:
        """Plot distributions of categorical features by class."""
        valid_cols = []
        for col in categorical_cols:
            if col not in df.columns:
                continue
            if df[col].nunique() > max_unique:
                print(f"Skipping '{col}' (too many categories: {df[col].nunique()})")
                continue
            valid_cols.append(col)

        if not valid_cols:
            print("No suitable categorical columns to plot.")
            return

        n_cols = min(2, len(valid_cols))
        n_rows = (len(valid_cols) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 6 * n_rows))
        axes = np.array(axes).reshape(-1)

        for i, col in enumerate(valid_cols):
            top_categories = df[col].value_counts().nlargest(top_n).index
            filtered_df = df[df[col].isin(top_categories)]

            ct = pd.crosstab(
                filtered_df[col],
                filtered_df[target_col],
                normalize='index'
            )

            ct.plot(kind='bar', stacked=True, ax=axes[i], legend=True)
            axes[i].set_title(f'{col} by Class (Top {top_n})')
            axes[i].set_ylabel('Proportion')
            axes[i].tick_params(axis='x', rotation=45)

        for j in range(len(valid_cols), len(axes)):
            axes[j].set_visible(False)

        plt.show()
    
    @staticmethod
    def plot_correlation_matrix(df: pd.DataFrame, figsize: tuple = (12, 10)) -> None:
        """Plot correlation matrix of numerical features."""
        corr = df.select_dtypes(include=[np.number]).corr()

        plt.figure(figsize=figsize)
        sns.heatmap(corr, annot=False, cmap='coolwarm', center=0)
        plt.title('Feature Correlation Matrix')
        plt.show()
    
    @staticmethod
    def plot_time_patterns(df: pd.DataFrame, time_col: str = 'hour_of_day', 
                          target_col: str = 'class') -> None:
        """Plot fraud patterns over time."""
        if time_col not in df.columns:
            print(f"Column {time_col} not found in dataframe")
            return
        
        summary = df.groupby(time_col)[target_col].agg(['mean', 'count'])

        fig, ax = plt.subplots(1, 2, figsize=(14, 5))

        summary['mean'].plot(kind='bar', ax=ax[0])
        ax[0].set_title(f'Fraud Rate by {time_col}')

        summary['count'].plot(kind='bar', ax=ax[1])
        ax[1].set_title(f'Volume by {time_col}')

        plt.show()


class ModelEvaluationVisualizer:
    """Class for model evaluation visualizations."""
    
    @staticmethod
    def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                            class_names: List[str] = ['Legitimate', 'Fraud']) -> None:
        """Plot confusion matrix."""
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()
    
    @staticmethod
    def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray) -> None:
        """Plot ROC curve."""
        from sklearn.metrics import roc_curve, auc
        
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.show()
    
    @staticmethod
    def plot_precision_recall_curve(y_true: np.ndarray, y_prob: np.ndarray) -> None:
        """Plot Precision-Recall curve."""
        from sklearn.metrics import precision_recall_curve, average_precision_score
        
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        avg_precision = average_precision_score(y_true, y_prob)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2,
                label=f'PR curve (AP = {avg_precision:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.show()
    
    @staticmethod
    def plot_feature_importance(feature_names: List[str], importance_scores: np.ndarray, 
                              top_n: int = 20) -> None:
        """Plot feature importance."""
        # Create dataframe and sort by importance
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_scores
        }).sort_values('importance', ascending=True).tail(top_n)
        
        plt.figure(figsize=(10, 8))
        plt.barh(importance_df['feature'], importance_df['importance'])
        plt.title(f'Top {top_n} Feature Importances')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        plt.show()

def create_comprehensive_eda_report(df: pd.DataFrame, target_col: str = 'class') -> None:
   
    print("COMPREHENSIVE EXPLORATORY DATA ANALYSIS REPORT")
    
    visualizer = EDAVisualizer()
    
    # Basic dataset info
    print(f"\nDataset Shape: {df.shape}")
    print(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Missing values
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        print(f"\nMissing Values:")
        print(missing_values[missing_values > 0])
    else:
        print("\nNo missing values found!")
    
    # Class distribution
    print(f"\nTarget Variable ({target_col}) Distribution:")
    print(df[target_col].value_counts())
    print(f"Fraud Rate: {df[target_col].mean():.4f}")
    
    # Plot class distribution
    visualizer.plot_class_distribution(df[target_col])
    
    # Numerical features analysis
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numerical_cols:
        numerical_cols.remove(target_col)
    
    if numerical_cols:
        visualizer.plot_numerical_distributions(df, numerical_cols[:6], target_col)
        visualizer.plot_correlation_matrix(df[numerical_cols + [target_col]])

    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if categorical_cols:
        visualizer.plot_categorical_distributions(df, categorical_cols[:4], target_col)

    if 'hour_of_day' in df.columns:
        visualizer.plot_time_patterns(df, 'hour_of_day', target_col)

    if 'day_of_week' in df.columns:
        visualizer.plot_time_patterns(df, 'day_of_week', target_col)

    print("EDA REPORT COMPLETED")

