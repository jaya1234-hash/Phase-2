# Air Quality Analysis Project
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             accuracy_score, classification_report)

# --- Load and Clean Data ---
def load_and_clean_data(filepath):
    try:
        df = pd.read_csv(filepath)
        df = df.dropna()
        if 'rownames' in df.columns:
            df = df.drop(columns=['rownames'])
        return df
    except FileNotFoundError:
        print("File not found.")
        return None

# --- Categorize Ozone Levels ---
def categorize_ozone(value):
    if value <= 50:
        return 'Good'
    elif value <= 100:
        return 'Moderate'
    else:
        return 'Unhealthy'

# --- Linear Regression Analysis ---
def ozone_regression(df):
    X = df.drop('Ozone', axis=1)
    y = df['Ozone']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print("Regression Metrics:")
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("MSE:", mean_squared_error(y_test, y_pred))
    print("R² Score:", r2_score(y_test, y_pred))
    
    plt.figure(figsize=(6,4))
    plt.scatter(y_test, y_pred, color='blue')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
    plt.xlabel("Actual Ozone")
    plt.ylabel("Predicted Ozone")
    plt.title("Actual vs Predicted Ozone")
    plt.tight_layout()
    plt.show()

# --- Classification Based on Ozone Levels ---
def ozone_classification(df):
    df['AQI_Category'] = df['Ozone'].apply(categorize_ozone)
    
    X = df.drop(['Ozone', 'AQI_Category'], axis=1)
    y = df['AQI_Category']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print("Classification Metrics:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    
    # Feature importance
    feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    sns.barplot(x=feature_importances, y=feature_importances.index)
    plt.title("Feature Importance for AQI Classification")
    plt.xlabel("Importance Score")
    plt.ylabel("Features")
    plt.tight_layout()
    plt.show()

# --- Visualizations and EDA ---
def visualize_data(df):
    # Ozone Level Category Pie Chart
    df['Ozone_Category'] = df['Ozone'].apply(categorize_ozone)
    category_counts = df['Ozone_Category'].value_counts()
    
    plt.figure(figsize=(6,6))
    plt.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%',
            startangle=140, colors=['green', 'gold', 'red'])
    plt.title('Ozone Level Categories')
    plt.axis('equal')
    plt.tight_layout()
    plt.show()
    
    # Histogram of Ozone
    plt.figure(figsize=(8,6))
    sns.histplot(df['Ozone'], bins=30, kde=True, color='blue')
    plt.title('Distribution of Ozone Levels')
    plt.xlabel('Ozone')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

    # Boxplot of Ozone vs Month
    if 'Month' in df.columns:
        plt.figure(figsize=(8,6))
        sns.boxplot(x='Month', y='Ozone', data=df)
        plt.title('Ozone Levels Across Months')
        plt.xlabel('Month')
        plt.ylabel('Ozone')
        plt.tight_layout()
        plt.show()
    
    # Correlation Matrix
    plt.figure(figsize=(10,8))
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.show()
    
    # Pairplot
    selected_features = ['Ozone', 'Solar.R', 'Wind', 'Temp']
    sns.pairplot(df[selected_features].dropna(), diag_kind='kde')
    plt.suptitle('Pairplot of Selected Features', y=1.02)
    plt.tight_layout()
    plt.show()

# --- Main Execution ---
def main():
    filepath = 'air_quality_dataset-2.csv'  # Change if needed
    df = load_and_clean_data(filepath)
    
    if df is not None:
        visualize_data(df)
        ozone_regression(df)
        ozone_classification(df)

if __name__ == "__main__":
    main()
