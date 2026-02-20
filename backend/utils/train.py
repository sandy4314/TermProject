# backend/utils/train.py
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import RandomOverSampler
import joblib
from preprocessing import preprocess_text
import os
import warnings
import time

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

def load_and_preprocess_data(file_path, sample_size=None):
    print(f"Loading dataset from {file_path}...")
    start_time = time.time()
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file {file_path} not found.")
    try:
        df = pd.read_csv(file_path)
        if sample_size:
            df = df.sample(n=sample_size, random_state=42)
            print(f"Using sampled dataset with {df.shape[0]} rows.")
        print(f"Dataset loaded successfully. Shape: {df.shape}")
        print("Columns:", df.columns.tolist())
    except Exception as e:
        raise Exception(f"Error loading dataset: {e}")
    
    # Handle missing values
    df['review'] = df['review'].fillna('')
    df['condition'] = df['condition'].fillna('unknown')
    
    # Map ratings to sentiment classes
    def map_rating_to_sentiment(rating):
        if rating >= 8:
            return 2  # Positive
        elif rating >= 5:
            return 1  # Neutral
        else:
            return 0  # Negative
    
    print("Mapping ratings to sentiment classes...")
    df['sentiment'] = df['rating'].apply(map_rating_to_sentiment)
    
    # Group rare conditions (fewer than 2 instances) into 'other'
    print("Handling rare conditions...")
    condition_counts = df['condition'].value_counts()
    rare_conditions = condition_counts[condition_counts < 2].index
    df['condition'] = df['condition'].apply(lambda x: 'other' if x in rare_conditions else x)
    print(f"Grouped {len(rare_conditions)} rare conditions into 'other'. Unique conditions: {len(df['condition'].unique())}")
    
    # Preprocess reviews
    print("Preprocessing reviews...")
    df['processed_review'] = df['review'].apply(preprocess_text)
    print(f"Preprocessing completed in {time.time() - start_time:.2f} seconds.")
    return df

def train_models(sample_size=50000):
    # Load dataset
    data_path = '../data/drug_reviews.csv'
    try:
        df = load_and_preprocess_data(data_path, sample_size=sample_size)
    except Exception as e:
        print(f"Failed to load or preprocess data: {e}")
        return
    
    # Split data for sentiment
    print("Splitting data for sentiment...")
    X = df['processed_review']
    y_sentiment = df['sentiment']
    y_condition = df['condition']
    
    X_train, X_test, y_sentiment_train, y_sentiment_test = train_test_split(
        X, y_sentiment, test_size=0.3, random_state=42, stratify=y_sentiment
    )
    
    # Split data for condition
    print("Splitting data for condition...")
    X_train_cond, X_test_cond, y_condition_train, y_condition_test = train_test_split(
        X, y_condition, test_size=0.3, random_state=42, stratify=y_condition
    )
    
    # Handle class imbalance for sentiment
    print("Balancing sentiment classes...")
    ros = RandomOverSampler(random_state=42)
    X_train_resampled, y_sentiment_train_resampled = ros.fit_resample(X_train.values.reshape(-1, 1), y_sentiment_train)
    X_train_resampled = X_train_resampled.flatten()
    
    # Vectorize text for sentiment
    print("Vectorizing text for sentiment...")
    vectorizer_sentiment = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_vec = vectorizer_sentiment.fit_transform(X_train_resampled)
    X_test_vec = vectorizer_sentiment.transform(X_test)
    
    # Vectorize text for condition
    print("Vectorizing text for condition...")
    vectorizer_condition = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_cond_vec = vectorizer_condition.fit_transform(X_train_cond)
    X_test_cond_vec = vectorizer_condition.transform(X_test_cond)
    
    # Train sentiment models
    print("Training Naive Bayes model...")
    nb_model = MultinomialNB(alpha=0.1)
    nb_model.fit(X_train_vec, y_sentiment_train_resampled)
    
    print("Training SVM model...")
    svm_model = LinearSVC(max_iter=2000, class_weight='balanced', C=0.5, random_state=42)
    svm_model.fit(X_train_vec, y_sentiment_train_resampled)
    
    print("Training Neural Network model...")
    start_time = time.time()
    nn_model = MLPClassifier(
        hidden_layer_sizes=(100, 50),
        max_iter=200,
        random_state=42,
        verbose=True,
        early_stopping=True,
        n_iter_no_change=10
    )
    nn_model.fit(X_train_vec, y_sentiment_train_resampled)
    print(f"Neural Network training completed in {time.time() - start_time:.2f} seconds.")
    
    # Train condition model
    print("Training Condition model...")
    condition_model = LinearSVC(max_iter=2000, class_weight='balanced', C=0.5, random_state=42)
    condition_model.fit(X_train_cond_vec, y_condition_train)
    
    # Evaluate models
    print("Evaluating sentiment models...")
    print("Sentiment Classification (Naive Bayes):")
    print(classification_report(y_sentiment_test, nb_model.predict(X_test_vec), target_names=['Negative', 'Neutral', 'Positive']))
    
    print("Sentiment Classification (SVM):")
    print(classification_report(y_sentiment_test, svm_model.predict(X_test_vec), target_names=['Negative', 'Neutral', 'Positive']))
    
    print("Sentiment Classification (Neural Network):")
    print(classification_report(y_sentiment_test, nn_model.predict(X_test_vec), target_names=['Negative', 'Neutral', 'Positive']))
    
    
    print("Condition Classification (SVM):")
    print(classification_report(y_condition_test, condition_model.predict(X_test_cond_vec)))
    

    # Save models and vectorizers
    print("Saving models...")
    os.makedirs('models', exist_ok=True)
    joblib.dump(vectorizer_sentiment, os.path.join('models', 'tfidf_vectorizer_sentiment.joblib'))
    joblib.dump(vectorizer_condition, os.path.join('models', 'tfidf_vectorizer_condition.joblib'))
    joblib.dump(nb_model, os.path.join('models', 'sentiment_nb.joblib'))
    joblib.dump(svm_model, os.path.join('models', 'sentiment_svm.joblib'))
    joblib.dump(nn_model, os.path.join('models', 'sentiment_nn.joblib'))
    joblib.dump(condition_model, os.path.join('models', 'condition_svm.joblib'))
    print("Training completed successfully!")

if __name__ == "__main__":
    print("Starting training process...")
    train_models(sample_size=50000)

