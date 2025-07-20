import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import uuid
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 1. Synthetic Dataset Generation
def generate_synthetic_cidds_data(n_samples=10000):
    np.random.seed(42)
    data = {
        'Src_IP': [f"192.168.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}" for _ in range(n_samples)],
        'Src_Port': np.random.randint(1024, 65535, n_samples),
        'Dest_IP': [f"192.168.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}" for _ in range(n_samples)],
        'Dest_Port': np.random.randint(80, 8080, n_samples),
        'Proto': np.random.choice(['TCP', 'UDP', 'ICMP'], n_samples),
        'Duration': np.random.exponential(100, n_samples),
        'Bytes': np.random.exponential(1000, n_samples),
        'Packets': np.random.poisson(10, n_samples),
        'Class': np.random.choice(['Normal', 'Attacker', 'Victim', 'Suspicious', 'Unknown'], n_samples, p=[0.8, 0.05, 0.05, 0.05, 0.05])
    }
    df = pd.DataFrame(data)
    
    # Introduce anomalies for non-normal classes
    anomaly_mask = df['Class'] != 'Normal'
    df.loc[anomaly_mask, 'Bytes'] *= np.random.uniform(2, 10, anomaly_mask.sum())
    df.loc[anomaly_mask, 'Packets'] *= np.random.uniform(2, 5, anomaly_mask.sum())
    df.loc[anomaly_mask, 'Duration'] *= np.random.uniform(1.5, 3, anomaly_mask.sum())
    
    return df

# 2. Data Preprocessing
def preprocess_data(df):
    # Encode categorical features
    label_encoders = {}
    for column in ['Src_IP', 'Dest_IP', 'Proto']:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le
    
    # Features for model
    features = ['Src_IP', 'Src_Port', 'Dest_IP', 'Dest_Port', 'Proto', 'Duration', 'Bytes', 'Packets']
    X = df[features]
    
    # Normalize numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split normal and anomalous data
    normal_data = X_scaled[df['Class'] == 'Normal']
    anomalous_data = X_scaled[df['Class'] != 'Normal']
    y_anomalous = (df['Class'] != 'Normal').astype(int)
    
    return normal_data, anomalous_data, X_scaled, y_anomalous, scaler, label_encoders

# 3. Build Multi-Layer Deep Autoencoder (M-LDAE)
def build_mldae(input_dim):
    model = Sequential([
        Dense(64, activation='relu', input_dim=input_dim),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),  # Bottleneck layer
        Dense(32, activation='relu'),
        Dense(64, activation='relu'),
        Dense(input_dim, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# 4. Train Autoencoder
def train_autoencoder(model, normal_data, epochs=50, batch_size=32):
    history = model.fit(
        normal_data, normal_data,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=1
    )
    return history

# 5. Anomaly Detection
def detect_anomalies(model, data, threshold):
    reconstructions = model.predict(data)
    mse = np.mean(np.power(data - reconstructions, 2), axis=1)
    return mse > threshold

# 6. Evaluate Model
def evaluate_model(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)
    
    logging.info(f"Accuracy: {accuracy:.4f}")
    logging.info(f"Precision: {precision:.4f}")
    logging.info(f"Recall: {recall:.4f}")
    logging.info(f"F1-Score: {f1:.4f}")
    logging.info(f"AUC-ROC: {auc:.4f}")
    
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, 'auc': auc}

# Main Execution
def main():
    # Generate synthetic dataset
    logging.info("Generating synthetic dataset...")
    df = generate_synthetic_cidds_data()
    
    # Preprocess data
    logging.info("Preprocessing data...")
    normal_data, anomalous_data, X_scaled, y_anomalous, scaler, label_encoders = preprocess_data(df)
    
    # Build and train autoencoder
    logging.info("Building and training M-LDAE...")
    model = build_mldae(input_dim=X_scaled.shape[1])
    history = train_autoencoder(model, normal_data)
    
    # Determine threshold using normal data
    reconstructions = model.predict(normal_data)
    mse = np.mean(np.power(normal_data - reconstructions, 2), axis=1)
    threshold = np.percentile(mse, 95)  # 95th percentile as threshold
    
    # Detect anomalies
    logging.info("Detecting anomalies...")
    y_pred = detect_anomalies(model, X_scaled, threshold)
    
    # Evaluate model
    logging.info("Evaluating model...")
    metrics = evaluate_model(y_anomalous, y_pred)
    
    # Save synthetic dataset
    df.to_csv('synthetic_cidds_data.csv', index=False)
    logging.info("Synthetic dataset saved as 'synthetic_cidds_data.csv'")
    
    return metrics, model, scaler, label_encoders

if __name__ == "__main__":
    metrics, model, scaler, label_encoders = main()