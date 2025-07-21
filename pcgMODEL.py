import os
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.combine import SMOTETomek
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import logging
import warnings
from scipy.signal import butter, lfilter, hilbert, find_peaks

warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Data paths
ROOT_PATH = r"J:\PCG\the-circor-digiscope-phonocardiogram-dataset-1.0.3"
DATA_PATH = os.path.join(ROOT_PATH, "training_data")
CSV_PATH = os.path.join(ROOT_PATH, "training_data.csv")
SAVE_DIR = ROOT_PATH  # Save plots and models in root directory

# Validate paths
if not os.path.exists(CSV_PATH):
    logging.error(f"CSV file not found at: {CSV_PATH}")
    raise FileNotFoundError(f"CSV file not found at: {CSV_PATH}. Please verify the file exists in the root directory.")
if not os.path.exists(DATA_PATH):
    logging.error(f"Training data directory not found at: {DATA_PATH}")
    raise FileNotFoundError(f"Training data directory not found at: {DATA_PATH}. Please verify the directory exists.")

# Model parameters
SAMPLE_RATE = 4000  # Hz, as per dataset
DURATION = 10  # seconds for each clip
N_MELS = 128  # Increased for better resolution
N_MFCC = 40  # Increased for more features
N_CHROMA = 12
N_CONTRAST = 5  # Reduced to avoid Nyquist error
MAX_TIME = 128
BATCH_SIZE = 8  # Reduced for finer gradients
EPOCHS = 300  # Increased for convergence
N_CLASSES = 2
CLASS_NAMES = ['Absent', 'Present']


# Cyclical learning rate schedule
class CyclicalLearningRate(tf.keras.callbacks.Callback):
    def __init__(self, base_lr, max_lr, step_size, total_steps):
        super().__init__()
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.total_steps = total_steps
        self.step = 0

    def on_batch_begin(self, batch, logs=None):
        self.step += 1
        cycle = np.floor(1 + self.step / (2 * self.step_size))
        x = np.abs(self.step / self.step_size - 2 * cycle + 1)
        lr = self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x))
        self.model.optimizer.learning_rate.assign(lr)
        logging.debug(f"Step {self.step}: Setting learning rate to {lr:.6f}")


# Data augmentation
def augment_audio(audio):
    # Noise injection
    if np.random.random() < 0.5:
        audio += np.random.normal(0, 0.01 * np.max(audio), audio.shape)
    # Pitch shift
    if np.random.random() < 0.5:
        audio = librosa.effects.pitch_shift(y=audio, sr=SAMPLE_RATE, n_steps=np.random.uniform(-2, 2))
    # Time stretch
    if np.random.random() < 0.5:
        audio = librosa.effects.time_stretch(y=audio, rate=np.random.uniform(0.8, 1.2))
    # Time masking
    if np.random.random() < 0.5:
        mask_start = np.random.randint(0, len(audio) // 2)
        mask_length = np.random.randint(1000, 5000)
        audio[mask_start:mask_start + mask_length] = 0
    return audio


# Noise reduction (bandpass filter)
def reduce_noise(audio):
    def butter_bandpass(lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def bandpass_filter(data, lowcut=20, highcut=400, fs=SAMPLE_RATE, order=5):
        b, a = butter_bandpass(lowcut, highcut, fs, order=order)
        return lfilter(b, a, data)

    return bandpass_filter(audio)


# Segment cardiac cycles using Hilbert transform
def segment_cardiac_cycles(audio, sr):
    analytic_signal = hilbert(audio)
    amplitude_envelope = np.abs(analytic_signal)
    peaks, _ = find_peaks(amplitude_envelope, height=np.mean(amplitude_envelope) + np.std(amplitude_envelope))
    segments = []
    for i in range(len(peaks) - 1):
        start = peaks[i]
        end = peaks[i + 1]
        if end - start >= sr * 0.2:  # Minimum cycle length (0.2s)
            segments.append(audio[start:end])
    if not segments:
        segments.append(audio[:sr])  # Fallback to first second if no cycles detected
    return segments


# Load and preprocess data
def load_data(debug_limit=None):
    logging.info(f"Loading training_data.csv from {CSV_PATH}")
    try:
        df = pd.read_csv(CSV_PATH)
    except Exception as e:
        logging.error(f"Failed to load CSV file: {e}")
        raise

    df = df[df['Murmur'].isin(['Absent', 'Present'])]
    if df.empty:
        logging.error("No valid records found with Murmur 'Absent' or 'Present'.")
        raise ValueError("No valid records found with Murmur 'Absent' or 'Present'.")

    df = df.reset_index(drop=True)
    labels = df['Murmur'].map({'Absent': 0, 'Present': 1}).values
    logging.info(f"Filtered dataset to {len(df)} records with valid Murmur labels")

    if debug_limit:
        df = df[:debug_limit]
        labels = labels[:debug_limit]
        logging.info(f"Debug mode: Processing only {debug_limit} records")

    features = []
    valid_labels = []

    logging.info("Extracting features from PCG files")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        patient_id = row['Patient ID']
        locations = row['Recording locations:'].split('+')
        patient_features = []

        logging.debug(f"Processing patient {patient_id} with locations {locations}")
        for loc in locations:
            wav_file = os.path.join(DATA_PATH, f"{patient_id}_{loc}.wav")
            if not os.path.exists(wav_file):
                logging.warning(f"File not found: {wav_file}")
                continue

            try:
                # Load audio with original sampling rate
                audio, sr = librosa.load(wav_file, sr=None, duration=DURATION)
                if sr != SAMPLE_RATE:
                    logging.warning(f"Sampling rate mismatch for {wav_file}: {sr} Hz, resampled to {SAMPLE_RATE} Hz")
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
                if np.max(np.abs(audio)) < 1e-6:
                    logging.warning(f"Audio too quiet for {wav_file}")
                    continue

                # Noise reduction
                audio = reduce_noise(audio)
                # Apply augmentation
                audio = augment_audio(audio)
                target_length = SAMPLE_RATE * DURATION
                if len(audio) < target_length:
                    audio = np.pad(audio, (0, target_length - len(audio)))
                else:
                    audio = audio[:target_length]

                # Segment into cardiac cycles
                cycles = segment_cardiac_cycles(audio, SAMPLE_RATE)

                # Extract STFT-based features for each cycle
                cycle_features = []
                for cycle in cycles:
                    # STFT with 25 ms window and 10 ms overlap
                    hop_length = int(0.010 * SAMPLE_RATE)  # 10 ms overlap
                    win_length = int(0.025 * SAMPLE_RATE)  # 25 ms window
                    stft = librosa.stft(cycle, n_fft=win_length, hop_length=hop_length)
                    spectrogram = np.abs(stft)
                    spectrogram = spectrogram[:, :MAX_TIME]
                    if spectrogram.shape[1] < MAX_TIME:
                        spectrogram = np.pad(spectrogram, ((0, 0), (0, MAX_TIME - spectrogram.shape[1])),
                                             mode='constant')

                    # Extract MFCC and delta MFCC
                    mfcc = librosa.feature.mfcc(y=cycle, sr=SAMPLE_RATE, n_mfcc=N_MFCC, n_fft=win_length,
                                                hop_length=hop_length)
                    mfcc = mfcc[:, :MAX_TIME]
                    if mfcc.shape[1] < MAX_TIME:
                        mfcc = np.pad(mfcc, ((0, 0), (0, MAX_TIME - mfcc.shape[1])), mode='constant')
                    delta_mfcc = librosa.feature.delta(mfcc)

                    # Concatenate features
                    combined = np.concatenate([spectrogram, mfcc, delta_mfcc], axis=0)
                    cycle_features.append(combined)

                if cycle_features:
                    patient_features.append(np.mean(cycle_features, axis=0))
            except Exception as e:
                logging.error(f"Error processing {wav_file}: {e}")
                continue

        if patient_features:
            features.append(np.mean(patient_features, axis=0))
            valid_labels.append(labels[idx])
        else:
            logging.warning(f"No valid features extracted for patient {patient_id}")

    if not features:
        logging.error("No valid features extracted for any patient.")
        raise ValueError("No valid features extracted for any patient.")

    features = np.array(features)
    valid_labels = np.array(valid_labels)
    logging.info(f"Loaded {len(features)} samples with shape {features.shape}")
    return features, valid_labels


# Balance dataset using SMOTE-Tomek
def balance_dataset(X, y):
    logging.info("Applying SMOTE-Tomek for class balancing")
    try:
        smote_tomek = SMOTETomek(random_state=42)
        X_resampled, y_resampled = smote_tomek.fit_resample(X.reshape(X.shape[0], -1), y)
        actual_freq_dim = X.shape[1]  # Dynamic based on loaded shape
        X_resampled = X_resampled.reshape(-1, actual_freq_dim, MAX_TIME, 1)
        logging.info(f"Balanced dataset: {X_resampled.shape}, {y_resampled.shape}")
        return X_resampled, y_resampled
    except Exception as e:
        logging.error(f"Error in SMOTE-Tomek balancing: {e}")
        raise


# CNN-LSTM model
def build_model(input_shape=(131, 128, 1)):  # Updated to match new feature dimension
    inputs = layers.Input(shape=input_shape)

    # Enhanced CNN block
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.5)(x)

    # Reshape for LSTM (timesteps, features)
    x = layers.Reshape((-1, x.shape[-1] * x.shape[-2]))(x)

    # Enhanced Bidirectional LSTM layers
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(x)
    x = layers.Dropout(0.5)(x)
    # Removed the third LSTM layer; use pooling instead
    # x = layers.Bidirectional(layers.LSTM(256))(x)
    # x = layers.Dropout(0.5)(x)

    # Global average pooling over timesteps
    x = layers.GlobalAveragePooling1D()(x)

    # Output layer
    outputs = layers.Dense(N_CLASSES, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# Test-time augmentation
def test_time_augmentation(model, X, n_augmentations=5):
    predictions = []
    try:
        for i in range(n_augmentations):
            logging.debug(f"Performing TTA augmentation {i + 1}/{n_augmentations}")
            noise = np.random.normal(0, 0.01, X.shape)
            X_aug = X + noise
            preds = model.predict(X_aug, verbose=0)
            if np.any(np.isnan(preds)) or np.any(np.isinf(preds)):
                logging.warning(f"NaN or Inf detected in TTA predictions for augmentation {i + 1}")
                continue
            predictions.append(preds)
        if not predictions:
            logging.error("No valid predictions from TTA")
            raise ValueError("No valid predictions from TTA")
        return np.mean(predictions, axis=0)
    except Exception as e:
        logging.error(f"Error in TTA prediction: {e}")
        raise


# Plot and save results
def plot_results(history, y_test, y_pred, save_dir):
    try:
        plt.figure(figsize=(8, 6))
        class_counts = pd.Series(y_test).value_counts()
        plt.pie(class_counts, labels=CLASS_NAMES, autopct='%1.1f%%', startangle=90)
        plt.title('Class Distribution (Test Set)')
        plt.savefig(os.path.join(save_dir, 'class_distribution.png'))
        plt.close()

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Val Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(save_dir, 'training_metrics.png'))
        plt.close()

        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
        plt.close()
    except Exception as e:
        logging.error(f"Error plotting results: {e}")
        raise


# Main pipeline
def main():
    # Load data
    X, y = load_data(debug_limit=None)

    # Standardize features
    try:
        scaler = StandardScaler()
        X = scaler.fit_transform(X.reshape(X.shape[0], -1)).reshape(X.shape)
    except Exception as e:
        logging.error(f"Error in feature standardization: {e}")
        raise

    # Split data
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    except Exception as e:
        logging.error(f"Error in train-test split: {e}")
        raise

    # Balance training data
    X_train, y_train = balance_dataset(X_train, y_train)

    # Reshape for CNN input
    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    # Compute class weights dynamically
    class_counts = pd.Series(y_train).value_counts()
    class_weights = {
        0: 1.0,
        1: (class_counts[0] / class_counts[1]) * 3.0  # Adjusted weight for better balance
    }

    # Build and train model
    model = build_model(input_shape=(X_train.shape[1], X_train.shape[2], 1))  # Dynamic input shape
    callbacks = [
        EarlyStopping(patience=30, restore_best_weights=True),
        ModelCheckpoint(os.path.join(SAVE_DIR, 'best_model.keras'), save_best_only=True),
        CyclicalLearningRate(base_lr=0.0001, max_lr=0.001, step_size=2000,
                             total_steps=EPOCHS * len(X_train) // BATCH_SIZE)
    ]

    try:
        history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                            batch_size=BATCH_SIZE, epochs=EPOCHS,
                            class_weight=class_weights, callbacks=callbacks)
    except Exception as e:
        logging.error(f"Error during model training: {e}")
        raise

    # Test-time augmentation
    y_pred_proba = test_time_augmentation(model, X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)

    # Evaluate
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn + 1e-10)
    specificity = tn / (tn + fp + 1e-10)
    logging.info(f"Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}")
    logging.info("\nClassification Report:\n" + classification_report(y_test, y_pred, target_names=CLASS_NAMES))

    # Save results
    plot_results(history, y_test, y_pred, SAVE_DIR)
    model.save(os.path.join(SAVE_DIR, 'final_model.keras'))
    np.save(os.path.join(SAVE_DIR, 'X_test.npy'), X_test)
    np.save(os.path.join(SAVE_DIR, 'y_test.npy'), y_test)

    logging.info(f"Results saved in {SAVE_DIR}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        raise
