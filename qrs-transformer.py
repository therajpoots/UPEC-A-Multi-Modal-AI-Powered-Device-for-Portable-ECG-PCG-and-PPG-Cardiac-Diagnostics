import pandas as pd
import numpy as np
import wfdb
import ast
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback, ReduceLROnPlateau
from imblearn.combine import SMOTETomek
import pickle
import glob
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define paths
path = r'J:\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3'
records_path = os.path.join(path, 'records100')
features_path = os.path.join(path, 'features')
sampling_rate = 100

# Load raw data
def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(os.path.join(path, f.replace('records100/', ''))) for f in tqdm(df.filename_lr)]
    else:
        data = [wfdb.rdsamp(os.path.join(path, f.replace('records500/', ''))) for f in tqdm(df.filename_hr)]
    data = np.array([signal for signal, meta in data])
    return data

# Load and convert annotation data
Y = pd.read_csv(os.path.join(path, 'ptbxl_database.csv'), index_col='ecg_id')
Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

# Load raw data
X = load_raw_data(Y, sampling_rate=sampling_rate, path=records_path)
logger.info(f'Data shape: {X.shape}')

# Load scp_statements.csv for diagnostic aggregation
agg_df = pd.read_csv(os.path.join(path, 'scp_statements.csv'), index_col=0)
agg_df = agg_df[agg_df.diagnostic == 1]

def aggregate_superclass_diagnostic(y_dic):
    tmp = []
    for key in y_dic.keys():
        if key in agg_df.index:
            tmp.append(agg_df.loc[key].diagnostic_class)
    return list(set(tmp))

# Apply diagnostic superclass
Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_superclass_diagnostic)
Y['diagnostic_superclass_len'] = Y['diagnostic_superclass'].apply(len)

# Plot diagnostic superclass length distribution
vc = Y['diagnostic_superclass_len'].value_counts()
sns.set_style("whitegrid")
plt.figure(figsize=(10, 6))
ax = sns.barplot(x=vc.values / vc.values.sum() * 100., y=vc.index.astype(str), palette="muted", orient='h')
ax.set_title("Diagnostic Superclass Length Distribution", fontsize=20)
ax.set_xlabel("Percentage over all samples")
ax.set_ylabel("Diagnostic superclass length")
for rect in ax.patches:
    ax.text(rect.get_width(), rect.get_y() + rect.get_height() / 2, "%.1f%%" % rect.get_width(), weight='bold')
plt.savefig('superclass_len_distribution.png')
plt.close()

def aggregate_subclass_diagnostic(y_dic):
    tmp = []
    for key in y_dic.keys():
        if key in agg_df.index:
            tmp.append(agg_df.loc[key].diagnostic_subclass)
    ret = list(set(tmp))
    ret = ['sub_' + r for r in ret if r]
    return ret

# Apply diagnostic subclass
Y['diagnostic_subclass'] = Y.scp_codes.apply(aggregate_subclass_diagnostic)
Y['diagnostic_subclass_len'] = Y['diagnostic_subclass'].apply(len)

# Plot diagnostic subclass length distribution
vc = Y['diagnostic_subclass_len'].value_counts()
sns.set_style("whitegrid")
plt.figure(figsize=(10, 6))
ax = sns.barplot(x=vc.values / vc.values.sum() * 100., y=vc.index.astype(str), palette="muted", orient='h')
ax.set_title("Diagnostic Subclass Length Distribution", fontsize=20)
ax.set_xlabel("Percentage over all samples")
ax.set_ylabel("Diagnostic subclass length")
for rect in ax.patches:
    ax.text(rect.get_width(), rect.get_y() + rect.get_height() / 2, "%.1f%%" % rect.get_width(), weight='bold')
plt.savefig('subclass_len_distribution.png')
plt.close()

# Prepare metadata and feature columns
all_superclass = pd.Series(np.concatenate(Y['diagnostic_superclass'].values))
all_subclass = pd.Series(np.concatenate(Y['diagnostic_subclass'].values))
superclass_cols = all_superclass.unique()
subclass_cols = all_subclass.unique()
update_cols = np.concatenate([superclass_cols, subclass_cols])
meta_cols = ['age', 'sex', 'height', 'weight', 'nurse', 'site', 'device']

class ClassUpdate:
    def __init__(self, cols):
        self.cols = cols

    def __call__(self, row):
        for sc in row['diagnostic_superclass']:
            row[sc] = 1
        for sc in row['diagnostic_subclass']:
            row[sc] = 1
        return row

def get_data_by_folds(folds, x, y, update_cols, feature_cols):
    assert len(folds) > 0, '# of provided folds should longer than 1'
    filt = np.isin(y.strat_fold.values, folds)
    x_selected = x[filt]
    y_selected = y[filt]
    for sc in update_cols:
        y_selected[sc] = 0
    cls_updt = ClassUpdate(update_cols)
    y_selected = y_selected.apply(cls_updt, axis=1)
    return x_selected, y_selected[list(feature_cols) + list(update_cols) + ['strat_fold']]

x_all, y_all = get_data_by_folds(np.arange(1, 11), X, Y, update_cols, meta_cols)

# Plot diagnostic superclass distribution
vc = y_all[superclass_cols].sum(axis=0)
sns.set_style("whitegrid")
plt.figure(figsize=(10, 6))
ax = sns.barplot(x=vc.values / y_all.shape[0] * 100., y=vc.index, palette="muted", orient='h')
ax.set_title("Diagnostic Superclass Distribution", fontsize=20)
ax.set_xlabel("Percentage over all samples")
ax.set_ylabel("Diagnostic superclass")
for rect in ax.patches:
    ax.text(rect.get_width(), rect.get_y() + rect.get_height() / 2, "%.1f%%" % rect.get_width(), weight='bold')
plt.savefig('superclass_distribution.png')
plt.close()

# Load extracted features
extfea_data = pd.read_csv(os.path.join(features_path, 'ecgdeli_features.csv'), index_col='ecg_id')

# Align data
Y_new = Y[:extfea_data.shape[0]]
X_new = X[:extfea_data.shape[0]]

# Assign features to X_new (only Lead I and generic features)
X_new = Y_new.assign(
    RR_Mean_Global=extfea_data['RR_Mean_Global'],
    ST_Elev_I=extfea_data['ST_Elev_I'],
    PR_Int_Global=extfea_data['PR_Int_Global'],
    PR_Int_I=extfea_data['PR_Int_I'],
    PQ_Int_Global=extfea_data['PQ_Int_Global'],
    PQ_Int_I=extfea_data['PQ_Int_I'],
    QRS_Dur_I=extfea_data['QRS_Dur_I'],
    QT_Int_I=extfea_data['QT_Int_I'],
    R_Amp_I=extfea_data['R_Amp_I'],
    Q_Amp_I=extfea_data['Q_Amp_I'],
    P_Amp_I=extfea_data['P_Amp_I']
)

# Validate and clip features
for col in ['RR_Mean_Global', 'ST_Elev_I', 'PR_Int_Global', 'PR_Int_I', 'PQ_Int_Global', 'PQ_Int_I',
            'QRS_Dur_I', 'QT_Int_I', 'R_Amp_I', 'Q_Amp_I', 'P_Amp_I']:
    if col not in X_new.columns:
        raise ValueError(f"Feature {col} not found in X_new")
    X_new[col] = X_new[col].clip(lower=X_new[col].quantile(0.01), upper=X_new[col].quantile(0.99))
    logger.info(f"Feature {col} - Mean: {X_new[col].mean():.4f}, Std: {X_new[col].std():.4f}, NaN count: {X_new[col].isna().sum()}")

# Select records with single superclass label
X_selected = X_new[Y_new['diagnostic_superclass_len'] == 1]
Y_selected = Y_new[Y_new['diagnostic_superclass_len'] == 1]

# Define selected features
selected_features = [
    'RR_Mean_Global', 'ST_Elev_I', 'PR_Int_Global', 'PR_Int_I',
    'PQ_Int_Global', 'PQ_Int_I', 'QRS_Dur_I', 'QT_Int_I',
    'R_Amp_I', 'Q_Amp_I', 'P_Amp_I'
]
X_selected_feature = X_selected[selected_features].fillna(0)

# Create labels for 5 classes
labels = []
for i in range(len(Y_selected)):
    superclass = Y_selected['diagnostic_superclass'].iloc[i]
    if superclass == ['NORM']:
        labels.append('NORM')
    elif superclass == ['MI']:
        labels.append('MI')
    elif superclass == ['STTC']:
        labels.append('STTC')
    elif superclass == ['CD']:
        labels.append('CD')
    elif superclass == ['HYP']:
        labels.append('HYP')

df_labels = pd.DataFrame(labels, columns=['labels'])
logger.info(f"Class distribution before oversampling:\n{df_labels['labels'].value_counts()}")

# Plot class distribution before oversampling
class_distribution_before = df_labels['labels'].value_counts()
plt.figure(figsize=(8, 6))
plt.pie(class_distribution_before, labels=class_distribution_before.index, autopct='%1.1f%%', shadow=True)
plt.title('Class Distribution Before Oversampling')
plt.axis('equal')
plt.savefig('class_distribution_before.png')
plt.close()

# Add strat_fold to labels
df_labels['strat_fold'] = Y_selected['strat_fold'].values

# Feature selection using mutual information
X_features = X_selected_feature.values
y_labels = pd.Categorical(df_labels['labels']).codes
selector = SelectKBest(score_func=mutual_info_classif, k=10)
X_selected_top = selector.fit_transform(X_features, y_labels)
selected_indices = selector.get_support(indices=True)
top_features = [selected_features[i] for i in selected_indices]
logger.info(f"Selected top 10 features: {top_features}")

# Create feature and label dataframe with top features
df_features_label = pd.DataFrame(X_selected_top, columns=top_features)
df_features_label['labels'] = df_labels['labels'].values
df_features_label['strat_fold'] = df_labels['strat_fold'].values

# Separate classes
class_norm = df_features_label[df_features_label['labels'] == 'NORM']
class_mi = df_features_label[df_features_label['labels'] == 'MI']
class_sttc = df_features_label[df_features_label['labels'] == 'STTC']
class_cd = df_features_label[df_features_label['labels'] == 'CD']
class_hyp = df_features_label[df_features_label['labels'] == 'HYP']

# Assign integer labels
class_norm = class_norm.assign(labels_int=0)
class_mi = class_mi.assign(labels_int=1)
class_sttc = class_sttc.assign(labels_int=2)
class_cd = class_cd.assign(labels_int=3)
class_hyp = class_hyp.assign(labels_int=4)

# Prepare features for SMOTE
class_norm_features = class_norm[top_features].to_numpy()
class_mi_features = class_mi[top_features].to_numpy()
class_sttc_features = class_sttc[top_features].to_numpy()
class_cd_features = class_cd[top_features].to_numpy()
class_hyp_features = class_hyp[top_features].to_numpy()

# Combine features and labels
X_for_smt = np.concatenate(
    (class_norm_features, class_mi_features, class_sttc_features, class_cd_features, class_hyp_features), axis=0)
y_for_smt = np.concatenate(
    (class_norm['labels_int'].values, class_mi['labels_int'].values, class_sttc['labels_int'].values,
     class_cd['labels_int'].values, class_hyp['labels_int'].values), axis=0)

# Apply SMOTE-Tomek
smote_tomek = SMOTETomek(random_state=42, sampling_strategy='auto')
X_oversampled, y_oversampled = smote_tomek.fit_resample(X_for_smt, y_for_smt)

# Create balanced features dataframe
balanced_features_df = pd.DataFrame(X_oversampled, columns=top_features)
balanced_features_df['labels'] = y_oversampled

# Plot class distribution after oversampling
class_distribution_after = balanced_features_df['labels'].value_counts()
plt.figure(figsize=(8, 6))
plt.pie(class_distribution_after, labels=['NORM', 'MI', 'STTC', 'CD', 'HYP'], autopct='%1.1f%%', shadow=True)
plt.title('Class Distribution After Oversampling')
plt.axis('equal')
plt.savefig('class_distribution_after.png')
plt.close()
logger.info(f"Class distribution after oversampling:\n{class_distribution_after}")

# Prepare features and labels
X_balanced_features = balanced_features_df[top_features].values
y_balanced_features = balanced_features_df['labels'].values

# Train-test split with stratification
X_train_features, X_test_features, y_train_features, y_test_features = train_test_split(
    X_balanced_features, y_balanced_features, test_size=0.2, random_state=42, shuffle=True, stratify=y_balanced_features)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_features)
X_test_scaled = scaler.transform(X_test_features)

# Check feature normalization
logger.info(f"Training features mean: {X_train_scaled.mean(axis=0)}")
logger.info(f"Training features std: {X_train_scaled.std(axis=0)}")

# Reshape for Transformer
X_train_reshaped = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
X_test_reshaped = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)

# Define named labels
named_labels = np.array(['NORM', 'MI', 'STTC', 'CD', 'HYP'])

# Convert labels to one-hot encoding
y_train_onehot = to_categorical(y_train_features, num_classes=5)
y_test_onehot = to_categorical(y_test_features, num_classes=5)

# Compute class weights
class_counts = np.sum(y_train_onehot, axis=0)
class_weights = {i: min(len(y_train_features) / (5 * count), 7.0) for i, count in enumerate(class_counts) if count > 0}
logger.info(f"Class weights: {class_weights}")

# Set environment for GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Define Focal Loss
class FocalLoss(keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1.0 - y_pred)
        ce = -tf.math.log(pt)
        alpha_t = tf.where(tf.equal(y_true, 1), self.alpha, 1.0 - self.alpha)
        fl = alpha_t * tf.pow(1.0 - pt, self.gamma) * ce
        return tf.reduce_mean(fl)

    def get_config(self):
        config = super().get_config()
        config.update({'gamma': self.gamma, 'alpha': self.alpha})
        return config

# Define TransformerBlock
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.3, layer_norm_eps=1e-6, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        self.layer_norm_eps = layer_norm_eps

        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads)
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="gelu", kernel_regularizer=regularizers.l2(1e-3)),
            layers.Dense(embed_dim, kernel_regularizer=regularizers.l2(1e-3)),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=layer_norm_eps)
        self.layernorm2 = layers.LayerNormalization(epsilon=layer_norm_eps)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)

    def call(self, inputs, training=False):
        if isinstance(inputs, tf.SparseTensor):
            inputs = tf.sparse.to_dense(inputs)
        inputs = tf.convert_to_tensor(inputs)
        attn_output = self.att(inputs, inputs, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'dropout_rate': self.dropout_rate,
            'layer_norm_eps': self.layer_norm_eps
        })
        return config

# Define AttentionPooling
class AttentionPooling(layers.Layer):
    def __init__(self, embed_dim, **kwargs):
        super(AttentionPooling, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.query = layers.Dense(embed_dim)
        self.softmax = layers.Softmax(axis=1)

    def call(self, inputs):
        query = self.query(inputs)
        attention_weights = self.softmax(query)
        weighted_sum = tf.reduce_sum(inputs * attention_weights, axis=1)
        return weighted_sum

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.embed_dim)

    def get_config(self):
        config = super().get_config()
        config.update({'embed_dim': self.embed_dim})
        return config

# Custom Learning Rate Schedule (Cyclic)
class CyclicLR(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_lr, max_lr, step_size, total_steps):
        super(CyclicLR, self).__init__()
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.total_steps = total_steps

    def __call__(self, step):
        cycle = tf.floor(1 + step / (2 * self.step_size))
        x = tf.abs(step / self.step_size - 2 * cycle + 1)
        lr = self.base_lr + (self.max_lr - self.base_lr) * tf.maximum(0.0, (1 - x))
        return lr

    def get_config(self):
        return {
            'base_lr': self.base_lr,
            'max_lr': self.max_lr,
            'step_size': self.step_size,
            'total_steps': self.total_steps
        }

# Define optimized CNN-Transformer model
def build_cnn_transformer_model():
    input_shape = (10, 1)
    num_classes = 5

    # Hyperparameters
    embed_dim = 332
    num_heads = 10
    ff_dim = 665
    num_cnn_layers = 4
    num_transformer_blocks = 8
    dropout_rate = 0.3
    weight_decay = 1e-3

    inputs = keras.Input(shape=input_shape)

    # Feature expansion
    x = layers.Dense(20, activation='gelu', kernel_regularizer=regularizers.l2(weight_decay))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Reshape((-1, 1))(x)

    # Data augmentation
    def augment(x):
        x = x * tf.random.uniform([tf.shape(x)[0], 1, 1], 0.95, 1.05)
        x = x + tf.random.uniform([tf.shape(x)[0], 1, 1], -0.03, 0.03)
        return x
    x = layers.Lambda(augment, output_shape=(20, 1))(x)
    x = layers.GaussianNoise(stddev=0.015)(x, training=True)

    # CNN layers
    filters = 83
    shortcut = x
    pooling_count = 0
    for i in range(num_cnn_layers):
        x = layers.Conv1D(
            filters=filters,
            kernel_size=3,
            padding='same',
            activation='gelu',
            kernel_initializer='he_normal',
            kernel_regularizer=regularizers.l2(weight_decay)
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(x)
        pooling_count += 1

        if i % 2 == 1 and i > 0:
            shortcut = layers.Conv1D(filters, 1, padding='same')(shortcut)
            for _ in range(pooling_count):
                shortcut = layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(shortcut)
            x = layers.Add()([x, shortcut])
            shortcut = x
            pooling_count = 0

        x = layers.Dropout(dropout_rate)(x, training=True)
        filters = min(filters * 2, 665)

    # Reshape for Transformer
    x = layers.Conv1D(filters=embed_dim, kernel_size=1, padding='same')(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)

    # Transformer blocks
    for _ in range(num_transformer_blocks):
        x_old = x
        transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim, dropout_rate)
        x = transformer_block(x, training=True)
        x = 0.8 * x + 0.2 * x_old

    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = AttentionPooling(embed_dim)(x)
    x = layers.Dropout(dropout_rate)(x, training=True)

    # Dense layers with residual
    x_dense = layers.Dense(332, activation='gelu', kernel_initializer='he_normal',
                           kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = layers.BatchNormalization()(x_dense)
    x = layers.Dropout(dropout_rate)(x, training=True)
    x = layers.Dense(166, activation='gelu', kernel_initializer='he_normal',
                     kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, layers.Dense(166, kernel_initializer='he_normal')(x_dense)])
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    # Cyclic learning rate
    lr_schedule = CyclicLR(
        base_lr=1e-4,
        max_lr=3e-4,
        step_size=2000,
        total_steps=12000
    )

    # Compile with AdamW
    optimizer = keras.optimizers.AdamW(
        learning_rate=lr_schedule,
        weight_decay=weight_decay,
        clipnorm=0.5
    )
    model.compile(
        optimizer=optimizer,
        loss=FocalLoss(gamma=2.0, alpha=0.25),
        metrics=['accuracy']
    )
    return model

# GPU strategy
if os.environ["CUDA_VISIBLE_DEVICES"].count(',') == 0:
    gpu_strategy = tf.distribute.get_strategy()
    logger.info('Single strategy')
else:
    gpu_strategy = tf.distribute.MirroredStrategy()
    logger.info('Multiple strategy')

logger.info('Mixed precision disabled')

# Custom callback for weighted F1-score
class WeightedF1Callback(Callback):
    def __init__(self, val_X, val_y):
        super(WeightedF1Callback, self).__init__()
        self.val_X = val_X
        self.val_y = val_y

    def on_epoch_end(self, epoch, logs=None):
        y_pred = self.model.predict(self.val_X, verbose=0)
        y_pred_labels = np.argmax(y_pred, axis=1)
        y_true_labels = np.argmax(self.val_y, axis=1)
        f1 = f1_score(y_true_labels, y_pred_labels, average='weighted')
        logs['val_weighted_f1_score'] = f1
        logger.info(f"Epoch {epoch+1} - val_weighted_f1_score: {f1:.4f}")

# Test-time augmentation
def tta_predict(model, X, n_augmentations=5):
    predictions = []
    for _ in range(n_augmentations):
        X_aug = X * np.random.uniform(0.95, 1.05, size=(X.shape[0], 1, 1)) + \
                np.random.uniform(-0.03, 0.03, size=(X.shape[0], 1, 1))
        X_aug += np.random.normal(0, 0.015, size=X.shape)
        preds = model.predict(X_aug, verbose=0)
        predictions.append(preds)
    return np.mean(predictions, axis=0)

# Custom objects
custom_objects = {
    'TransformerBlock': TransformerBlock,
    'AttentionPooling': AttentionPooling,
    'CyclicLR': CyclicLR,
    'FocalLoss': FocalLoss
}

# Check for existing checkpoints
saved_models = glob.glob('model_epoch_*.keras')
initial_epoch = 0
model = None

if saved_models:
    try:
        latest_model = max(saved_models, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        epoch_num = int(latest_model.split('_')[-1].split('.')[0])
        model = keras.models.load_model(latest_model, custom_objects=custom_objects)
        initial_epoch = epoch_num + 1
        logger.info(f"Resuming training from epoch {initial_epoch} using checkpoint: {latest_model}")
    except Exception as e:
        logger.warning(f"Error loading checkpoint: {e}. Starting from scratch.")
        with gpu_strategy.scope():
            model = build_cnn_transformer_model()
else:
    logger.info("No checkpoints found. Starting training from scratch.")
    with gpu_strategy.scope():
        model = build_cnn_transformer_model()

# Define callbacks
checkpoint = ModelCheckpoint(
    'best_model.keras',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    save_weights_only=False
)
periodic_checkpoint = ModelCheckpoint(
    'model_epoch_{epoch:03d}.keras',
    save_weights_only=False,
    save_freq='epoch'
)
early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=30,
    restore_best_weights=True,
    mode='max'
)
reduce_lr = ReduceLROnPlateau(
    monitor='val_accuracy',
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    mode='max'
)

# Train model with stratified k-fold
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
fold_results = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_balanced_features, y_balanced_features)):
    logger.info(f"Training fold {fold+1}/{n_splits}")
    X_train_fold = X_balanced_features[train_idx]
    y_train_fold = y_balanced_features[train_idx]
    X_val_fold = X_balanced_features[val_idx]
    y_val_fold = y_balanced_features[val_idx]

    X_train_fold_scaled = scaler.fit_transform(X_train_fold)
    X_val_fold_scaled = scaler.transform(X_val_fold)

    X_train_fold_reshaped = X_train_fold_scaled.reshape(X_train_fold_scaled.shape[0], X_train_fold_scaled.shape[1], 1)
    X_val_fold_reshaped = X_val_fold_scaled.reshape(X_val_fold_scaled.shape[0], X_val_fold_scaled.shape[1], 1)

    y_train_fold_onehot = to_categorical(y_train_fold, num_classes=5)
    y_val_fold_onehot = to_categorical(y_val_fold, num_classes=5)

    with gpu_strategy.scope():
        model = build_cnn_transformer_model()

    weighted_f1_callback = WeightedF1Callback(X_val_fold_reshaped, y_val_fold_onehot)

    history = model.fit(
        X_train_fold_reshaped,
        y_train_fold_onehot,
        validation_data=(X_val_fold_reshaped, y_val_fold_onehot),
        epochs=100,
        batch_size=256,
        callbacks=[early_stopping, checkpoint, periodic_checkpoint, weighted_f1_callback, reduce_lr],
        class_weight=class_weights,
        verbose=1,
        initial_epoch=0,
        shuffle=True
    )

    val_pred = tta_predict(model, X_val_fold_reshaped)
    val_pred_labels = np.argmax(val_pred, axis=1)
    val_true_labels = np.argmax(y_val_fold_onehot, axis=1)
    val_accuracy = np.mean(val_pred_labels == val_true_labels)
    val_f1 = f1_score(val_true_labels, val_pred_labels, average='weighted')
    fold_results.append((val_accuracy, val_f1))
    logger.info(f"Fold {fold+1} - Validation Accuracy: {val_accuracy:.4f}, F1 Score: {val_f1:.4f}")

# Train final model
with gpu_strategy.scope():
    model = build_cnn_transformer_model()

weighted_f1_callback = WeightedF1Callback(X_test_reshaped, y_test_onehot)

history = model.fit(
    X_train_reshaped,
    y_train_onehot,
    validation_data=(X_test_reshaped, y_test_onehot),
    epochs=100,
    batch_size=256,
    callbacks=[early_stopping, checkpoint, periodic_checkpoint, weighted_f1_callback, reduce_lr],
    class_weight=class_weights,
    verbose=1,
    initial_epoch=initial_epoch,
    shuffle=True
)

# Evaluate model
Y_pred = tta_predict(model, X_test_reshaped)
Y_pred_labels = np.argmax(Y_pred, axis=1)
y_true_labels = np.argmax(y_test_onehot, axis=1)

# Confusion matrix
cf_matrix = confusion_matrix(y_true_labels, Y_pred_labels, labels=np.arange(len(named_labels)))
cf_matrix_percentage = cf_matrix / np.sum(cf_matrix, axis=1, keepdims=True) * 100

# Test set metrics
true_positive = np.diag(cf_matrix)
false_negative = np.sum(cf_matrix, axis=1) - true_positive
false_positive = np.sum(cf_matrix, axis=0) - true_positive
true_negative = np.sum(cf_matrix) - (true_positive + false_positive + false_negative)
test_sensitivity = true_positive / (true_positive + false_negative + 1e-10)
test_specificity = true_negative / (true_negative + false_positive + 1e-10)

# Print results
logger.info("Test Set Results:")
logger.info("Confusion Matrix (Counts):")
logger.info(cf_matrix)
logger.info("Confusion Matrix (Percentage):")
logger.info(np.round(cf_matrix_percentage, 4))
logger.info(f"Test Sensitivity per Class: {test_sensitivity}")
logger.info(f"Test Specificity per Class: {test_specificity}")
logger.info("\nClassification Report:")
logger.info(classification_report(y_true_labels, Y_pred_labels, target_names=named_labels))

# Plot training and validation metrics
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(history.history['val_weighted_f1_score'], label='Validation F1')
plt.title('F1 Score')
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.legend()

plt.tight_layout()
plt.savefig('metrics.png')
plt.close()

# Plot test set confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cf_matrix_percentage, annot=True, fmt='.2f', cmap='Blues', xticklabels=named_labels,
            yticklabels=named_labels)
plt.title('Confusion Matrix - Test Set (Percentage)')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.savefig('test_confusion_matrix.png')
plt.close()

# Save model and test data
model.save('qrs_transformer_optimized.keras', include_optimizer=True)

with open('test_data.npy', 'wb') as f:
    np.save(f, X_test_reshaped)

with open('test_labels_onehot.npy', 'wb') as f:
    np.save(f, y_test_onehot)

# Save scaler and features
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('top_features.pkl', 'wb') as f:
    pickle.dump(top_features, f)

# Log cross-validation resultsxxxxxxxxxxxxxx
logger.info(f"Cross-validation results: {fold_results}")
logger.info(f"Average validation accuracy: {np.mean([r[0] for r in fold_results]):.4f}")
logger.info(f"Average validation F1 score: {np.mean([r[1] for r in fold_results]):.4f}")

if __name__ == "__main__":
    logger.info("QRS Transformer classification completed successfully.")