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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from imblearn.combine import SMOTETomek
import pickle
import glob
import json

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
print('Data shape:', X.shape)

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
ax = sns.barplot(x=vc.values / vc.values.sum() * 100., y=vc.index, palette="muted", orient='h')
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
ax = sns.barplot(x=vc.values / vc.values.sum() * 100., y=vc.index, palette="muted", orient='h')
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

# Assign features to X_new
X_new = Y_new.assign(
    RR_Mean_Global=extfea_data['RR_Mean_Global'],
    ST_Elev_I=extfea_data['ST_Elev_I'], ST_Elev_II=extfea_data['ST_Elev_II'], ST_Elev_III=extfea_data['ST_Elev_III'],
    ST_Elev_V1=extfea_data['ST_Elev_V1'], ST_Elev_V2=extfea_data['ST_Elev_V2'], ST_Elev_V3=extfea_data['ST_Elev_V3'],
    ST_Elev_V4=extfea_data['ST_Elev_V4'], ST_Elev_V5=extfea_data['ST_Elev_V5'], ST_Elev_V6=extfea_data['ST_Elev_V6'],
    ST_Elev_aVF=extfea_data['ST_Elev_aVF'], ST_Elev_aVR=extfea_data['ST_Elev_aVR'],
    ST_Elev_aVL=extfea_data['ST_Elev_aVL'],
    PR_Int_Global=extfea_data['PR_Int_Global'], PR_Int_I=extfea_data['PR_Int_I'], PR_Int_II=extfea_data['PR_Int_II'],
    PR_Int_III=extfea_data['PR_Int_III'], PR_Int_V1=extfea_data['PR_Int_V1'], PR_Int_V2=extfea_data['PR_Int_V2'],
    PR_Int_V3=extfea_data['PR_Int_V3'], PR_Int_V4=extfea_data['PR_Int_V4'], PR_Int_V5=extfea_data['PR_Int_V5'],
    PR_Int_V6=extfea_data['PR_Int_V6'], PR_Int_aVF=extfea_data['PR_Int_aVF'], PR_Int_aVL=extfea_data['PR_Int_aVL'],
    PR_Int_aVR=extfea_data['PR_Int_aVR'], PQ_Int_Global=extfea_data['PQ_Int_Global'], PQ_Int_I=extfea_data['PQ_Int_I'],
    PQ_Int_II=extfea_data['PQ_Int_II'], PQ_Int_III=extfea_data['PQ_Int_III'], PQ_Int_V1=extfea_data['PQ_Int_V1'],
    PQ_Int_V2=extfea_data['PQ_Int_V2'], PQ_Int_V3=extfea_data['PQ_Int_V3'], PQ_Int_V4=extfea_data['PQ_Int_V4'],
    PQ_Int_V5=extfea_data['PQ_Int_V5'], PQ_Int_V6=extfea_data['PQ_Int_V6'], PQ_Int_aVF=extfea_data['PQ_Int_aVF'],
    PQ_Int_aVL=extfea_data['PQ_Int_aVL'], PQ_Int_aVR=extfea_data['PQ_Int_aVR'], QRS_Dur_I=extfea_data['QRS_Dur_I'],
    QRS_Dur_II=extfea_data['QRS_Dur_II'], QRS_Dur_III=extfea_data['QRS_Dur_III'], QRS_Dur_V1=extfea_data['QRS_Dur_V1'],
    QRS_Dur_V2=extfea_data['QRS_Dur_V2'], QRS_Dur_V3=extfea_data['QRS_Dur_V3'], QRS_Dur_V4=extfea_data['QRS_Dur_V4'],
    QRS_Dur_V5=extfea_data['QRS_Dur_V5'], QRS_Dur_V6=extfea_data['QRS_Dur_V6'], QRS_Dur_aVF=extfea_data['QRS_Dur_aVF'],
    QRS_Dur_aVL=extfea_data['QRS_Dur_aVL'], QRS_Dur_aVR=extfea_data['QRS_Dur_aVR'], QT_Int_I=extfea_data['QT_Int_I'],
    QT_Int_II=extfea_data['QT_Int_II'], QT_Int_III=extfea_data['QT_Int_III'], QT_Int_V1=extfea_data['QT_Int_V1'],
    QT_Int_V2=extfea_data['QT_Int_V2'], QT_Int_V3=extfea_data['QT_Int_V3'], QT_Int_V4=extfea_data['QT_Int_V4'],
    QT_Int_V5=extfea_data['QT_Int_V5'], QT_Int_V6=extfea_data['QT_Int_V6'], QT_Int_aVF=extfea_data['QT_Int_aVF'],
    QT_Int_aVL=extfea_data['QT_Int_aVL'], QT_Int_aVR=extfea_data['QT_Int_aVR'], R_Amp_I=extfea_data['R_Amp_I'],
    R_Amp_II=extfea_data['R_Amp_II'], R_Amp_III=extfea_data['R_Amp_III'], R_Amp_V1=extfea_data['R_Amp_V1'],
    R_Amp_V2=extfea_data['R_Amp_V2'], R_Amp_V3=extfea_data['R_Amp_V3'], R_Amp_V4=extfea_data['R_Amp_V4'],
    R_Amp_V5=extfea_data['R_Amp_V5'], R_Amp_V6=extfea_data['R_Amp_V6'], R_Amp_aVF=extfea_data['R_Amp_aVF'],
    R_Amp_aVL=extfea_data['R_Amp_aVL'], R_Amp_aVR=extfea_data['R_Amp_aVR'], Q_Amp_I=extfea_data['Q_Amp_I'],
    Q_Amp_II=extfea_data['Q_Amp_II'], Q_Amp_III=extfea_data['Q_Amp_III'], Q_Amp_V1=extfea_data['Q_Amp_V1'],
    Q_Amp_V2=extfea_data['Q_Amp_V2'], Q_Amp_V3=extfea_data['Q_Amp_V3'], Q_Amp_V4=extfea_data['Q_Amp_V4'],
    Q_Amp_V5=extfea_data['Q_Amp_V5'], Q_Amp_V6=extfea_data['Q_Amp_V6'], Q_Amp_aVF=extfea_data['Q_Amp_aVF'],
    Q_Amp_aVL=extfea_data['Q_Amp_aVL'], Q_Amp_aVR=extfea_data['Q_Amp_aVR'], P_Amp_I=extfea_data['P_Amp_I'],
    P_Amp_II=extfea_data['P_Amp_II'], P_Amp_III=extfea_data['P_Amp_III'], P_Amp_V1=extfea_data['P_Amp_V1'],
    P_Amp_V2=extfea_data['P_Amp_V2'], P_Amp_V3=extfea_data['P_Amp_V3'], P_Amp_V4=extfea_data['P_Amp_V4'],
    P_Amp_V5=extfea_data['P_Amp_V5'], P_Amp_V6=extfea_data['P_Amp_V6'], P_Amp_aVF=extfea_data['P_Amp_aVF'],
    P_Amp_aVL=extfea_data['P_Amp_aVL'], P_Amp_aVR=extfea_data['P_Amp_aVR']
)

# Select records with single superclass label
X_selected = X_new[Y_new['diagnostic_superclass_len'] == 1]
Y_selected = Y_new[Y_new['diagnostic_superclass_len'] == 1]

# Define selected features
selected_features = [
    'RR_Mean_Global', 'ST_Elev_I', 'ST_Elev_II', 'ST_Elev_III', 'ST_Elev_V1', 'ST_Elev_V2', 'ST_Elev_V3', 'ST_Elev_V4',
    'ST_Elev_V5', 'ST_Elev_V6', 'ST_Elev_aVF', 'ST_Elev_aVL', 'ST_Elev_aVR', 'PR_Int_I', 'PR_Int_II', 'PR_Int_III',
    'PR_Int_V1', 'PR_Int_V2', 'PR_Int_V3', 'PR_Int_V4', 'PR_Int_V5', 'PR_Int_V6', 'PR_Int_aVF', 'PR_Int_aVL', 'PR_Int_aVR',
    'PQ_Int_Global', 'PQ_Int_I', 'PQ_Int_II', 'PQ_Int_III', 'PQ_Int_V1', 'PQ_Int_V2', 'PQ_Int_V3', 'PQ_Int_V4', 'PQ_Int_V5',
    'PQ_Int_V6', 'QRS_Dur_I', 'QRS_Dur_II', 'QRS_Dur_III', 'QRS_Dur_V1', 'QRS_Dur_V2', 'QRS_Dur_V3', 'QRS_Dur_V4',
    'QRS_Dur_V5', 'QRS_Dur_V6', 'QRS_Dur_aVF', 'QRS_Dur_aVL', 'QRS_Dur_aVR', 'QT_Int_I', 'QT_Int_II', 'QT_Int_III',
    'QT_Int_V1', 'QT_Int_V2', 'QT_Int_V3', 'QT_Int_V4', 'QT_Int_V5', 'QT_Int_V6', 'QT_Int_aVF', 'QT_Int_aVL', 'QT_Int_aVR',
    'R_Amp_I', 'R_Amp_II', 'R_Amp_III', 'R_Amp_V1', 'R_Amp_V2', 'R_Amp_V3', 'R_Amp_V4', 'R_Amp_V5', 'R_Amp_V6',
    'R_Amp_aVF', 'R_Amp_aVL', 'R_Amp_aVR', 'Q_Amp_I', 'Q_Amp_II', 'Q_Amp_III', 'Q_Amp_V1', 'Q_Amp_V2', 'Q_Amp_V3',
    'Q_Amp_V4', 'Q_Amp_V5', 'Q_Amp_V6', 'Q_Amp_aVF', 'Q_Amp_aVL', 'Q_Amp_aVR', 'P_Amp_I', 'P_Amp_II', 'P_Amp_III',
    'P_Amp_V1', 'P_Amp_V2', 'P_Amp_V3', 'P_Amp_V4', 'P_Amp_V5', 'P_Amp_V6', 'P_Amp_aVF', 'P_Amp_aVL', 'P_Amp_aVR'
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
selector = SelectKBest(score_func=mutual_info_classif, k=50)
X_selected_top = selector.fit_transform(X_features, y_labels)
selected_indices = selector.get_support(indices=True)
top_features = [selected_features[i] for i in selected_indices]
print(f"Selected top 50 features: {top_features}")

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
smote_tomek = SMOTETomek(random_state=42, sampling_strategy='all')
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

# Prepare features and labels
X_balanced_features = balanced_features_df[top_features].values
y_balanced_features = balanced_features_df['labels'].values

# Train-test split
X_train_features, X_test_features, y_train_features, y_test_features = train_test_split(
    X_balanced_features, y_balanced_features, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_features)
X_test_scaled = scaler.transform(X_test_features)

# Reshape for Transformer
X_train_reshaped = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
X_test_reshaped = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)

# Define named labels
named_labels = np.array(['NORM', 'MI', 'STTC', 'CD', 'HYP'])

# Convert labels to one-hot encoding
y_train_onehot = to_categorical(y_train_features, num_classes=5)
y_test_onehot = to_categorical(y_test_features, num_classes=5)

# Compute class weights for loss with capping
class_counts = np.sum(y_train_onehot, axis=0)
class_weights = {i: min(len(y_train_features) / (5 * count), 5.0) for i, count in enumerate(class_counts) if count > 0}

# Set environment for GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Define TransformerBlock
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1, layer_norm_eps=1e-6, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        self.layer_norm_eps = layer_norm_eps

        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads)
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="relu", kernel_regularizer=regularizers.l2(1e-4)),
            layers.Dense(embed_dim, kernel_regularizer=regularizers.l2(1e-4)),
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

# Custom Learning Rate Schedule
class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate, warmup_steps, total_steps):
        super(WarmupCosineDecay, self).__init__()
        self.initial_learning_rate = initial_learning_rate
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.cosine_decay = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=initial_learning_rate,
            first_decay_steps=total_steps // 3,
            t_mul=2.0,
            m_mul=1.0,
            alpha=1e-6
        )

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_lr = (step / self.warmup_steps) * self.initial_learning_rate
        cosine_lr = self.cosine_decay(step - self.warmup_steps)
        return tf.where(step < self.warmup_steps, warmup_lr, cosine_lr)

    def get_config(self):
        return {
            'initial_learning_rate': self.initial_learning_rate,
            'warmup_steps': self.warmup_steps,
            'total_steps': self.total_steps
        }

# Define CNN-Transformer model
def build_cnn_transformer_model():
    input_shape = (50, 1)  # 50 features, 1 channel
    num_classes = 5

    # Hyperparameters
    embed_dim = 256
    num_heads = 8
    ff_dim = 512
    num_cnn_layers = 3
    num_transformer_blocks = 6
    dropout_rate = 0.2
    initial_learning_rate = 1e-3
    weight_decay = 1e-4

    inputs = keras.Input(shape=input_shape)

    # Data augmentation
    x = layers.Lambda(lambda x: x * tf.random.uniform([tf.shape(x)[0], 1, 1], 0.9, 1.1) +
                      tf.random.uniform([tf.shape(x)[0], 1, 1], -0.05, 0.05))(inputs)
    x = layers.GaussianNoise(stddev=0.02)(x, training=True)

    # CNN layers with residual connections
    filters = 64
    shortcut = x
    pooling_count = 0
    for i in range(num_cnn_layers):
        x = layers.Conv1D(
            filters=filters,
            kernel_size=3,
            padding='same',
            activation='relu',
            kernel_initializer='he_normal',
            kernel_regularizer=regularizers.l2(1e-4)
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(pool_size=2, strides=2)(x)
        pooling_count += 1

        # Residual connection every 2 layers
        if i % 2 == 1 and i > 0:
            shortcut = layers.Conv1D(filters, 1, padding='same')(shortcut)
            for _ in range(pooling_count):
                shortcut = layers.MaxPooling1D(pool_size=2, strides=2)(shortcut)
            x = layers.Add()([x, shortcut])
            shortcut = x
            pooling_count = 0

        x = layers.Dropout(dropout_rate)(x, training=True)
        filters = min(filters * 2, 512)

    # Reshape for Transformer
    x = layers.Conv1D(filters=embed_dim, kernel_size=1, padding='same')(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)

    # Transformer blocks
    for _ in range(num_transformer_blocks):
        x_old = x
        transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim, dropout_rate)
        x = transformer_block(x, training=True)
        x = 0.7 * x + 0.3 * x_old

    # Attention pooling
    x = AttentionPooling(embed_dim)(x)
    x = layers.Dropout(dropout_rate)(x, training=True)
    x = layers.Dense(256, activation='relu', kernel_initializer='he_normal',
                     kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x, training=True)
    x = layers.Dense(128, activation='relu', kernel_initializer='he_normal',
                     kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    # Learning rate schedule with warmup and cosine decay
    warmup_steps = 1000
    total_steps = 10000
    lr_schedule = WarmupCosineDecay(
        initial_learning_rate=initial_learning_rate,
        warmup_steps=warmup_steps,
        total_steps=total_steps
    )

    # Compile with AdamW
    optimizer = keras.optimizers.AdamW(
        learning_rate=lr_schedule,
        weight_decay=weight_decay,
        clipnorm=1.0
    )
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy']
    )
    return model

# GPU strategy
if os.environ["CUDA_VISIBLE_DEVICES"].count(',') == 0:
    gpu_strategy = tf.distribute.get_strategy()
    print('Single strategy')
else:
    gpu_strategy = tf.distribute.MirroredStrategy()
    print('Multiple strategy')

# Mixed precision disabled
print('Mixed precision disabled')

# Custom callback for weighted F1-score
class WeightedF1Callback(Callback):
    def __init__(self, val_X, val_y):
        super(WeightedF1Callback, self).__init__()
        self.val_X = val_X
        self.val_y = val_y

    def on_epoch_end(self, epoch, logs=None):
        try:
            y_pred = self.model.predict(self.val_X, verbose=0)
            y_pred_labels = np.argmax(y_pred, axis=1)
            y_true_labels = np.argmax(self.val_y, axis=1)
            f1 = f1_score(y_true_labels, y_pred_labels, average='weighted')
            logs['val_weighted_f1_score'] = f1
            print(f" - val_weighted_f1_score: {f1:.4f}")
        except Exception as e:
            print(f"Error computing F1 score: {e}")

# Test-time augmentation with error handling
def tta_predict(model, X, n_augmentations=5, batch_size=32):
    predictions = []
    try:
        for _ in range(n_augmentations):
            X_aug = X * np.random.uniform(0.9, 1.1, size=(X.shape[0], 1, 1)) + np.random.uniform(-0.05, 0.05, size=(X.shape[0], 1, 1))
            X_aug += np.random.normal(0, 0.02, size=X.shape)
            preds = model.predict(X_aug, batch_size=batch_size, verbose=0)
            if np.any(np.isnan(preds)) or np.any(np.isinf(preds)):
                raise ValueError("NaN or Inf detected in predictions")
            predictions.append(preds)
        mean_preds = np.mean(predictions, axis=0)
        return mean_preds
    except Exception as e:
        print(f"Error in TTA prediction: {e}")
        return None

# Custom callback to save history
class HistoryCheckpoint(Callback):
    def on_epoch_end(self, epoch, logs=None):
        try:
            with open(f'history_epoch_{epoch}.json', 'w') as f:
                json.dump(self.model.history.history, f)
        except Exception as e:
            print(f"Error saving history for epoch {epoch}: {e}")

# Initialize weighted F1 callback
weighted_f1_callback = WeightedF1Callback(X_test_reshaped, y_test_onehot)

# Check for existing checkpoints and history
saved_models = glob.glob('model_epoch_*.h5')
initial_epoch = 0
history_dict = {}

if saved_models:
    try:
        latest_model = max(saved_models, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        epoch_num = int(latest_model.split('_')[-1].split('.')[0])
        if os.path.exists(latest_model) and os.path.getsize(latest_model) > 0:
            model = keras.models.load_model(
                latest_model,
                custom_objects={
                    'TransformerBlock': TransformerBlock,
                    'AttentionPooling': AttentionPooling,
                    'WarmupCosineDecay': WarmupCosineDecay
                }
            )
            initial_epoch = epoch_num + 1
            print(f"Resuming training from epoch {initial_epoch}")
            # Load history if exists
            history_file = f'history_epoch_{epoch_num}.json'
            if os.path.exists(history_file):
                with open(history_file, 'r') as f:
                    history_dict = json.load(f)
                print("Loaded previous training history")
        else:
            print(f"Checkpoint {latest_model} is invalid. Starting from scratch.")
            with gpu_strategy.scope():
                model = build_cnn_transformer_model()
    except Exception as e:
        print(f"Error loading checkpoint: {e}. Starting from scratch.")
        with gpu_strategy.scope():
            model = build_cnn_transformer_model()
else:
    print("No checkpoints found. Starting from scratch.")
    with gpu_strategy.scope():
        model = build_cnn_transformer_model()

# Compile model
model.compile(
    optimizer=model.optimizer if saved_models else keras.optimizers.AdamW(
        learning_rate=WarmupCosineDecay(1e-3, 1000, 10000),
        weight_decay=1e-4,
        clipnorm=1.0
    ),
    loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)

# Define callbacks
checkpoint = ModelCheckpoint(
    'best_model.h5',
    monitor='val_weighted_f1_score',
    save_best_only=True,
    mode='max',
    save_weights_only=False
)
periodic_checkpoint = ModelCheckpoint(
    'model_epoch_{epoch}.h5',
    save_weights_only=False,
    save_freq='epoch'
)
history_checkpoint = HistoryCheckpoint()
early_stopping = EarlyStopping(
    monitor='val_weighted_f1_score',
    patience=30,
    restore_best_weights=True,
    mode='max'
)

# Train model
history = model.fit(
    X_train_reshaped,
    y_train_onehot,
    validation_data=(X_test_reshaped, y_test_onehot),
    epochs=100,
    batch_size=64,
    callbacks=[early_stopping, checkpoint, periodic_checkpoint, history_checkpoint, weighted_f1_callback],
    class_weight=class_weights,
    verbose=1,
    initial_epoch=initial_epoch
)

# Merge history if resuming
if history_dict:
    for key in history.history:
        history.history[key] = history_dict.get(key, []) + history.history[key]

# Evaluate model
Y_pred = tta_predict(model, X_test_reshaped)
if Y_pred is None:
    raise ValueError("Prediction failed; cannot compute metrics")
Y_pred_labels = np.argmax(Y_pred, axis=1)
y_true_labels = np.argmax(y_test_onehot, axis=1)

# Confusion matrix
try:
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
    print("Test Set Results:")
    print("Confusion Matrix (Counts):")
    print(cf_matrix)
    print("Confusion Matrix (Percentage):")
    print(np.round(cf_matrix_percentage, 4))
    print("Test Sensitivity per Class:", test_sensitivity)
    print("Test Specificity per Class:", test_specificity)
    print("\nClassification Report:")
    print(classification_report(y_true_labels, Y_pred_labels, target_names=named_labels))
except Exception as e:
    print(f"Error computing metrics: {e}")

# Plot training and validation metrics
plt.figure(figsize=(12, 4))
metrics = {'loss': 'Loss', 'accuracy': 'Accuracy', 'val_weighted_f1_score': 'F1 Score'}
for i, (key, title) in enumerate([('loss', 'Loss'), ('accuracy', 'Accuracy'), ('val_weighted_f1_score', 'F1 Score')]):
    plt.subplot(1, 3, i+1)
    if key in history.history:
        plt.plot(history.history[key], label=f'Training {title}')
    if f'val_{key}' in history.history:
        plt.plot(history.history[f'val_{key}'], label=f'Validation {title}')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(title)
    plt.legend()
plt.tight_layout()
try:
    plt.savefig('metrics.png')
    plt.close()
except Exception as e:
    print(f"Error plotting metrics: {e}")

# Plot test set confusion matrix
plt.figure(figsize=(8, 6))
try:
    sns.heatmap(cf_matrix_percentage, annot=True, fmt='.2f', cmap='Blues', xticklabels=named_labels, yticklabels=named_labels)
    plt.title('Confusion Matrix - Test Set (Percentage)')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.savefig('test_confusion_matrix.png')
    plt.close()
except Exception as e:
    print(f"Error plotting confusion matrix: {e}")

# Save model
try:
    model.save('qrs_transformer_optimized.h5')
    print("Model saved successfully")
except Exception as e:
    print(f"Error saving model: {e}")

# Save test data and scaler
try:
    with open('test_data.npy', 'wb') as f:
        np.save(f, X_test_reshaped)
    with open('test_labels_onehot.npy', 'wb') as f:
        np.save(f, y_test_onehot)
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open('top_features.pkl', 'wb') as f:
        pickle.dump(top_features, f)
    print("Data files saved successfully")
except Exception as e:
    print(f"Error saving data files: {e}")

if __name__ == "__main__":
    print("QRS Transformer classification completed successfully.")
