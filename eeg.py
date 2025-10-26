import os
import glob
import numpy as np
import pandas as pd
from scipy.fft import fft
from scipy.signal import iirfilter, filtfilt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# ----------------- CONFIG -----------------
DATA_DIR = './data/edf'       
LABELS = {                    
    'subject1.edf': 1,        
    'subject2.edf': 0,        
    'subject3.edf':0,
    'subject4.edf':1,
    'subject5.edf':1,
    'subject6.edf':1,
    'subject7.edf':1,
    'subject8.edf':1,
    'subject17.edf':0,
    'subject18.edf':0,
    'subject19.edf':0,
    'subject20.edf':0,
    'subject9.edf':1,
    'subject11.edf':0,
    'subject12.edf':0,
    'subject13.edf':0,
    'subject14.edf':0,
    'subject15.edf':0,
    'subject16.edf':0,
    'subject10.edf':1,
    'subject21.edf':0,
    'subject22.edf':0,
}
FS = 128
SEG_SECONDS = 30
CHANNEL_PRIORITY = ['O1','O2','T7','T8','P7','P8']

# ----------------- UTILS -----------------
def bandpass_filter(sig, low=0.2, high=45, fs=FS, order=4):
    nyq = 0.5*fs
    b, a = iirfilter(order, [low/nyq, high/nyq], btype='band', ftype='butter')
    return filtfilt(b, a, sig)

def fft_power(sig):
    N = len(sig)
    psd = np.abs(fft(sig))**2
    freqs = np.fft.fftfreq(N, 1/FS)
    return freqs[:N//2], psd[:N//2]

def spectral_entropy(sig):
    freqs, psd = fft_power(sig)
    psd_norm = psd / (np.sum(psd)+1e-12)
    return -np.sum(psd_norm * np.log2(psd_norm+1e-12)) / np.log2(len(psd_norm)+1e-12)

def kolmogorov_complexity(sig):
    med = np.median(sig)
    binary = (sig > med).astype(int)
    return np.sum(binary[1:] != binary[:-1]) / (len(binary)-1+1e-12)

def segment_signal(sig, seg_sec=SEG_SECONDS):
    seg_len = seg_sec*FS
    for i in range(len(sig)//seg_len):
        yield sig[i*seg_len:(i+1)*seg_len]

# ----------------- LOAD EDF -----------------
def load_edf_file(fp):
    import mne
    raw = mne.io.read_raw_edf(fp, preload=True, verbose=False)
    chs = raw.ch_names
    for ch in CHANNEL_PRIORITY:
        if ch in chs:
            return raw.get_data(picks=[ch])[0], ch
    return raw.get_data(picks=[chs[0]])[0], chs[0]

# ----------------- BUILD DATASET -----------------
def build_dataset():
    records = []
    edf_files = glob.glob(os.path.join(DATA_DIR,'*.edf')) + glob.glob(os.path.join(DATA_DIR,'*.EDF'))
    for fp in edf_files:
        basename = os.path.basename(fp)
        if basename not in LABELS:
            print(f"Skipping {basename}, no label assigned")
            continue
        label = LABELS[basename]
        try:
            sig, ch = load_edf_file(fp)
        except:
            continue
        sig = bandpass_filter(sig)
        for idx, seg in enumerate(segment_signal(sig)):
            feats = {
                'spectral_entropy': spectral_entropy(seg),
                'kolmogorov_complexity': kolmogorov_complexity(seg),
                'file': basename,
                'segment': idx,
                'channel': ch,
                'label': label
            }
            records.append(feats)
    df = pd.DataFrame(records)
    return df

# ----------------- METRICS -----------------
def compute_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp + 1e-12)
    sensitivity = tp / (tp + fn + 1e-12)
    precision = tp / (tp + fp + 1e-12)
    fpr = fp / (fp + tn + 1e-12)
    fnr = fn / (fn + tp + 1e-12)
    f1 = 2*tp / (2*tp + fp + fn + 1e-12)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-12)
    return {'specificity': specificity, 'sensitivity': sensitivity, 'precision': precision,
            'FPR': fpr, 'FNR': fnr, 'F1': f1, 'accuracy': accuracy}

# ----------------- TRAIN & EVALUATE -----------------
def train_evaluate(df):
    X = df[['spectral_entropy','kolmogorov_complexity']].values
    y = df['label'].values
    strat = y if len(np.unique(y))>1 else None
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,stratify=strat,random_state=42)
    
    models = {
        'SVM_RBF': SVC(kernel='rbf', C=1, gamma='scale'),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'ANN': MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000, activation='relu', solver='adam'),
        'SVM_POLY': SVC(kernel='poly', degree=3, C=1),
        'SVM_SIGMOID': SVC(kernel='sigmoid', C=1)
    }
    
    metrics_results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics_results[name] = compute_metrics(y_test, y_pred)
    
    # ----------------- PLOTTING BAR GRAPHS -----------------
    metrics_to_plot = ['specificity', 'sensitivity', 'precision']
    fig, axs = plt.subplots(1, len(metrics_to_plot), figsize=(15,5))
    for i, metric in enumerate(metrics_to_plot):
        vals = [metrics_results[alg][metric] for alg in models.keys()]
        axs[i].bar(models.keys(), vals, color='skyblue')
        axs[i].set_ylim(0,1.05)
        axs[i].set_ylabel(metric)
        axs[i].set_title(metric)
        axs[i].set_xticklabels(models.keys(), rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    
    # ----------------- CREATE METRICS TABLE -----------------
    df_metrics = pd.DataFrame(metrics_results).T
    best_algos = df_metrics.idxmax()
    df_metrics['best_algorithm'] = df_metrics.idxmax(axis=0)
    
    print("\n--- METRICS TABLE ---")
    print(df_metrics)
    print("\n--- BEST ALGORITHM FOR EACH METRIC ---")
    print(best_algos)
    
    return metrics_results, df_metrics, best_algos

# ----------------- RUN -----------------
def run():
    df = build_dataset()
    if df.empty:
        print("No data extracted. Check EDF files and labels")
        return
    df.to_csv('features.csv', index=False)
    results, df_metrics, best_algos = train_evaluate(df)
    return results, df_metrics, best_algos

if __name__=='__main__':
    run()
