import numpy as np
import scipy.stats as stats
from scipy.signal import welch
import pywt
import pandas as pd

def compute_hjorth_parameters(x: np.ndarray):
    """Computes Hjorth parameters: Activity, Mobility, and Complexity."""
    dx = np.diff(x)
    ddx = np.diff(dx)
    
    var_x = np.var(x)
    var_dx = np.var(dx)
    var_ddx = np.var(ddx)
    
    activity = var_x
    # Prevent division by zero
    if var_x == 0 or var_dx == 0:
        return activity, 0, 0
        
    mobility = np.sqrt(var_dx / var_x)
    mobility_dx = np.sqrt(var_ddx / var_dx)
    complexity = mobility_dx / mobility
    
    return activity, mobility, complexity

def compute_katz_fd(x: np.ndarray):
    """Computes Katz's Fractal Dimension."""
    n = len(x) - 1
    if n < 1:
        return 0
    
    L = np.sum(np.abs(np.diff(x)))
    # Distance from first point
    d = np.max(np.abs(x - x[0]))
    
    if L == 0 or d == 0:
        return 0
        
    katz_fd = np.log10(n) / (np.log10(d / L) + np.log10(n))
    return katz_fd

def extract_features_1d(x: np.ndarray, fs: float = 128.0) -> dict:
    """
    Extracts the 20 features described in Table 1 of Hag et al. (2023)
    from a 1D EEG signal segment.
    """
    features = {}
    
    # ------------------ TIME DOMAIN ------------------
    features['line_length'] = np.sum(np.abs(np.diff(x)))
    features['kurtosis'] = stats.kurtosis(x)
    features['peak_to_peak'] = np.ptp(x)
    features['skewness'] = stats.skew(x)
    
    act, mob, comp = compute_hjorth_parameters(x)
    features['hjorth_activity'] = act
    features['hjorth_mobility'] = mob
    features['hjorth_complexity'] = comp
    
    # ------------------ FREQUENCY DOMAIN ------------------
    # Calculate Power Spectral Density (PSD) using Welch's method
    freqs, psd = welch(x, fs=fs, nperseg=min(256, len(x)))
    
    total_power = np.sum(psd)
    if total_power == 0:
        total_power = 1e-10 # Prevent div by zero
        
    bands = {
        'theta': (4, 8),
        'alpha': (8, 12),
        'sigma': (12, 15),
        'low_beta': (15, 20),
        'high_beta': (20, 30)
    }
    
    for band_name, (low, high) in bands.items():
        idx = np.logical_and(freqs >= low, freqs <= high)
        band_power = np.sum(psd[idx])
        features[f'rp_{band_name}'] = band_power / total_power

    # ------------------ TIME-FREQUENCY DOMAIN ------------------
    # 1. Wavelet Energy (db4, 6 levels)
    # PyWavelets wavedec returns [cA6, cD6, cD5, cD4, cD3, cD2, cD1]
    coeffs = pywt.wavedec(x, 'db4', level=6)
    
    # Extract energy of the 6 detail coefficients (D1 to D6)
    # coeffs[1] is cD6, coeffs[6] is cD1
    for i in range(1, 7):
        energy = np.sum(coeffs[i]**2)
        # reverse index so 1 is D1, 6 is D6
        level_idx = 7 - i 
        features[f'wavelet_energy_d{level_idx}'] = energy
        
    # 2. Spectral Entropy
    psd_norm = psd / total_power
    psd_norm = psd_norm[psd_norm > 0] # Filter zeros for log
    features['spectral_entropy'] = -np.sum(psd_norm * np.log2(psd_norm))
    
    # 3. Katz's Fractal Dimension
    features['katz_fd'] = compute_katz_fd(x)
    
    return features

def process_eeg_dataset(data_matrix: np.ndarray, channels: list = None) -> pd.DataFrame:
    """
    Processes a matrix of EEG data of shape (n_trials, n_channels, n_samples)
    and extracts features for each channel.
    """
    n_trials, n_channels, n_samples = data_matrix.shape
    if channels is None:
        channels = [f'CH{i+1}' for i in range(n_channels)]
        
    all_rows = []
    
    for trial_idx in range(n_trials):
        row = {'trial_idx': trial_idx}
        
        for ch_idx in range(n_channels):
            channel_name = channels[ch_idx]
            signal = data_matrix[trial_idx, ch_idx, :]
            
            # Extract 20 features for this channel
            ch_features = extract_features_1d(signal, fs=128.0)
            
            # Prefix the feature name with the channel name
            for feat_name, val in ch_features.items():
                row[f'{channel_name}_{feat_name}'] = val
                
        all_rows.append(row)
        
    return pd.DataFrame(all_rows)

if __name__ == "__main__":
    print("Testing Feature Extraction on mock EEG data...")
    # Mock data: 2 trials, 32 channels, 480 samples (4-second segment at 128Hz)
    mock_data = np.random.randn(2, 32, 480)
    
    # Common DEAP 32 channels (Standard 10-20 system)
    deap_channels = ['Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 
                     'CP5', 'CP1', 'P3', 'P7', 'PO3', 'O1', 'Oz', 'Pz', 
                     'Fp2', 'AF4', 'Fz', 'F4', 'F8', 'FC6', 'FC2', 'Cz', 
                     'C4', 'T8', 'CP6', 'CP2', 'P4', 'P8', 'PO4', 'O2']
                     
    df_features = process_eeg_dataset(mock_data, deap_channels)
    print(f"Extracted feature DataFrame shape: {df_features.shape}")
    print(f"Number of expected features: 32 channels * 20 features = {32 * 20}")
    print("Columns (First 25):")
    print(df_features.columns[1:26].tolist())
