import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Any

def load_deap_file(filepath: str) -> Dict[str, np.ndarray]:
    """
    Loads a single DEAP dataset .dat file.
    
    Args:
        filepath (str): Path to the .dat file.
        
    Returns:
        Dict[str, np.ndarray]: Dictionary containing 'data' and 'labels' arrays.
    """
    with open(filepath, 'rb') as f:
        # DEAP dataset files are Python 2 pickle files, hence encoding='latin1'
        data = pickle.load(f, encoding='latin1')
    return data

def process_deap_targets(labels: np.ndarray) -> pd.DataFrame:
    """
    Processes the raw DEAP labels into stress targets.
    
    Args:
        labels (np.ndarray): The raw labels array of shape (n_trials, 4).
                             Index 0: Valence, Index 1: Arousal, 
                             Index 2: Dominance, Index 3: Liking.
                             
    Returns:
        pd.DataFrame: DataFrame containing Valence, Arousal, Continuous Stress,
                      and Discrete Stress values.
    """
    # Extract Valence (index 0) and Arousal (index 1)
    valence = labels[:, 0]
    arousal = labels[:, 1]
    
    # Calculate Continuous Stress Index: y = (9 - Valence) + Arousal
    continuous_stress = (9 - valence) + arousal
    
    # Calculate Discrete Stress Label (Binary Thresholding)
    # 1 (STRESS) if Valence < 5.0 and Arousal > 5.0, else 0 (CALM)
    discrete_stress = np.where((valence < 5.0) & (arousal > 5.0), 1, 0)
    
    # Create DataFrame for the targets
    targets_df = pd.DataFrame({
        'valence': valence,
        'arousal': arousal,
        'continuous_stress': continuous_stress,
        'discrete_stress': discrete_stress
    })
    
    return targets_df

def prepare_pipeline_data(data_dir: str, file_pattern: str = ".dat") -> pd.DataFrame:
    """
    Loads all DEAP files in the data directory and prepares them for the pipeline.
    
    Args:
        data_dir (str): Path to the directory containing .dat files.
        file_pattern (str): File extension to match.
        
    Returns:
        pd.DataFrame: A structured DataFrame containing trial IDs, stress targets, 
                      and raw EEG data arrays.
    """
    all_data_records: List[Dict[str, Any]] = []
    
    # Ensure data directory exists
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
        
    # Iterate through files
    for filename in sorted(os.listdir(data_dir)):
        if filename.endswith(file_pattern):
            participant_id = filename.split('.')[0]
            filepath = os.path.join(data_dir, filename)
            
            # Load data
            subject_data = load_deap_file(filepath)
            
            # Extract raw EEG data and labels
            # raw_eeg_data shape: (40, 40, 8064) -> (n_trials, n_channels, n_samples)
            raw_eeg_data = subject_data['data']  
            labels = subject_data['labels']      
            
            # Process targets for this participant
            targets_df = process_deap_targets(labels)
            
            # Assemble records for each trial
            n_trials = raw_eeg_data.shape[0]
            for trial_idx in range(n_trials):
                record = {
                    'participant_id': participant_id,
                    'trial_id': f"{participant_id}_trial_{trial_idx+1}",
                    'valence': targets_df.iloc[trial_idx]['valence'],
                    'arousal': targets_df.iloc[trial_idx]['arousal'],
                    'continuous_stress': targets_df.iloc[trial_idx]['continuous_stress'],
                    'discrete_stress': int(targets_df.iloc[trial_idx]['discrete_stress']),
                    'raw_eeg_data': raw_eeg_data[trial_idx]
                }
                all_data_records.append(record)
                
    # Create final DataFrame to pass to mne and feature extraction
    return pd.DataFrame(all_data_records)

if __name__ == "__main__":
    # ---------------------------------------------------------
    # Test Execution Block
    # ---------------------------------------------------------
    print("Initializing test execution for DEAP target processing pipeline...")
    
    # Create a mock data directory and file for testing if they don't exist
    test_dir = "data"
    os.makedirs(test_dir, exist_ok=True)
    test_file_path = os.path.join(test_dir, "s01.dat")
    
    # Generate mock data to simulate a single DEAP dataset file (`s01.dat`)
    if not os.path.exists(test_file_path):
        print(f"Creating mock DEAP data file at '{test_file_path}' for testing...")
        # DEAP format: 40 trials, 40 channels (32 EEG + 8 peripheral), 8064 samples
        mock_data = np.random.randn(40, 40, 8064) 
        # Labels: 40 trials, 4 dimensions (Valence, Arousal, Dominance, Liking) in range [1, 9]
        mock_labels = np.random.uniform(1.0, 9.0, (40, 4)) 
        
        mock_dict = {'data': mock_data, 'labels': mock_labels}
        
        with open(test_file_path, 'wb') as f:
            pickle.dump(mock_dict, f, protocol=2)
            
    print(f"Loading and processing data from '{test_dir}/'...")
    try:
        # Run pipeline function
        pipeline_df = prepare_pipeline_data(data_dir=test_dir)
        
        print("\n=== Processing Successful! Pipeline DataFrame Summary ===")
        print(f"Total shape: {pipeline_df.shape} (trials, features)")
        
        print("\nTarget Variables (First 5 Trials):")
        print(pipeline_df[['trial_id', 'valence', 'arousal', 'continuous_stress', 'discrete_stress']].head())
        
        print("\nRaw EEG Data Array Shape for the First Trial:")
        # Should be (40, 8064) representing channels x samples
        print(pipeline_df['raw_eeg_data'].iloc[0].shape) 
        
        print("\nDiscrete Stress Class Distribution:")
        print(pipeline_df['discrete_stress'].value_counts().to_string())
        
    except Exception as e:
        print(f"Error during execution: {e}")
