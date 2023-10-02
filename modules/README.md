# Important modules include:

1. leadfields
   
Loads the head model files and saved list of subjects

2. augmentation
   
Generates N times more data based on given n_calibration_trials per subject, by shifting the source dipole at the given dipole_shift distance. Resulted data contains  n_calibration_trials*N

3. filters
   
Band-pass filtering and laplacian filtering of the raw generated data

4. create_datasets
   
Selects data for training and testing during classification

5. classify
   
Performs LDA and NN classification

6. visualize_results
   
Plots classification results for each target subject before and after classification
