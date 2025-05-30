import numpy as np

# read npy file

# Specify the path to your .npy file
file_path = 'training_data/depth_data/sample_0001_20250530_121142_117_depth.npy'

# Load the .npy file
data = np.load(file_path)

# Print the contents of the .npy file
print(data)


