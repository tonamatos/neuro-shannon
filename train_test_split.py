import os
import shutil
from sklearn.model_selection import train_test_split

# Function to split dataset into train and test sets
def split_dataset(directory, test_size=0.2):
    # List all files in the directory
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    
    # Split the files into train and test sets
    train_files, test_files = train_test_split(files, test_size=test_size, random_state=42)
    
    # Create train and test directories if they don't exist
    train_dir = os.path.join(directory, 'train')
    test_dir = os.path.join(directory, 'test')
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Move files to train and test directories
    for file in train_files:
        shutil.move(os.path.join(directory, file), os.path.join(train_dir, file))
    
    for file in test_files:
        shutil.move(os.path.join(directory, file), os.path.join(test_dir, file))

# Usage
directory = './data/mis_sat'  # Replace with your directory path
split_dataset(directory)
