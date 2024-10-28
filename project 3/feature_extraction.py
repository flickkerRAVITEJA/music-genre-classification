import os
import librosa
import pandas as pd

# Define the function to extract audio features
def extract_features(file_path):
    """Extract MFCC features from an audio file."""
    print(f"Extracting features from: {file_path}")
    try:
        y, sr = librosa.load(file_path, mono=True, sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        return mfccs.mean(axis=1)  # Return the mean of the MFCCs
    except Exception as e:
        print(f"Error extracting features from {file_path}: {e}")
        return None

# Set paths and initialize variables
DATASET_PATH = r"C:\Users\mh12k\OneDrive\Desktop\music_dataset"
CSV_OUTPUT = "features.csv"
data = []

# Verify the dataset path exists
if not os.path.exists(DATASET_PATH):
    print(f"Error: The dataset path {DATASET_PATH} does not exist.")
    exit()

print(f"Dataset path exists: {DATASET_PATH}")

# List genres (subfolders) and process each one
genres = os.listdir(DATASET_PATH)
print(f"Found genres: {genres}")

for genre in genres:
    genre_folder = os.path.join(DATASET_PATH, genre)

    if not os.path.isdir(genre_folder):
        print(f"Skipping {genre_folder} - not a folder.")
        continue

    print(f"Processing genre: {genre}")

    for filename in os.listdir(genre_folder):
        file_path = os.path.join(genre_folder, filename)

        if filename.endswith(".wav"):
            features = extract_features(file_path)  # Call the function
            if features is not None:
                data.append([genre] + list(features))  # Store genre + features
                print(f"Processed {file_path}")
        else:
            print(f"Skipping non-wav file: {filename}")

# Check if any data was collected
if len(data) == 0:
    print("Error: No data to save. Check your dataset path and files.")
else:
    # Save extracted features to CSV
    df = pd.DataFrame(data, columns=["genre"] + [f"feature_{i}" for i in range(len(data[0]) - 1)])
    df.to_csv(CSV_OUTPUT, index=False)
    print(f"Features saved to {CSV_OUTPUT}")
