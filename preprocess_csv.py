import string
import pandas as pd
import numpy as np
import pickle
from gensim.models import KeyedVectors
import torch

# Load Pre-trained Word2Vec Model
def load_word2vec_model(path):
    """
    Load pre-trained Word2Vec model.
    Args:
        path (str): Path to the pre-trained Word2Vec binary file.
    Returns:
        KeyedVectors: Loaded Word2Vec model.
    """
    print("Loading pre-trained Word2Vec model...")
    word2vec_model = KeyedVectors.load_word2vec_format(path, binary=True)
    print("Model loaded successfully!")
    return word2vec_model

# Clean Lyrics
def clean_lyrics(lyrics):
    """
    Cleans the lyrics by removing special characters, punctuation, etc.
    """
    lyrics = lyrics.lower()
    lyrics = lyrics.translate(str.maketrans("", "", string.punctuation))
    lyrics = lyrics.replace("\n", " ").replace("\r", " ")
    return lyrics

# Tokenize Lyrics
def tokenize_lyrics(lyrics):
    """
    Tokenizes the cleaned lyrics into words using split and cleans extra spaces.
    """
    splet = lyrics.split()
    for i in range(len(splet)):
        splet[i] = splet[i].replace(' ', '')
    return splet

# Vectorize Tokens
def vectorize_lyrics(tokens, word2vec_model):
    """
    Converts tokens into a matrix of word embeddings using the pre-trained Word2Vec model.
    Returns:
        torch.Tensor: Tensor of shape (num_words, embedding_dim).
        list: List of tokens that are in the Word2Vec model.
    """
    valid_tokens = [token for token in tokens if token in word2vec_model]
    vectors = [torch.tensor(word2vec_model[token], dtype=torch.float32) for token in valid_tokens]
    return (torch.stack(vectors) if vectors else torch.empty((0, 300), dtype=torch.float32), valid_tokens)

# Preprocess Lyrics and Save as Pickle
def preprocess_lyrics(csv_path, word2vec_path, output_pickle_path):
    """
    Preprocess the lyrics dataset and save it as a single pickle file.
    Args:
        csv_path (str): Path to the CSV file containing the dataset.
        word2vec_path (str): Path to the pre-trained Word2Vec binary file.
        output_pickle_path (str): Path to save the pickle file.
    """
    # Load dataset
    train_df = pd.read_csv(csv_path, header=None)

    # Add an index column for song IDs
    train_df.insert(0, 'song_id', range(len(train_df)))

    # Load pre-trained Word2Vec model
    word2vec_model = load_word2vec_model(word2vec_path)

    song_data = {}
    for idx, row in train_df.iterrows():
        song_id = row['song_id']  # Use the new song ID column
        song_lyrics = row[2]  # Original lyrics column

        # Preprocess lyrics
        clean_text = clean_lyrics(song_lyrics)
        tokens = tokenize_lyrics(clean_text)
        embedding_matrix, valid_tokens = vectorize_lyrics(tokens, word2vec_model)

        # Save song data
        song_data[song_id] = {
            "lyrics_vectors": embedding_matrix,
            "length": embedding_matrix.shape[0],       # Add song length
            "cleaned_lyrics": ' '.join(valid_tokens)  # Save only valid tokens as text
        }

        # Update the CSV with cleaned lyrics
        train_df.at[idx, 2] = ' '.join(valid_tokens)

    # Save the updated CSV
    updated_csv_path = csv_path.replace(".csv", "_updated.csv")
    train_df.to_csv(updated_csv_path, index=False)
    print(f"Updated CSV saved to {updated_csv_path}")

    # Save the entire dataset to a single pickle file
    with open(output_pickle_path, 'wb') as f:
        pickle.dump(song_data, f)

    print(f"Processed data saved to {output_pickle_path}")

# Paths (update these with your local file paths)
train_csv_path = "lyrics_train_set.csv"  # Path to your train CSV file
test_csv_path = "lyrics_test_set.csv"  # Path to your test CSV file

word2vec_path = "GoogleNews-vectors-negative300.bin"  # Path to pre-trained Word2Vec model

train_output_pickle_path = "train_song_dataset_with_w2v.pkl"  # Output pickle file path
test_output_pickle_path = "test_song_dataset_with_w2v.pkl"  # Output pickle file path

# Run Preprocessing
preprocess_lyrics(train_csv_path, word2vec_path, train_output_pickle_path)
preprocess_lyrics(test_csv_path, word2vec_path, test_output_pickle_path)
