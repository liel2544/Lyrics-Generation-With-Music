import os
import pickle
import pretty_midi
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd


def get_instructor_beats(midi_file, idx_word, mean_time):
    """
    Extract number of beats and number of active instruments within a word's time window.
    """
    beats, instructor = 0, 0
    start_time = idx_word * mean_time
    end_time = start_time + mean_time

    for beat in midi_file.get_beats():
        if start_time <= beat <= end_time:
            beats += 1

    for instrument in midi_file.instruments:
        for note in instrument.notes:
            if start_time <= note.start <= end_time:
                instructor += 1

    return torch.tensor([beats, instructor], dtype=torch.float32)


def extract_piano_feature(midi_file, idx_word, num_words):
    """
    Extract piano roll features for a word's time window.
    """
    try:
        piano_roll = midi_file.get_piano_roll()
        notes_per_word = int(piano_roll.shape[1] / num_words)

        start_index = idx_word * notes_per_word
        end_index = start_index + notes_per_word

        piano_for_lyric = piano_roll[:, start_index:end_index].transpose()
        piano_sum = torch.tensor(piano_for_lyric.sum(axis=0), dtype=torch.float32)

        return piano_sum
    except Exception:
        return torch.zeros(128, dtype=torch.float32)


def extract_mellidies_feature_methods(midi_file, num_words):
    """
    Extract melody features for each word in a song using beats and instruments.
    """
    midi_file.remove_invalid_notes()
    mean_time_word = midi_file.get_end_time() / num_words  # Calculate mean time per word
    song_feature = []

    for index_word in range(num_words):
        beats_and_instruments = get_instructor_beats(midi_file, index_word, mean_time_word)
        piano_features = extract_piano_feature(midi_file, index_word, mean_time_word)
        features = torch.cat((beats_and_instruments, piano_features))
        song_feature.append(features)

    return torch.stack(song_feature)


def extract_midi_features(midi_path, num_words):
    """
    Extract combined MIDI features for a song.
    """
    try:
        midi_file = pretty_midi.PrettyMIDI(midi_path)
        features = extract_mellidies_feature_methods(midi_file, num_words)
        return torch.tensor(features, dtype=torch.float32)

    except Exception as e:
        print(f"Error processing MIDI file {midi_path}: {e}")
        unified_vector_length = 2  # Adjust based on features combined
        return torch.zeros((num_words, unified_vector_length), dtype=torch.float32)


def process_set(csv_path, pickle_path, midi_dir, output_pickle_path):
    """
    Process a dataset (train/test) and add combined MIDI features to the pickle file.
    """
    # Load CSV
    df = pd.read_csv(csv_path)

    # Load existing pickle file
    with open(pickle_path, 'rb') as f:
        song_data = pickle.load(f)

    # Process each song
    for _, row in tqdm(df.iterrows(), total=len(df)):
        song_id = row['song_id']
        artist = row['0'].replace(' ', '_')
        title = row['1'].replace(' ', '_')

        midi_file = f"{artist}_-_{title}.mid"
        midi_path = os.path.join(midi_dir, midi_file)

        if not os.path.exists(midi_path):
            print(f"MIDI file not found: {midi_path}")
            continue

        num_words = song_data[song_id]['length']
        midi_features = extract_midi_features(midi_path, num_words)

        song_data[song_id]['midi_features'] = midi_features

    with open(output_pickle_path, 'wb') as f:
        pickle.dump(song_data, f)

    print(f"Updated pickle file saved to {output_pickle_path}")


# Paths (update these with your local paths)
train_csv_path = "lyrics_train_set_updated.csv"  # Updated train CSV file
test_csv_path = "lyrics_test_set_updated.csv"  # Updated test CSV file
train_pickle_path = "train_song_dataset_with_w2v.pkl"  # Existing train pickle file
test_pickle_path = "test_song_dataset_with_w2v.pkl"  # Existing test pickle file
midi_dir = "midi_files"  # Directory containing MIDI files
train_output_pickle_path = "train_song_dataset_with_w2v_and_midi.pkl"  # Updated train pickle file
test_output_pickle_path = "test_song_dataset_with_w2v_and_midi.pkl"  # Updated test pickle file

# Process train and test sets
process_set(train_csv_path, train_pickle_path, midi_dir, train_output_pickle_path)
process_set(test_csv_path, test_pickle_path, midi_dir, test_output_pickle_path)
