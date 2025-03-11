import torch
from torch.utils.data import Dataset, DataLoader
import pickle

class LyricsMIDIDataset(Dataset):
    """
    Custom Dataset to handle lyrics and MIDI features for each song.

    Each sample consists of the current word's embedding concatenated with its
    MIDI features as input and the next word's embedding as the target.
    """

    def __init__(self, pickle_path):
        with open(pickle_path, 'rb') as f:
            self.song_data = pickle.load(f)

    def __len__(self):
        return len(self.song_data)

    def __getitem__(self, idx):
        song = self.song_data[idx]
        lyrics_vectors = song['lyrics_vectors']
        midi_features = song['midi_features']

        inputs = torch.cat((lyrics_vectors, midi_features), dim=1)
        labels = lyrics_vectors[1:]
        inputs = inputs[:-1]

        return inputs, labels, (lyrics_vectors.shape[0] - 1)


# Modified Collate Function
def collate_fn_sequential(batch):
    """
    Custom collate function to prepare batches for sequential processing of songs.
    Each song is processed individually from start to end.
    """
    inputs = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    lengths = [item[2] for item in batch]

    # Convert lists to tensors
    inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True)
    lengths = torch.tensor(lengths, dtype=torch.long)

    return inputs, labels, lengths

# Updated DataLoader Function
def create_sequential_dataloader(pickle_path, batch_size, shuffle=True):
    """
    Create a DataLoader for sequential song-wise processing.

    Args:
        pickle_path (str): Path to the pickle file containing the dataset.
        batch_size (int): Number of songs per batch.

    Returns:
        DataLoader: DataLoader for the dataset.
    """
    dataset = LyricsMIDIDataset(pickle_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn_sequential)
    return dataloader



# # Testing usage
# pickle_path = "train_song_dataset_with_w2v_and_midi.pkl"  # Adjust to your indexed pickle path
# batch_size = 3  # Number of songs per batch
# train_dataloader = create_sequential_dataloader(pickle_path, batch_size)
# i = 0
#
# # Tests for verifying dataset consistency
# for batch_inputs, batch_labels, batch_lengths in train_dataloader:
#     for song_inputs, song_labels, song_length in zip(batch_inputs, batch_labels, batch_lengths):
#         # song_length is the real length for *this* song
#         # but song_inputs.shape[0] is the max length in the batch
#
#         print("Processing song of length:", song_length)
#
#         # Check real portion only
#         real_inputs = song_inputs[:song_length]
#         real_labels = song_labels[:song_length]
#
#         print("Song inputs shape (unpadded):", real_inputs.shape)
#         print("Song labels shape (unpadded):", real_labels.shape)
#
#         # The next line now checks the unpadded portion
#         assert real_labels.shape[0] == song_length, "Labels length mismatch with song length"
#         print("Test passed: Labels length matches song length.")
#
#     i += 1
#     if i == 2:
#         break
