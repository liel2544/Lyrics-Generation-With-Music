import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter  # Import TensorBoard SummaryWriter

from torch.utils.tensorboard import SummaryWriter  # Import TensorBoard SummaryWriter

def train_model(model, train_loader, val_loader, device,
                epochs=10, lr=0.0001, checkpoint_path="model_checkpoint.pt"):
    """
    Train the model and log training/validation loss using TensorBoard.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for the training dataset.
        val_loader (DataLoader): DataLoader for the validation dataset.
        device (torch.device): Device to train on.
        epochs (int): Number of epochs.
        lr (float): Learning rate.
        checkpoint_path (str): Path to save the model checkpoint.

    Returns:
        nn.Module: Trained model.
    """
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir="runs/LyricsMIDILSTM2")  # Log directory for TensorBoard
    model.to(device)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for inputs, labels, lengths in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs, _ = model(inputs)

            # Compute loss
            loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1, labels.size(-1)))
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        writer.add_scalar("Loss/Train", avg_train_loss, epoch)  # Log training loss

        # Validate
        val_loss = validate_model(model, val_loader, criterion, device)
        writer.add_scalar("Loss/Validation", val_loss, epoch)  # Log validation loss

        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Validation Loss: {val_loss:.4f}")

        # Save model checkpoint
        torch.save(model.state_dict(), checkpoint_path)

    writer.close()  # Close TensorBoard writer
    return model


# Validation function (unchanged)
def validate_model(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for inputs, labels, lengths in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs, _ = model(inputs)

            # Compute loss
            loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1, labels.size(-1)))
            val_loss += loss.item()

    return val_loss / len(val_loader)





# Testing function
def test_model(model, test_loader, device, word2vec_path, initial_words, num_samples=3, top_k=3):
    """
    Test the model on the test dataset and generate lyrics with a "top_k sampling" approach.

    Args:
        model (nn.Module): The trained model to test.
        test_loader (DataLoader): DataLoader for the test dataset.
        device (torch.device): The device to use for testing.
        word2vec_path (str): Path to the pre-trained Word2Vec binary file.
        initial_words (list): List of initial words to start the generation.
        num_samples (int): Number of generations per song.
        top_k (int): Number of top candidates to sample from.

    Returns:
        dict: Generated lyrics for each song and initial word.
    """
    from gensim.models import KeyedVectors
    print("starting to load")
    word2vec_model = KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
    print("finished loading")
    model.eval()
    results = {}

    with torch.no_grad():
        for inputs, labels, lengths in test_loader:
            inputs = inputs.to(device)
            batch_size = inputs.size(0)

            # Process each "song" in the batch individually
            for i in range(batch_size):
                midi_features = inputs[i : i+1, :, 300:]
                song_length = lengths[i].item()

                for initial_word in initial_words:
                    if initial_word not in word2vec_model:
                        print(f"Word '{initial_word}' not found in Word2Vec vocabulary. Skipping.")
                        continue

                    for _ in range(num_samples):
                        generated = []
                        hidden = None

                        # Convert initial word to Word2Vec vector
                        current_word_vec = torch.tensor(
                            word2vec_model[initial_word],
                            dtype=torch.float32,
                            device=device
                        ).unsqueeze(0).unsqueeze(0)

                        for t in range(song_length):
                            # Get the MIDI features for the current time-step
                            current_midi_step = midi_features[:, t : t+1, :]

                            # Create input = word_vec + MIDI
                            input_ = torch.cat((current_word_vec, current_midi_step), dim=-1)

                            # Forward pass
                            output, hidden = model(input_, hidden)

                            # Convert output embedding to numpy
                            predicted_vec = output[:, -1, :].squeeze(0).cpu().numpy()

                            # --- TOP-K SAMPLING ---
                            top_candidates = word2vec_model.similar_by_vector(predicted_vec, topn=top_k)
                            # top_candidates is a list of (word, similarity)

                            words = [w for (w, sim) in top_candidates]
                            sims  = [sim for (w, sim) in top_candidates]

                            shifted_sims = [s + 1.0 for s in sims]  # shift to make sure they're all positive
                            sum_sims = sum(shifted_sims)
                            probs = [s / sum_sims for s in shifted_sims]

                            # 4) Randomly pick the next word based on these probabilities
                            predicted_word = random.choices(words, weights=probs, k=1)[0]

                            generated.append(predicted_word)

                            # Prepare next input
                            current_word_vec = torch.tensor(
                                word2vec_model[predicted_word],
                                dtype=torch.float32,
                                device=device
                            ).unsqueeze(0).unsqueeze(0)

                        # Store generated sequence
                        results[f"Song {len(results) + 1}, Start Word: {initial_word}"] = generated

    return results





#
# # Example usage
# if __name__ == "__main__":
#     # Dummy setup for testing the code structure
#     from LSTM import LyricsMIDILSTM
#
#     input_dim = 300  # Word2Vec dimension
#     midi_feature_dim = 2  # MIDI features dimension
#     hidden_dim = 512
#     output_dim = 300  # Word2Vec embedding dimension
#     num_layers = 2
#     dropout = 0.1
#
#     model = LyricsMIDILSTM(input_dim, hidden_dim, output_dim, midi_feature_dim, num_layers, dropout)
#
#     # Set up dummy DataLoaders
#     from data_sets_loaders import create_sequential_dataloader
#
#     # Set up DataLoaders
#     train_loader = create_sequential_dataloader("first_lastm_data/train_song_dataset_with_w2v_and_midi.pkl", batch_size=16)
#     val_loader = create_sequential_dataloader("first_lastm_data/test_song_dataset_with_w2v_and_midi.pkl", batch_size=16)
#     test_loader = create_sequential_dataloader("first_lastm_data/test_song_dataset_with_w2v_and_midi.pkl", batch_size=16)  # Replace with actual DataLoaders
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     # Training example
#     trained_model = train_model(model, train_loader, val_loader, device)
#
#     # Testing example
#     test_results = test_model(trained_model, test_loader, device, initial_words=["love", "music", "dance"], num_samples=3)
#     print(test_results)
