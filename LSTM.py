import torch
import torch.nn as nn

class LyricsMIDILSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, midi_feature_dim, num_layers=1, dropout=0.3):
        super(LyricsMIDILSTM, self).__init__()

        self.input_dim = input_dim  # Word2Vec dimension
        self.midi_feature_dim = midi_feature_dim  # MIDI feature dimension
        self.hidden_dim = hidden_dim  # LSTM hidden size
        self.output_dim = output_dim  # Word2Vec output size
        self.num_layers = num_layers

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_dim + midi_feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hidden=None):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim + midi_feature_dim).
            hidden (tuple): Optional hidden and cell state for LSTM.

        Returns:
            torch.Tensor: Predicted next word embeddings of shape (batch_size, seq_len, output_dim).
            tuple: Updated hidden and cell states.
        """
        # Pass through LSTM
        lstm_out, hidden = self.lstm(x, hidden)

        # Apply dropout
        lstm_out = self.dropout(lstm_out)

        # Project to output space
        predictions = self.fc(lstm_out)

        return predictions, hidden

# # Example usage
# if __name__ == "__main__":
#     input_dim = 300  # Word2Vec dimension
#     midi_feature_dim = 2  # MIDI features dimension
#     hidden_dim = 512
#     output_dim = 300  # Word2Vec embedding dimension
#     num_layers = 2
#     dropout = 0.1
#
#     model = LyricsMIDILSTM(input_dim, hidden_dim, output_dim, midi_feature_dim, num_layers, dropout)
#
#     # Dummy input: batch of 3 songs, each with 50 time steps
#     batch_size = 3
#     seq_len = 50
#     dummy_input = torch.rand(batch_size, seq_len, input_dim + midi_feature_dim)
#
#     # Forward pass
#     predictions, hidden = model(dummy_input)
#     print("Predictions shape:", predictions.shape)  # Should be (batch_size, seq_len, output_dim)
