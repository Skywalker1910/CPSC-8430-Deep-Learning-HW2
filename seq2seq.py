import torch.nn as nn
import torch

class EncoderLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2):
        super(EncoderLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

    def forward(self, x):
        # x should be of shape (batch_size, sequence_length, input_dim)
        outputs, (hidden, cell) = self.lstm(x)  # Hidden and cell states should match batch size
        return hidden, cell  # Return hidden and cell for the decoder


class DecoderLSTM(nn.Module):
    def __init__(self, embed_dim, hidden_dim, output_dim, num_layers=2):
        super(DecoderLSTM, self).__init__()
        self.embedding = nn.Embedding(output_dim, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, input, hidden, cell):
        # Input shape: (batch_size, 1) -> we expect a batch with one word at a time
        embedded = self.embedding(input)  # (batch_size, 1, embed_dim)
        
        # Ensure the input to LSTM is correctly batched
        lstm_output, (hidden, cell) = self.lstm(embedded, (hidden, cell))  # Properly batched
        predictions = self.fc_out(lstm_output.squeeze(1))  # (batch_size, output_dim)
        
        return predictions, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
    
    def forward(self, video_features, captions, teacher_forcing_ratio=0.5):
        batch_size = captions.shape[0]
        max_len = captions.shape[1]
        output_dim = self.decoder.fc_out.out_features
        outputs = torch.zeros(batch_size, max_len, output_dim).to(self.device)
        
        hidden, cell = self.encoder(video_features)  # Properly batched hidden and cell states
        input = captions[:, 0]  # <BOS> token, shape: (batch_size)

        for t in range(1, max_len):
            output, hidden, cell = self.decoder(input.unsqueeze(1), hidden, cell)
            outputs[:, t] = output
            top1 = output.argmax(1)
            input = captions[:, t] if torch.rand(1).item() < teacher_forcing_ratio else top1
        
        return outputs
