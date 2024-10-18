import nltk
nltk.data.path.append('C:/Users/ADITYA MORE/AppData/Roaming/nltk_data') 
nltk.download('punkt_tab')

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from dataset import MSVDDataset
from seq2seq import Seq2Seq, EncoderLSTM, DecoderLSTM
import pickle

# Hyperparameters
input_dim = 4096  # Example for pre-extracted video features
hidden_dim = 256
embed_dim = 256
output_dim = None  # To be set based on the vocab size
batch_size = 32
num_epochs = 200
learning_rate = 0.001

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Modify the paths to match your dataset structure
train_data_dir = './data/MLDS_hw2_1_data/training_data/'  # The directory containing 'feat/'
train_label_file = './data/MLDS_hw2_1_data/training_label.json'

# Load dataset and build vocab
train_dataset = MSVDDataset(train_data_dir, train_label_file, max_len=30)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Save the vocabulary for inference use
with open('vocab.pkl', 'wb') as f:
    pickle.dump(train_dataset.vocab, f)

# Initialize model, loss function, and optimizer
encoder = EncoderLSTM(input_dim, hidden_dim).to(device)
decoder = DecoderLSTM(embed_dim, hidden_dim, len(train_dataset.vocab)).to(device)
model = Seq2Seq(encoder, decoder, device).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=train_dataset.vocab['<PAD>'])
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for video, caption in train_dataloader:
        video, caption = video.to(device), caption.to(device)
        optimizer.zero_grad()
        output = model(video, caption)
        loss = criterion(output.view(-1, output.size(2)), caption.view(-1))
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss / len(train_dataloader)}')

# Save the trained model
torch.save(model, 'models/model.pth')
