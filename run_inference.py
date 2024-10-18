import torch
from dataset import MSVDDataset
from seq2seq import Seq2Seq
import pickle
import argparse

# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, help='Directory containing the test data')
parser.add_argument('--output', type=str, help='File to write generated captions')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model and vocab
model = torch.load('models/model.pth').to(device)
model.eval()

with open('vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)
inv_vocab = {v: k for k, v in vocab.items()}

# Load the test dataset
test_data_dir = args.data_dir
test_label_file = './data/MLDS_hw2_1_data/testing_label.json'
test_dataset = MSVDDataset(test_data_dir, test_label_file, vocab, mode='test')
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

# Inference and caption generation
with open(args.output, 'w') as f_out:
    for video, _ in test_dataloader:
        video = video.to(device)
        caption = [vocab['<BOS>']]
        hidden, cell = model.encoder(video)

        for _ in range(30):  # Max caption length
            input_tensor = torch.tensor([caption[-1]]).unsqueeze(0).to(device)
            output, hidden, cell = model.decoder(input_tensor, hidden, cell)
            
            # Check the shape of output
            print(f"Output shape: {output.shape}")
            
            # Use argmax based on the shape of output
            if output.dim() == 2:
                predicted_token = output.argmax(1).item()  # Use argmax(1) for 2D tensors
            else:
                predicted_token = output.argmax(2).item()  # Use argmax(2) for 3D tensors

            if predicted_token == vocab['<EOS>']:
                break
            caption.append(predicted_token)

        caption_words = [inv_vocab[idx] for idx in caption[1:]]  # Exclude <BOS>
        generated_caption = ' '.join(caption_words)
        f_out.write(generated_caption + '\n')
