import os
import json
import torch
from torch.utils.data import Dataset
import numpy as np
from nltk.tokenize import word_tokenize

class MSVDDataset(Dataset):
    def __init__(self, data_dir, label_file, vocab=None, max_len=30, mode='train'):
        """
        data_dir: Directory containing video features (npy files).
        label_file: Path to the JSON file containing captions.
        vocab: Pre-built vocabulary (if available), otherwise it will be built.
        max_len: Maximum caption length for padding.
        mode: 'train' or 'test'
        """
        self.data_dir = data_dir
        self.max_len = max_len
        self.mode = mode
        self.videos = self.load_video_features()
        self.captions = self.load_captions(label_file)
        
        # Build vocabulary if not provided
        if vocab is None:
            self.vocab = self.build_vocab(self.captions)
        else:
            self.vocab = vocab

    def load_video_features(self):
        """Load video feature files from the 'feat' directory inside training_data or testing_data."""
        feat_dir = os.path.join(self.data_dir, 'feat')  # Append 'feat' to the provided data_dir path
        video_files = [os.path.join(feat_dir, f) for f in os.listdir(feat_dir) if f.endswith('.npy')]
        
        # Debugging: print the path and number of files found
        print(f"Looking for .npy files in: {feat_dir}")
        print(f"Found {len(video_files)} .npy files")

        video_features = {os.path.basename(f).replace('.npy', ''): np.load(f) for f in video_files}
        return video_features

    def load_captions(self, label_file):
        """Load captions from the JSON label file."""
        with open(label_file, 'r') as f:
            captions_data = json.load(f)
        
        # Create a dictionary with video ids as keys and corresponding captions as values
        captions = {item['id']: item['caption'] for item in captions_data}
        return captions

    def build_vocab(self, captions):
        """Build vocabulary from captions."""
        word_counts = {}
        for caption_list in captions.values():
            for caption in caption_list:
                tokens = word_tokenize(caption.lower())
                for token in tokens:
                    word_counts[token] = word_counts.get(token, 0) + 1
        
        vocab = {'<PAD>': 0, '<BOS>': 1, '<EOS>': 2, '<UNK>': 3}
        idx = 4
        for word, count in word_counts.items():
            if count >= 3:  # Threshold for including words in the vocabulary
                vocab[word] = idx
                idx += 1
        return vocab

    def __len__(self):
        """Return the number of videos in the dataset."""
        return len(self.videos)

    def __getitem__(self, idx):
        """Return the video feature and caption for a given index."""
        video_id = list(self.videos.keys())[idx]
        video_feature = torch.tensor(self.videos[video_id], dtype=torch.float32)

        if self.mode == 'train':
            # Get a random caption for this video
            caption_list = self.captions[video_id]
            caption = np.random.choice(caption_list)
            
            # Tokenize and convert caption to indices
            tokens = word_tokenize(caption.lower())
            caption_indices = [self.vocab.get(token, self.vocab['<UNK>']) for token in tokens]
            caption_indices = [self.vocab['<BOS>']] + caption_indices + [self.vocab['<EOS>']]
            caption_indices = caption_indices[:self.max_len]  # Truncate if necessary
            caption_indices += [self.vocab['<PAD>']] * (self.max_len - len(caption_indices))  # Pad to max_len

            return video_feature, torch.tensor(caption_indices, dtype=torch.long)
        else:
            # For test mode, we just return the video feature (no caption available)
            return video_feature, torch.tensor([])

