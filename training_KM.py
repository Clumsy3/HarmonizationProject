import os
import json
import math
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ====== Configuration & Paths ======

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Paths
TOKENIZED_DATASET_PATH = "TokenizedDataset"
TOKENIZER_PATH = "tokenizer.pkl"
MODEL_SAVE_PATH = "music_transformer_model.pth"

# Special Tokens
PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
SEP_TOKEN = "<sep>"
UNKNOWN_TOKEN = "<unk>"

# Model & Training Hyperparameters
EMB_SIZE = 512
NHEAD = 8
NUM_LAYERS = 6
FFN_HID_DIM = 2048
DROPOUT = 0.1
MAX_SEQ_LEN = 1024

BATCH_SIZE = 16
NUM_EPOCHS = 50
LR = 1e-4
CLIP_GRAD = 1.0

# ====== Tokenizer Load ======

def load_tokenizer(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data['token_to_id'], data['id_to_token']

token_to_id, id_to_token = load_tokenizer(TOKENIZER_PATH)
VOCAB_SIZE = len(token_to_id)
PAD_IDX = token_to_id[PAD_TOKEN]
SOS_IDX = token_to_id[SOS_TOKEN]
EOS_IDX = token_to_id[EOS_TOKEN]
SEP_IDX = token_to_id[SEP_TOKEN]
UNK_IDX = token_to_id[UNKNOWN_TOKEN]

# ====== Dataset (From JSON) ======

class TokenizedMIDIDataset(Dataset):
    def __init__(self, json_dir, token_to_id, max_seq_len):
        self.json_dir = json_dir
        self.token_to_id = token_to_id
        self.max_seq_len = max_seq_len

        self.pad_id = token_to_id[PAD_TOKEN]
        self.sos_id = token_to_id[SOS_TOKEN]
        self.eos_id = token_to_id[EOS_TOKEN]
        self.unk_id = token_to_id[UNKNOWN_TOKEN]

        self.json_files = [os.path.join(json_dir, f) for f in os.listdir(json_dir) if f.endswith(".json")]

    def __len__(self):
        return len(self.json_files)

    def __getitem__(self, idx):
        path = self.json_files[idx]
        with open(path, "r") as f:
            tokens = json.load(f)

        token_ids = [self.token_to_id.get(tok, self.unk_id) for tok in tokens]
        input_seq = [self.sos_id] + token_ids + [self.eos_id]

        # Truncate
        input_seq = input_seq[:self.max_seq_len]
        target_seq = input_seq[1:]
        input_seq = input_seq[:-1]

        # Pad
        input_pad = self.max_seq_len - len(input_seq)
        target_pad = self.max_seq_len - len(target_seq)

        input_seq += [self.pad_id] * input_pad
        target_seq += [self.pad_id] * target_pad

        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(target_seq, dtype=torch.long)

# ====== Positional Encoding ======

class PositionalEncoding(nn.Module):
    def __init__(self, emb_size, dropout, maxlen=5000):
        super().__init__()
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pe = torch.zeros((maxlen, emb_size))
        pe[:, 0::2] = torch.sin(pos * den)
        pe[:, 1::2] = torch.cos(pos * den)
        pe = pe.unsqueeze(0)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pe)

    def forward(self, x):
        x = x + self.pos_embedding[:, :x.size(1), :]
        return self.dropout(x)

# ====== Transformer Decoder Model ======

class MusicTransformerDecoder(nn.Module):
    def __init__(self, vocab_size, emb_size, nhead, ffn_hid_dim, num_layers, dropout, max_seq_len):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, emb_size)
        self.pos_encoder = PositionalEncoding(emb_size, dropout, max_seq_len)

        decoder_layer = TransformerDecoderLayer(d_model=emb_size, nhead=nhead,
                                                dim_feedforward=ffn_hid_dim, dropout=dropout, batch_first=True)
        self.decoder = TransformerDecoder(decoder_layer, num_layers)
        self.output_proj = nn.Linear(emb_size, vocab_size)

    def forward(self, src, tgt_mask=None, src_padding_mask=None):
        x = self.token_embedding(src)
        x = self.pos_encoder(x)
        out = self.decoder(tgt=x, memory=x,
                           tgt_mask=tgt_mask,
                           memory_mask=None,
                           tgt_key_padding_mask=src_padding_mask,
                           memory_key_padding_mask=src_padding_mask)
        return self.output_proj(out)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

# ====== Training Function ======

def train_epoch(model, dataloader, optimizer, criterion, pad_idx, clip_grad):
    model.train()
    total_loss = 0
    for src, tgt in tqdm(dataloader, desc="Training"):
        src, tgt = src.to(DEVICE), tgt.to(DEVICE)
        tgt_mask = model.generate_square_subsequent_mask(src.size(1)).to(DEVICE)
        src_padding_mask = (src == pad_idx)

        optimizer.zero_grad()
        output = model(src, tgt_mask, src_padding_mask)
        loss = criterion(output.view(-1, output.size(-1)), tgt.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

# ====== Main Training Loop ======

if __name__ == "__main__":
    dataset = TokenizedMIDIDataset(TOKENIZED_DATASET_PATH, token_to_id, MAX_SEQ_LEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    model = MusicTransformerDecoder(VOCAB_SIZE, EMB_SIZE, NHEAD, FFN_HID_DIM, NUM_LAYERS, DROPOUT, MAX_SEQ_LEN).to(DEVICE)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = optim.AdamW(model.parameters(), lr=LR)

    print(f"Training model with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} parameters.")

    for epoch in range(1, NUM_EPOCHS + 1):
        avg_loss = train_epoch(model, dataloader, optimizer, criterion, PAD_IDX, CLIP_GRAD)
        print(f"Epoch {epoch}/{NUM_EPOCHS} - Loss: {avg_loss:.4f}")

        if epoch % 5 == 0 or epoch == NUM_EPOCHS:
            ckpt_path = f"model_epoch_{epoch}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'token_to_id': token_to_id,
                'id_to_token': id_to_token
            }, ckpt_path)
            print(f"Checkpoint saved to {ckpt_path}")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Final model saved to {MODEL_SAVE_PATH}")
