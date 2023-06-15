import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import nltk
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
import torch.nn.utils.rnn as rnn_utils

class TranslationDataset(Dataset):
    def __init__(self, source_path, target_path, max_length=None):
        self.source_sentences = self.load_sentences(source_path, max_length)
        self.target_sentences = self.load_sentences(target_path, max_length)

        self.source_vocab = self.build_vocab(self.source_sentences)
        self.target_vocab = self.build_vocab(self.target_sentences)

        self.source_indices = self.encode_sentences(self.source_sentences, self.source_vocab)
        self.target_indices = self.encode_sentences(self.target_sentences, self.target_vocab)

    def load_sentences(self, file_path, max_length):
        with open(file_path, 'r', encoding='utf-8') as f:
            sentences = [line.strip() for line in f.readlines()]
            if max_length:
                sentences = [sentence[:max_length] for sentence in sentences]
        return sentences

    def build_vocab(self, sentences):
        vocab = {'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3}
        for sentence in sentences:
            for word in sentence.split():
                if word not in vocab:
                    vocab[word] = len(vocab)
        return vocab

    def encode_sentences(self, sentences, vocab):
        indices = []
        for sentence in sentences:
            indices.append([vocab.get(word, vocab['<UNK>']) for word in sentence.split()])
        return indices

    def __len__(self):
        return len(self.source_sentences)

    def __getitem__(self, index):
        source_indices = self.source_indices[index]
        target_indices = self.target_indices[index]
        return source_indices, target_indices

def collate_fn(batch):
    # Sort the batch in descending order of source sentence length
    batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)

    # Separate the source and target sentences
    source_sentences, target_sentences = zip(*batch)

    # Pad the source sentences to have the same length
    source_sentences_padded = pad_sequence(source_sentences, batch_first=True)

    # Pad the target sentences to have the same length
    target_sentences_padded = pad_sequence(target_sentences, batch_first=True)

    return source_sentences_padded, target_sentences_padded
    
class TranslatorAttention(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        super(TranslatorAttention, self).__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.attention = nn.Linear(hidden_dim * 2, 1)

    def forward(self, source_indices):
        embedded = self.embedding(source_indices)
        output, (hidden, cell) = self.lstm(embedded)
        hidden_concat = torch.cat((hidden[-2], hidden[-1]), dim=1)
        attention_weights = torch.softmax(self.attention(hidden_concat), dim=1)
        print(attention_weights.shape)
        print(output.shape)
        context_vector = torch.bmm(attention_weights.unsqueeze(2), output.permute(0, 2, 1)).squeeze(2)

        prediction = self.fc(context_vector)
        return prediction
    
def pad_sequence(sequences, batch_first=False, padding_value=0):
    return torch.nn.utils.rnn.pad_sequence([torch.tensor(seq) for seq in sequences], batch_first=batch_first, padding_value=padding_value)

def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for source_indices, target_indices in iterator:
            source_indices = source_indices.to(device)
            target_indices = target_indices.to(device)
            output = model(source_indices)
            output = output.view(-1, output.shape[-1])
            target_indices = target_indices.view(-1)
            loss = criterion(output, target_indices)
            epoch_loss += loss.item()
    return epoch_loss / len(iterator)

def train_attention(model, iterator, optimizer, criterion):
    model.train()
    epoch_loss = 0
    for source_indices, target_indices in iterator:
        source_indices = source_indices.to(device)
        target_indices = target_indices.to(device)
        optimizer.zero_grad()
        output = model(source_indices)
        output = output.view(-1, output.shape[-1])
        target_indices = target_indices.view(-1)
        loss = criterion(output, target_indices)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)

def calculate_bleu_score(model, iterator, target_vocab):
    references = []
    hypotheses = []

    model.eval()
    with torch.no_grad():
        for source_indices, target_indices in iterator:
            source_indices = source_indices.to(device)
            output = model(source_indices)
            output = output.argmax(dim=-1).cpu().numpy()

            # Convert indices to words
            references.extend([[target_vocab.idx2word[idx] for idx in indices] for indices in target_indices.tolist()])
            hypotheses.extend([target_vocab.idx2word[idx] for idx in indices] for indices in output)

    bleu_score = corpus_bleu([[reference] for reference in references], hypotheses)
    return bleu_score

source_file = './data/train_ch.txt'
target_file = './data/train_en.txt'
val_source_file = './data/val_ch.txt'
val_target_file = './data/val_en.txt'
test_source_file = './data/test_ch.txt'
test_target_file = './data/test_en.txt'

max_length = 100
embedding_dim = 256
hidden_dim = 512
batch_size = 32
num_epochs = 10
learning_rate = 0.001
print("init arg")
dataset = TranslationDataset(source_file, target_file, max_length)
val_dataset = TranslationDataset(val_source_file, val_target_file, max_length)
test_dataset = TranslationDataset(test_source_file, test_target_file, max_length)
print("get dataset")
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
print("load data")
input_dim = len(dataset.source_vocab)
output_dim = len(dataset.target_vocab)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device: ", device)
model = TranslatorAttention(input_dim, embedding_dim, hidden_dim, output_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss(ignore_index=dataset.target_vocab['<PAD>'])
print("create model, optimizer and loss function")
best_valid_loss = float('inf')

for epoch in range(num_epochs):
    train_loss = train_attention(model, train_loader, optimizer, criterion)
    valid_loss = evaluate(model, val_loader, criterion)
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'translator_model.pt')
    print(f'Epoch: {epoch+1}, Training Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}')

model.load_state_dict(torch.load('translator_model.pt'))

test_loss = evaluate(model, test_loader, criterion)
print(f'Test Loss: {test_loss:.4f}')

# 在测试集上计算 Bleu 分数
bleu_score = calculate_bleu_score(model, test_loader, dataset.target_vocab)
print(f"Bleu Score: {bleu_score:.4f}")