from torch import nn
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from gensim import downloader
import numpy as np
import re
from torch.optim import Adam
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

"""
This script demonstrates a basic pipeline for training a LSTM neural network classifier
for Named Entity Recognition (NER) using pre-trained word vectors (e.g., GloVe) and evaluating
its performance using F1-score.
"""

GLOVE_PATH = 'glove-twitter-200'
glove_vectors = downloader.load(GLOVE_PATH)
model = glove_vectors


def read_data(file_path):
    """
    Reads data from the specified file.

    Args:
        file_path (str): Path to the data file.

    Returns:
        tuple: Tuple containing formatted sentences and corresponding labels.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    sentences = []
    sentence = []

    for line in lines:
        if line.isspace():
            sentences.append(sentence)
            sentence = []
        else:
            sentence.append(line[:-1])

    formated_sentences = []
    formated_tags = []
    for sentence in sentences:
        formated_sentence, tags = [], []
        sentence = [sen.strip().lower() for sen in sentence]
        for sen in sentence:
            word, tag = sen.split('\t')
            tags.append(0 if tag == 'o' else 1)
            formated_sentence.append(re.sub(r'\W+', '', word))
        formated_sentences.append(formated_sentence)
        formated_tags.append(tags)

    return (formated_sentences, formated_tags)


def preprocess_data(formated_sentences, formated_labels, models):
    """
    Preprocesses the data.

    Args:
        formated_sentences (list): List of formatted sentences.
        formated_labels (list): List of corresponding labels.
        models (list): List of pre-trained word vector models.

    Returns:
        tuple: Tuple containing representations, sentence labels, and all labels.
    """
    representations = []
    sentence_labels = []
    all_labels = []

    for sentence, labels in zip(formated_sentences, formated_labels):
        all_labels += labels
        embedded_sentence = []
        for word, tag in zip(sentence, labels):
            word_embedding = []
            for model in models:
                if word in model:
                    word_embedding += list(model[word])
                else:
                    word_embedding += [6] * model.vector_size
            embedded_sentence.append(torch.as_tensor(word_embedding, dtype=torch.float32))
        embedded_sentence = torch.stack(embedded_sentence)
        representations.append(embedded_sentence)
        sentence_labels.append(torch.LongTensor(labels))

    return representations, sentence_labels, all_labels


class LSTM_NER(nn.Module):
    """
    LSTM-based Named Entity Recognition (NER) model.
    """

    def __init__(self, vocab_size, tag_dim, weights, hidden_dim=64, num_layers=2, dropout=0.4):
        """
        Initializes the LSTM_NER model.

        Args:
            vocab_size (int): Size of the vocabulary.
            tag_dim (int): Dimensionality of the tags.
            weights (tensor): Class weights.
            hidden_dim (int): Dimensionality of the hidden layers.
            num_layers (int): Number of LSTM layers.
            dropout (float): Dropout probability.
        """
        super(LSTM_NER, self).__init__()
        self.lstm = nn.LSTM(vocab_size, hidden_dim, num_layers=num_layers, dropout=dropout, batch_first=True)
        self.hidden2tag_1 = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.ELU()
        self.hidden2tag_2 = nn.Linear(hidden_dim, tag_dim)
        self.weights = weights
        self.loss = nn.CrossEntropyLoss(weight=self.weights, reduction='mean')

    def forward(self, sentence, tags=None):
        """
        Forward pass of the model.

        Args:
            sentence (tensor): Input sentence.
            tags (tensor): True tags (optional).

        Returns:
            tuple: Tuple containing predicted logits and loss (if tags are provided).
        """
        sentence_3d = sentence.unsqueeze(1)
        lstm_out, _ = self.lstm(sentence_3d)
        lstm_out = self.hidden2tag_1(lstm_out)
        lstm_out = self.activation(lstm_out)
        lstm_out = self.hidden2tag_2(lstm_out)
        lstm_out = lstm_out.squeeze(1)
        if tags is None:
            return lstm_out, None

        loss = self.loss(lstm_out, tags)
        return lstm_out, loss


def train(model, data, optimizer, num_epochs):
    """
    Trains the LSTM_NER model.

    Args:
        model: LSTM_NER model.
        data (dict): Dictionary containing train and dev datasets.
        optimizer: Optimization algorithm.
        num_epochs (int): Number of training epochs.

    Returns:
        None
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_F1 = []
    dev_F1 = []
    model.to(device)
    max_F1 = 0
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1} of {num_epochs}")
        print("-" * 25)
        for phase in ["train", "dev"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            labels = []
            predictions = []
            dataset = data[phase]
            for sentence, sentence_labels in zip(dataset.tokenized_sen, dataset.labels):
                if phase == "train":
                    outputs, loss = model(sentence, sentence_labels)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                else:
                    with torch.no_grad():
                        outputs, loss = model(sentence, sentence_labels)

                prediction = outputs.argmax(dim=-1).clone().detach().cpu()
                labels += sentence_labels.cpu().view(-1).tolist()
                predictions += prediction.view(-1).tolist()

            epoch_F1 = f1_score(labels, predictions)
            if phase == "train":
                train_F1.append(epoch_F1)
            else:
                dev_F1.append(epoch_F1)
            print(f"{phase} F1: {epoch_F1}")

            # update max f1 score
            if phase == "dev" and epoch_F1 > max_F1:
                max_F1 = epoch_F1

    print(f"Max F1: {max_F1:.4f}")
    return (train_F1, dev_F1)


# Read and preprocess data
train_sentences, train_labels = read_data(os.path.join('data', 'train.tagged'))
x_train, y_train, all_labels_train = preprocess_data(train_sentences, train_labels, [model])

test_sentences, test_labels = read_data(os.path.join('data', 'dev.tagged'))
x_test, y_test, all_labels_test = preprocess_data(test_sentences, test_labels, [model])


# Create datasets
class EntityDataSet(Dataset):
    def __init__(self, x, y, all_labels):
        self.tokenized_sen = x
        self.labels = y
        self.all_labels = all_labels

    def __getitem__(self, idx):
        return self.tokenized_sen[idx], self.labels[idx]

    def __len__(self):
        return len(self.labels)

def main():
    train_data = EntityDataSet(x_train, y_train, all_labels_train)
    test_data = EntityDataSet(x_test, y_test, all_labels_test)

    # Compute class weights
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_data.all_labels),
                                         y=np.array(train_data.all_labels))
    class_weights = torch.FloatTensor(class_weights)

    # Initialize model and optimizer
    nn_model = LSTM_NER(vocab_size=train_data.tokenized_sen[0].shape[1], tag_dim=2, weights=class_weights)
    optimizer = Adam(params=nn_model.parameters())
    num_epochs = 6

    # Train the model
    datasets = {"train": train_data, "dev": test_data}
    train_f1, test_f1 = train(nn_model, datasets, optimizer, num_epochs)

    epochs = range(1, len(train_f1) + 1)

    plt.plot(epochs, train_f1, 'b', label='Training accuracy')
    plt.plot(epochs, test_f1, 'r', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('Train_validation_accuracies_LSTM.png')


if __name__ == "__main__":
    main()
