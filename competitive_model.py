from torch import nn
import os
import torch
from torch.utils.data import Dataset
from sklearn.metrics import f1_score
from gensim import downloader
import numpy as np
import re
from torch.optim import Adam
import pickle
from sklearn.utils.class_weight import compute_class_weight
import random

model_1 = downloader.load('glove-wiki-gigaword-300')
model_2 = downloader.load('word2vec-google-news-300')
model_3 = downloader.load('glove-twitter-200')
model_4 = downloader.load('fasttext-wiki-news-subwords-300')


def read_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    sentences = []
    sentence = []
    sentences_labels = []
    sentence_labels = []

    for line in lines:
        line = re.sub(r'\ufeff', '', line)
        if line.isspace():
            sentences.append(sentence)
            sentences_labels.append(sentence_labels)
            sentence = []
            sentence_labels = []
        else:
            word, label = line.split('\t')
            label = 0 if label.strip() == "O" else 1
            sentence.append(word)
            sentence_labels.append(label)

    return sentences, sentences_labels


def preprocess(sentences, models, sentences_labels):
    data = []
    labels_list = []
    all_labels = []
    feature_mapping = {}

    for sentence, labels in zip(sentences, sentences_labels):
        all_labels += labels
        tokenized_sen = []
        for word in sentence:

            word_embedding = []
            if word not in feature_mapping:
                for model in models:
                    if word in model:
                        word_embedding += list(model[word])
                    else:
                        word_embedding += [0] * model.vector_size

                word_embedding.append(int(word[0].isupper()))
                word_embedding.append(int(word.islower()))
                word_embedding.append(int(word.isupper()))
                word_embedding.append(int(any(char.isdigit() for char in word)))
                word_embedding.append(int(all(char.isdigit() for char in word)))
                word_embedding.append(int(any(char == '@' for char in word)))
                word_embedding.append(int(any(char == '#' for char in word)))
                word_embedding.append(int(any(char in {',', '.', ':', ';', "'", '"'} for char in word)))

                concatenated_vec = torch.as_tensor(word_embedding, dtype=torch.float32)
                feature_mapping[word] = concatenated_vec
                tokenized_sen.append(concatenated_vec)
            else:
                tokenized_sen.append(feature_mapping[word])

        final_tokenized_sen = torch.stack(tokenized_sen)
        data.append(final_tokenized_sen)
        labels_list.append(torch.LongTensor(labels))

    return data, labels_list, all_labels


class LSTM_NER(nn.Module):
    def __init__(self, vocab_size, tag_dim, weights, hidden_dim=64, dropout=0.4):
        super(LSTM_NER, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(vocab_size, hidden_dim, num_layers=2, bidirectional=True, dropout=0.3)
        self.hidden2tag_1 = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.activation = nn.ReLU()
        self.hidden2tag_2 = nn.Linear(hidden_dim * 2, tag_dim)
        self.weights = weights
        self.loss = nn.CrossEntropyLoss(weight=self.weights, reduction='mean')

    def forward(self, sentence, tags=None):
        sentence_3d = sentence.unsqueeze(1)
        sentence_3d = self.dropout(sentence_3d)
        lstm_out, _ = self.lstm(sentence_3d)
        lstm_out = self.hidden2tag_1(lstm_out)
        lstm_out = self.activation(lstm_out)
        lstm_out = self.hidden2tag_2(lstm_out)
        lstm_out = lstm_out.squeeze(1)
        if tags is None:
            return lstm_out, None

        loss = self.loss(lstm_out, tags)
        return lstm_out, loss


class EntityDataSet(Dataset):
    def __init__(self, file_path, models):
        self.file_path = file_path
        self.sentences, labels = read_data(file_path)
        self.tokenized_sen, self.labels, self.all_labels = preprocess(self.sentences, models, labels)
        self.vector_dim = self.tokenized_sen[0].shape[-1]


def train(model, data, optimizer, num_epochs):
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
                    if sum(sentence_labels) != 0 or 0.3 < random.random():
                        outputs, loss = model(sentence, sentence_labels)
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                        prediction = outputs.argmax(dim=-1).clone().detach().cpu()
                        labels += sentence_labels.cpu().view(-1).tolist()
                        predictions += prediction.view(-1).tolist()

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

    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)


models = [model_1, model_2, model_3, model_4]


def Train_Final_Model():
    num_epochs = 30
    train_set = EntityDataSet(os.path.join('data', 'train.tagged'), models)
    test_set = EntityDataSet(os.path.join('data', 'dev.tagged'), models)
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_set.all_labels),
                                         y=np.array(train_set.all_labels))
    print(f"Classes weights : {class_weights}")
    class_weights = torch.FloatTensor(class_weights)
    nn_model = LSTM_NER(vocab_size=train_set.vector_dim, tag_dim=2, weights=class_weights)

    optimizer = Adam(params=nn_model.parameters())

    datasets = {"train": train_set, "dev": test_set}
    train(nn_model, datasets, optimizer, num_epochs)


Train_Final_Model()


def test_read_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    sentences = []
    sentence = []
    sentences_labels = []
    sentence_labels = []

    for line in lines:
        line = re.sub(r'\ufeff', '', line)
        if line.isspace():
            sentences.append(sentence)
            sentence = []
        else:
            word = line.strip()
            sentence.append(word)
    return sentences


# Define the preprocessing function for new data
def preprocess_new_data(sentences, models):
    data = []
    feature_mapping = {}

    for sentence in sentences:
        tokenized_sen = []
        for word in sentence:
            word_embedding = []
            if word not in feature_mapping:
                for model in models:
                    if word in model:
                        word_embedding += list(model[word])
                    else:
                        word_embedding += [0] * model.vector_size

                word_embedding.append(int(word[0].isupper()))
                word_embedding.append(int(word.islower()))
                word_embedding.append(int(word.isupper()))
                word_embedding.append(int(any(char.isdigit() for char in word)))
                word_embedding.append(int(all(char.isdigit() for char in word)))
                word_embedding.append(int(any(char == '@' for char in word)))
                word_embedding.append(int(any(char == '#' for char in word)))
                word_embedding.append(int(any(char in {',', '.', ':', ';', "'", '"'} for char in word)))

                concatenated_vec = torch.as_tensor(word_embedding, dtype=torch.float32)
                feature_mapping[word] = concatenated_vec
                tokenized_sen.append(concatenated_vec)
            else:
                tokenized_sen.append(feature_mapping[word])

        final_tokenized_sen = torch.stack(tokenized_sen)
        data.append(final_tokenized_sen)

    return data


def predict(model, new_data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    predictions = []
    for sentence in new_data:
        sentence_tensor = sentence.to(device)  # Add batch dimension
        with torch.no_grad():
            outputs, _ = model(sentence_tensor)
        # Get the predicted labels
        predicted_labels = outputs.argmax(dim=-1).cpu().numpy()
        predictions.append(predicted_labels.tolist())
    return predictions


def main():
    test_sentences = test_read_data(os.path.join('data', 'test.untagged'))
    test_embedding = preprocess_new_data(test_sentences, models)

    with open('model.pkl', 'rb') as file:
        best_model = pickle.load(file)

    predictions = predict(best_model, test_embedding)

    with open('competitive_model.tagged', 'w', encoding="utf-8") as file:
        for sentence, tags in zip(test_sentences, predictions):
            for word, tag in zip(sentence, tags):
                file.write(f"{word}\t{tag}\n")
            file.write('\n')


if __name__ == "__main__":
    main()
