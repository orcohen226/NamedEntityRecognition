import os
import re
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
from gensim import downloader
import matplotlib.pyplot as plt

# Load pre-trained GloVe vectors
GLOVE_PATH = 'glove-twitter-100'
glove_vectors = downloader.load(GLOVE_PATH)
model = glove_vectors


class EntityDataSet(Dataset):
    """
    Dataset class for entity recognition.
    
    Args:
        file_path (str): Path to the dataset file.
        model: Pre-trained word embedding model.
    """

    def __init__(self, file_path, model):
        """
        Initialize the EntityDataSet.

        Args:
            file_path (str): Path to the dataset file.
            model: Pre-trained word embedding model.
        """
        self.file_path = file_path
        self.model = model
        self.tokenized_sen, self.labels = self.preprocess_data(file_path, self.model)
        self.vector_dim = self.tokenized_sen.shape[-1]

    def preprocess_data(self, file_path, model):
        """
        Preprocess the dataset.

        Args:
            file_path (str): Path to the dataset file.
            model: Pre-trained word embedding model.

        Returns:
            tuple: Tokenized sentences and corresponding labels.
        """
        # Preprocessing steps
        formated_sentences = self.read_data(file_path)
        context_window = 2
        dims = model.vector_size

        representations = []
        labels = []
        oov = np.ones(dims) * 6
        pad = np.ones(dims) * 7
        for sentence in formated_sentences:
            sentence_representation = [pad] * context_window
            tags = [None] * context_window

            for word, tag in sentence:
                bin_tag = 0 if tag == 'o' else 1

                if word not in model.key_to_index:
                    sentence_representation.append(oov)
                    tags.append(bin_tag)
                else:
                    vec = model[word]
                    sentence_representation.append(vec)
                    tags.append(bin_tag)

            tags = tags + [None] * context_window
            sentence_representation = sentence_representation + [pad] * context_window

            for i in range(context_window, len(sentence_representation) - context_window):
                labels.append(tags[i])
                representations.append(
                    np.concatenate(sentence_representation[i - context_window:i + context_window + 1]))

        # Convert data and labels to PyTorch tensors
        representations_tensor = torch.tensor(np.array(representations), dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.long)

        return representations_tensor, labels_tensor

    def read_data(self, file_path):
        """
        Read data from the dataset file.

        Args:
            file_path (str): Path to the dataset file.

        Returns:
            list: List of formatted sentences.
        """
        # Read data from file
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

        for sentence in sentences:
            formated_sentence = []
            sentence = [sen.strip().lower() for sen in sentence]
            for sen in sentence:
                word, tag = sen.split('\t')
                formated_sentence.append([re.sub(r'\W+', '', word), tag])
            formated_sentences.append(formated_sentence)

        return formated_sentences

    def __getitem__(self, item):
        """
        Get item from the dataset.

        Args:
            item (int): Index of the item.

        Returns:
            dict: Dictionary containing input_ids and labels.
        """
        cur_sen = self.tokenized_sen[item]
        cur_sen = cur_sen.squeeze()
        label = self.labels[item]
        data = {"input_ids": cur_sen, "labels": label}
        return data

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.labels)


class EntityNN(nn.Module):
    """
    Neural network model for entity recognition.

    Args:
        input_dim (int): Dimensionality of input features.
        num_classes (int): Number of classes.
        hidden_dims (list): List of hidden layer dimensions.
        activation (str): Activation function for hidden layers.
        class_weights (list): Weights for different classes.
        dropout_prob (float): Dropout probability for regularization.
    """

    def __init__(self, input_dim, num_classes, hidden_dims=[64, 32], activation='relu', class_weights=[0.5, 0.5],
                 dropout_prob=0.3):
        """
        Initialize the EntityNN model.

        Args:
            input_dim (int): Dimensionality of input features.
            num_classes (int): Number of classes.
            hidden_dims (list): List of hidden layer dimensions.
            activation (str): Activation function for hidden layers.
            class_weights (list): Weights for different classes.
            dropout_prob (float): Dropout probability for regularization.
        """
        super(EntityNN, self).__init__()
        self.hidden_layers = nn.ModuleList()
        prev_dim = input_dim
        for dim in hidden_dims:
            self.hidden_layers.append(nn.Linear(prev_dim, dim))
            if activation == 'relu':
                self.hidden_layers.append(nn.ReLU())
            elif activation == 'tanh':
                self.hidden_layers.append(nn.Tanh())
            elif activation == 'leaky_relu':
                self.hidden_layers.append(nn.LeakyReLU())
            elif activation == 'elu':
                self.hidden_layers.append(nn.ELU())
            self.hidden_layers.append(nn.Dropout(p=dropout_prob))
            prev_dim = dim
        self.output_layer = nn.Linear(prev_dim, num_classes)
        self.loss = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights))

    def forward(self, input_ids, labels=None):
        """
        Forward pass of the model.

        Args:
            input_ids: Input tensor.
            labels: Target labels.

        Returns:
            tuple: Tuple containing output tensor and loss.
        """
        x = input_ids
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.output_layer(x)
        if labels is None:
            return x, None
        loss = self.loss(x, labels)
        return x, loss


def train(model, data_loaders, optimizer, num_epochs: int):
    """
    Train the neural network model.

    Args:
        model: Neural network model.
        data_loaders (dict): Data loaders for train and test sets.
        optimizer: Optimizer for training.
        num_epochs (int): Number of training epochs.

    Returns:
        tuple: Tuple containing best accuracy, epoch, test and train accuracies.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    train_epoch_accuracies = []
    test_epoch_accuracies = []
    best_acc = 0.0
    best_epoch = None
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            labels, preds = [], []

            for batch in data_loaders[phase]:
                input_ids = batch['input_ids'].to(device)
                labels_batch = batch['labels'].to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs, loss = model(input_ids, labels_batch)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                pred = outputs.argmax(dim=-1)
                labels += labels_batch.cpu().view(-1).tolist()
                preds += pred.cpu().view(-1).tolist()
                running_loss += loss.item() * input_ids.size(0)

            epoch_loss = running_loss / len(data_loaders[phase].dataset)
            epoch_acc = f1_score(labels, preds)

            if phase.title() == "Test":
                print(f'{phase.title()} Loss: {epoch_loss:.4e} Accuracy: {epoch_acc:.4f}')
                test_epoch_accuracies.append((epoch, epoch_acc))
            else:
                print(f'{phase.title()} Loss: {epoch_loss:.4e} Accuracy: {epoch_acc:.4f}')
                train_epoch_accuracies.append((epoch, epoch_acc))

            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_epoch = epoch
                torch.save(model.state_dict(), 'best_model.pth')

    return best_acc, best_epoch, test_epoch_accuracies, train_epoch_accuracies


def main():
    # Define data folder
    data_folder = 'data'

    # Load datasets
    train_set = EntityDataSet(os.path.join(data_folder, 'train.tagged'), model)
    test_set = EntityDataSet(os.path.join(data_folder, 'dev.tagged'), model)

    # Calculate class weights
    all_tags = train_set.labels
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(np.array(all_tags)),
                                         y=np.array(all_tags))

    # Define activation function and batch size
    activation = 'elu'
    batch_size = 8

    # Create data loaders
    datasets = {"train": train_set, "test": test_set}
    train_set = datasets['train']
    data_loaders = {"train": DataLoader(datasets["train"], batch_size=batch_size, shuffle=True),
                    "test": DataLoader(datasets["test"], batch_size=batch_size, shuffle=False)}

    # Define neural network architecture
    hidden_dim = [128, 32]
    nn_model = EntityNN(input_dim=train_set.vector_dim, hidden_dims=hidden_dim, num_classes=2, activation=activation,
                        class_weights=class_weights)

    # Define optimizer and number of epochs
    optimizer = Adam(params=nn_model.parameters())
    num_epochs = 10

    # Train the model
    res = train(model=nn_model, data_loaders=data_loaders, optimizer=optimizer, num_epochs=num_epochs)

    # Print results
    print(f'Best Accuracy: {res[0]:.4f}, Epoch: {res[1]}')
    print(f'Test Accuracies: {res[2]}')
    print(f'Train Accuracies: {res[3]}')

    train_acc = list(zip(*res[3]))[1]
    val_acc = list(zip(*res[2]))[1]
    # Assuming train_acc and val_acc are lists containing accuracy values for each epoch
    epochs = range(1, len(train_acc) + 1)

    plt.plot(epochs, train_acc, 'b', label='Training accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.savefig('Train_validation_accuracies_FF.png')


if __name__ == "__main__":
    main()
