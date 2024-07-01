from gensim import downloader
import numpy as np
import os
import re
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from torch.utils.data import Dataset

"""
This script demonstrates a basic pipeline for training an SVM classifier using GloVe word vectors
and evaluating its performance using F1-score.
"""

# Load GloVe word vectors
GLOVE_PATH = 'glove-twitter-200'
glove_vectors = downloader.load(GLOVE_PATH)
model = glove_vectors


class EntityDataSet(Dataset):
    """
    Custom PyTorch Dataset class for handling entity tagging data.
    """

    def __init__(self, file_path, model):
        """
        Initializes the EntityDataSet object.

        Args:
            file_path (str): Path to the data file.
            model: Pre-trained word vector model (e.g., GloVe).
        """
        self.file_path = file_path
        self.model = model
        self.tokenized_sen, self.labels = self.preprocess_data(file_path, self.model)
        self.vector_dim = self.tokenized_sen.shape[-1]

    def load_model(self, vector_type):
        """
        Loads a specific type of pre-trained word vector model.

        Args:
            vector_type (str): Type of pre-trained word vector model to load.

        Returns:
            Model: Loaded pre-trained word vector model.
        """
        if vector_type == 'glove':
            return downloader.load(self.GLOVE_PATH)
        else:
            raise KeyError(f"{vector_type} is not a supported vector type")

    def preprocess_data(self, file_path, model):
        """
        Preprocesses the raw data.

        Args:
            file_path (str): Path to the data file.
            model: Pre-trained word vector model.

        Returns:
            tuple: Tuple containing the tokenized sentences and corresponding labels.
        """
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

        return np.array(representations), np.array(labels).astype('long')

    def read_data(self, file_path):
        """
        Reads the data from the specified file.

        Args:
            file_path (str): Path to the data file.

        Returns:
            list: List of tokenized sentences.
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
        Retrieves an item from the dataset.

        Args:
            item: Index of the item to retrieve.

        Returns:
            dict: Dictionary containing the input data and corresponding labels.
        """
        cur_sen = self.tokenized_sen[item]
        label = self.labels[item]
        data = {"input_ids": cur_sen, "labels": label}
        return data

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.labels)


def main():
    # Set data folder
    data_folder = 'data'

    # Create train and test datasets
    train_set = EntityDataSet(os.path.join(data_folder, 'train.tagged'), model)
    test_set = EntityDataSet(os.path.join(data_folder, 'dev.tagged'), model)

    # Extract input features and labels
    x_train = train_set.tokenized_sen
    x_test = test_set.tokenized_sen
    y_train = train_set.labels
    y_test = test_set.labels

    # Train SVM model
    svm_model = SVC(kernel='linear', C=0.1)
    svm_model.fit(x_train, y_train)

    # Make predictions
    predictions = svm_model.predict(x_test)

    # Calculate F1-score
    f1 = f1_score(y_test, predictions)
    print("F1 Score:", f1)


if __name__ == "__main__":
    main()
