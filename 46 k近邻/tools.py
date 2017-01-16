import numpy as np


def save_npz(file_name, train, train_labels):
    np.savez(file_name, train=train, train_labels=train_labels)


def load_npz_test(file_name):
    with np.load(file_name) as data:
        print(data.files)
        train = data['train']
        train_labels = data['train_labels']
        return train, train_labels