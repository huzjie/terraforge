"""Data loader: batching with deterministic shuffling."""
import random


class DataLoader:
    def __init__(self, dataset, batch_size=8, shuffle=True, seed=42):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        idx = list(range(len(self.dataset)))
        if self.shuffle:
            random.Random(self.seed).shuffle(idx)
        for i in range(0, len(idx), self.batch_size):
            yield [self.dataset[j] for j in idx[i:i + self.batch_size]]

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size
