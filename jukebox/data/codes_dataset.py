import librosa
import math
import numpy as np
import torch
import jukebox.utils.dist_adapter as dist
from torch.utils.data import Dataset
from jukebox.utils.dist_utils import print_all

class FilesTextDataset(Dataset):
    def __init__(self, hps, data_file):
        super().__init__()
        self.data_file = data_file
        self.min_length = hps.min_length 
        self.max_length = hps.max_length
        self.sample_length = hps.n_ctx
        self.init_dataset(hps)

    def filter(self, data, name):
        # Remove files too short or too long
        keep = []
        for i in range(len(data)):
            if len(data[i]) < self.min_length:
                continue
            if len(data[i]) > self.max_length:
                continue
            keep.append(i)
        print_all(f' min: {self.min_length}, max: {self.max_length}')
        print_all(f"Keeping {len(keep)} of {len(data)} samples")
        self.data = [data[i] for i in keep]
        self.names = [name[i] for i in keep]

    def init_dataset(self, hps):
        # Load list of files and starts/durations
        f = open(f'{self.data_file}')  
        root = f.readline().strip()
        name = []
        data = []
        
        for line in f:
            line = line.strip().split('\t')
            if len(line) == 2:
                name.append(root+'/'+ line[0])
                data.append(np.array(list(map(int, line[1].split()))))

        f.close()

        cache = dist.get_rank() % 8 == 0 if dist.is_available() else True
        self.filter(data, name)

    def get_metadata(self, filename, test):
        """
        Insert metadata loading code for your dataset here.
        If artist/genre labels are different from provided artist/genre lists,
        update labeller accordingly.

        Returns:
            (artist, genre, full_lyrics) of type (str, str, str). For
            example, ("unknown", "classical", "") could be a metadata for a
            piano piece.
        """
        return None, None, None
    

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return {'name': self.names[item], 'data': self.data[item]}

    def collate(self, batch):
        names = [b['name'] for b in batch]
        samples = [b['data'] for b in batch]
        lengths = [len(s) for s in samples]
        size = self.sample_length
        inputs = []
        for b in samples:
            if len(b) == size:
                inputs.append(b)
            else:
                diff = len(b) - size
                start = np.random.randint(0, diff+1)
                end = start + size
                inputs.append(b[start:end])
        batch = torch.stack([torch.from_numpy(b) for b in inputs], 0)
        return batch