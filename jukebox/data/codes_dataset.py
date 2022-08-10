import librosa
import math
import numpy as np
import torch
import itertools
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

    def load_label_offset(self, label_path, inds):
        with open(label_path) as f:
            lengths = []
            code_lengths = [len(line.encode("utf-8")) for line in f]
            offsets = list(itertools.accumulate([0] + code_lengths))
            offsets = [(offsets[i], offsets[i+1]) for i in inds]
        return offsets


    def init_dataset(self, hps):
        # Load list of files and starts/durations
        len_f = open(f'{self.data_file}.len')
        lengths = []
        for line in len_f:
            lengths.append(int(line.strip()))
        len_f.close()
        keep = []
        for i in range(len(lengths)):
            if lengths[i] < self.min_length:
                continue
            if lengths[i] > self.max_length:
                continue
            keep.append(i)
        self.offsets = self.load_label_offset(self.data_file+'.label2', keep)
        cache = dist.get_rank() % 8 == 0 if dist.is_available() else True

        """
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
        """

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
        return len(self.offsets)

    def __getitem__(self, item):
        with open(f"{self.data_file}"+'.label2') as f:
            offset_s, offset_e = self.offsets[item]
            f.seek(offset_s)
            label = f.read(offset_e - offset_s)
            data = np.array(list(map(int, label.split())))
        return {'name': item, 'data': data}

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