import json
import os
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import librosa
import random
import soundfile as sf
from config import win_size, win_shift, fft_num, dataset_path, chunk_length

class To_Tensor(object):
    def __call__(self, x, type):
        if type == 'float':
            return torch.FloatTensor(x)
        elif type == 'int':
            return  torch.IntTensor(x)

class TrainDataset(Dataset):
    def __init__(self, json_dir, batch_size):
        self.json_dir = json_dir
        self.batch_size = batch_size
        json_pos= os.path.join(json_dir, 'train', 'files.json')
        with open(json_pos, 'r') as f:
            json_list = json.load(f)

        minibatch = []
        start = 0
        while True:
            end = min(len(json_list), start+ batch_size)
            minibatch.append(json_list[start:end])
            start = end
            if end == len(json_list):
                break
        self.minibatch = minibatch

    def __len__(self):
        return len(self.minibatch)

    def __getitem__(self, index):
        return self.minibatch[index]


class CvDataset(Dataset):
    def __init__(self, json_dir, batch_size):
        self.json_dir = json_dir
        self.batch_size = batch_size
        json_pos= os.path.join(json_dir, 'dev', 'files.json')
        with open(json_pos, 'r') as f:
            json_list = json.load(f)

        minibatch = []
        start = 0
        while True:
            end = min(len(json_list), start+ batch_size)
            minibatch.append(json_list[start:end])
            start = end
            if end == len(json_list):
                break
        self.minibatch = minibatch

    def __len__(self):
        return len(self.minibatch)

    def __getitem__(self, index):
        return self.minibatch[index]


class TrainDataLoader(object):
    def __init__(self, data_set, batch_size, num_workers=0):
        self.data_loader = DataLoader(dataset= data_set,
                                      batch_size=batch_size,
                                      shuffle=1,
                                      num_workers=num_workers,
                                      collate_fn=self.collate_fn)

    @staticmethod
    def collate_fn(batch):
        feats, labels, frame_mask_list = generate_feats_labels(batch)
        return BatchInfo(feats, labels, frame_mask_list)

    def get_data_loader(self):
        return self.data_loader

def generate_feats_labels(batch):
    batch = batch[0]
    feat_list, label_list, frame_mask_list = [], [], []
    max_frame = 0
    to_tensor = To_Tensor()
    for id in range(len(batch)):
        clean_file_name = '%s_%s.wav' %(batch[id].split('_')[0], batch[id].split('_')[1])
        mix_file_name = '%s.wav'  %(batch[id])
        feat_wav, _= sf.read(os.path.join(dataset_path, 'train', 'mix', mix_file_name))
        label_wav, _ = sf.read(os.path.join(dataset_path, 'train', 'clean', clean_file_name))

        c = np.sqrt(len(feat_wav) / np.sum(feat_wav ** 2.0))
        feat_wav = feat_wav * c
        label_wav = label_wav * c

        if len(feat_wav) > chunk_length:
            wav_start = random.randint(0, len(feat_wav)- chunk_length)
            feat_wav = feat_wav[wav_start:wav_start + chunk_length]
            label_wav = label_wav[wav_start:wav_start + chunk_length]
        # Note that centre setting is given for librosa-based fft for default, so fft_num is added
        frame_num = (len(feat_wav) - win_size + fft_num) // win_shift + 1
        frame_mask_list.append(frame_num)
        feat_x = np.abs(librosa.stft(feat_wav, n_fft=fft_num, hop_length=win_shift, window='hamming').T)
        label_x = np.abs(librosa.stft(label_wav, n_fft=fft_num, hop_length=win_shift, window='hamming').T)
        feat_x, label_x = feat_x[0:frame_num, :], label_x[0:frame_num, :]
        feat_x, label_x = to_tensor(feat_x, 'float'), to_tensor(label_x, 'float')
        feat_list.append(feat_x)
        label_list.append(label_x)

    feat_list = nn.utils.rnn.pad_sequence(feat_list, batch_first=True)
    label_list = nn.utils.rnn.pad_sequence(label_list, batch_first=True)
    return feat_list, label_list, frame_mask_list

def cv_generate_feats_labels(batch):
    batch = batch[0]
    feat_list, label_list, frame_mask_list = [], [], []
    max_frame = 0
    to_tensor = To_Tensor()
    for id in range(len(batch)):
        clean_file_name = '%s_%s.wav' %(batch[id].split('_')[0], batch[id].split('_')[1])
        mix_file_name = '%s.wav' % (batch[id])
        feat_wav, _ = sf.read(os.path.join(dataset_path, 'dev', 'mix', mix_file_name))
        label_wav, _ = sf.read(os.path.join(dataset_path, 'dev', 'clean', clean_file_name))

        c = np.sqrt(len(feat_wav) / np.sum(feat_wav ** 2.0))
        feat_wav = feat_wav * c
        label_wav = label_wav * c

        if len(feat_wav) > chunk_length:
            wav_start = random.randint(0, len(feat_wav) - chunk_length)
            feat_wav = feat_wav[wav_start:wav_start + chunk_length]
            label_wav = label_wav[wav_start:wav_start + chunk_length]
            # Note that centre setting is given for librosa-based fft for default, so fft_num is added
        frame_num = (len(feat_wav) - win_size + fft_num) // win_shift + 1
        frame_mask_list.append(frame_num)
        feat_x = np.abs(librosa.stft(feat_wav, n_fft=fft_num, hop_length=win_shift, window='hamming').T)
        label_x = np.abs(librosa.stft(label_wav, n_fft=fft_num, hop_length=win_shift, window='hamming').T)
        feat_x, label_x = feat_x[0:frame_num, :], label_x[0:frame_num, :]
        feat_x, label_x = to_tensor(feat_x, 'float'), to_tensor(label_x, 'float')
        feat_list.append(feat_x)
        label_list.append(label_x)

    feat_list = nn.utils.rnn.pad_sequence(feat_list, batch_first=True)
    label_list = nn.utils.rnn.pad_sequence(label_list, batch_first=True)
    return feat_list, label_list, frame_mask_list

class CvDataLoader(object):
    def __init__(self, data_set, batch_size, num_workers = 0):

        self.data_loader = DataLoader(dataset=data_set,
                                      batch_size=batch_size,
                                      shuffle=1,
                                      num_workers=num_workers,
                                      collate_fn=self.collate_fn)

    @staticmethod
    def collate_fn(batch):
        feats, labels, frame_mask_list = cv_generate_feats_labels(batch)
        return BatchInfo(feats, labels, frame_mask_list)

    def get_data_loader(self):
        return self.data_loader


class BatchInfo(object):
    def __init__(self, feats, labels, frame_mask_list):
        self.feats = feats
        self.labels = labels
        self.frame_mask_list = frame_mask_list