import os
import torch
import torchaudio
import pandas as pd

from torch.utils.data import Dataset

class AnuraSet(Dataset):
    """ AnuraSet: A dataset for bioacoustic classification of tropical anurans

    Args:
        annotations_file (string): path of the metadata csv table with labels of 
            AnuraSet and audio samples information
        audio_dir(string): path of the folder with audio samples of the AnuraSet
            associated with the metadata table
        transformation (callable?): L A function/transform that takes audios before 
            feature extraction and returns a transformed version.This tranformantions 
            include melspectogram and augmentations.
        device (string): if using cuda (GPU) or CPU
        train (bool): If True, creates dataset from using 'train' samples in the 
                'subset' column of the metadata
    """
    
    def __init__(self,
                 annotations_file, 
                 audio_dir, 
                 transformation,
                 train=True,
                ):

        if isinstance(annotations_file, str):
            df = pd.read_csv(annotations_file)
        else:
            df = annotations_file.copy()
        
        if train:
            df = df[df["subset"]=="train"]
        else:
            df = df[df["subset"]=="test"]
        
        self.annotations = df 
        self.audio_dir = audio_dir
        self.transformation = transformation

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self,index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, _ = torchaudio.load(audio_sample_path)
        
        signal = self.transformation(signal)
        return signal, label, index
    
    def _get_audio_sample_path(self, index):
        path = os.path.join(self.audio_dir, self.annotations.iloc[index, 0])
        return path
    
    def _get_audio_sample_label(self, index):
        return torch.Tensor(self.annotations.iloc[index,8:])
