import os
import torch
import torchaudio
import pandas as pd

from torch.utils.data import Dataset

class AnuraSet(Dataset):
    """ AnuraSet🐸: A dataset for bioacoustic classification of tropical anurans

    Args:
        annotations_file (string): path of the metadata csv table with labels of 
            AnuraSet and audio samples information
        audio_dir(string): path of the folder with audio samples of the AnuraSet
            associated with the metadata table
        transformation (callable?): L A function/transform that takes audios before 
            feature extraction and reutnrs a transformed version.This tranformantions 
            include melspectogram and augmentations.
        target_sample_rate (int): sample rate to use in an internal preprocessing
            to modify audio samples 
        num_samples (int): number of samples to use in an internal preprocessing
            of the class to adjust the size of the samples 
        train (bool): If True, creates dataset from using 'train' samples in the 
                'subset' column of the metadata
    """
    
    def __init__(self, 
                 annotations_file, 
                 audio_dir, 
                 transformation,
                 target_sample_rate,
                 num_samples,
                 device,
                 train=True,
                ):

        df = pd.read_csv(annotations_file)
        
        if train:
            df = df[df['subset']=='train']
        else:
            df = df[df['subset']=='test']
        
        self.annotations = df 
        self.audio_dir = audio_dir
        self.device = device
        self.transformation = transformation.to(self.device)
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples
        
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self,index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, sr = torchaudio.load(audio_sample_path)
        signal = signal.to(self.device)
        signal = self._resample_if_necessary(signal, sr)
        signal = self._mix_down_if_necessary(signal)
        signal = self._cut_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)
        signal = self.transformation(signal)
        return signal, label
    
    def _cut_if_necessary(self, signal):
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal
        
    def _right_pad_if_necessary(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal
    
    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate).cuda()
            signal = resampler(signal)
        return signal
    
    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal
    
    def _get_audio_sample_path(self, index):
        path = os.path.join(self.audio_dir, self.annotations.iloc[index, 0])
        return path
    
    def _get_audio_sample_label(self, index):
        return torch.Tensor(self.annotations.iloc[index,8:])


if __name__ == "__main__":
    # Modify to your parent directory
    #DIR_VSCODE = "/mnt/batch/tasks/shared/LS_root/mounts/clusters/gpu-baseline/code/Users/jscanass/"
    os.chdir(DIR_VSCODE)
    ANNOTATIONS_FILE = 'anurasetv3/ametadata.csv' 
    AUDIO_DIR =  "anurasetv3/audio/"
    SAMPLE_RATE = 22050
    NUM_SAMPLES = SAMPLE_RATE*3
    
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device {device}")
    
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )
    
    training_data = AnuraSet(
        annotations_file=ANNOTATIONS_FILE, 
        audio_dir=AUDIO_DIR, 
        transformation=mel_spectrogram,
        target_sample_rate=SAMPLE_RATE,
        num_samples=NUM_SAMPLES,
        device=device,
        train=True
                        )
    print(f"There are {len(training_data)} samples in the training set.")

    test_data = AnuraSet(
        annotations_file=ANNOTATIONS_FILE, 
        audio_dir=AUDIO_DIR, 
        transformation=mel_spectrogram,
        target_sample_rate=SAMPLE_RATE,
        num_samples=NUM_SAMPLES,
        device=device,
        train=False
                        )
    print(f"There are {len(test_data)} samples in the test set.")
