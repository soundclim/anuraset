import os
import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader
from torchsummary import summary 

from anuraset import AnuraSet
from models import CNNetwork_2D
from tqdm import tqdm

BATCH_SIZE = 128
EPOCHS = 3
LEARNING_RATE = 0.01

DIR_VSCODE = "/mnt/batch/tasks/shared/LS_root/mounts/clusters/gpu-baseline/code/Users/jscanass/"
os.chdir(DIR_VSCODE)

#ANNOTATIONS_FILE = 'metadata_sample1000.csv' 
ANNOTATIONS_FILE = "anuraset_pruebas/data/datasets/anurasetv3/metadata.csv"
AUDIO_DIR =  "anuraset_pruebas/data/datasets/anurasetv3/audio/"
SAMPLE_RATE = 22050
NUM_SAMPLES = SAMPLE_RATE*3

MODEL = 'CNN_2NETWORK'
DATASET_SUBSET = 'training_for_test'
VALIDATION_FOLD = None


def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    return train_dataloader


def train_single_epoch(model, data_loader, loss_fn, optimiser, device):
    ## running averages
    ## loss_total, oa_total = 0.0, 0.0    
    size = len(data_loader.dataset)
    for idx, (input, target) in enumerate(data_loader):
        input, target = input.to(device), target.to(device)

        # calculate loss
        prediction = model(input)
        loss = loss_fn(prediction, target)

        # backpropagate error and update weights
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        if idx % 100 == 0:
            loss, current = loss.item(), idx * len(input)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    print(f"loss single epoch: {loss.item()}")

def train(model, data_loader, loss_fn, optimiser, device, epochs):
    for i in tqdm(range(epochs)):
        print(f"Epoch {i+1}")
        train_single_epoch(model, data_loader, loss_fn, optimiser, device)
        print("---------------------------")
    print("Finished training")

if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")

    # Define Transformation

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

    # TODO: call val not test!
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

    train_dataloader = create_data_loader(training_data, BATCH_SIZE)

    cnn = CNNetwork_2D().to(device)
    summary(cnn.cuda(), (1, 64, 130))
    
    # initialise loss funtion + optimiser
    #loss_fn = torch.nn.CrossEntropyLoss()
    loss_fn = nn.BCEWithLogitsLoss() # Multilabel loss
    optimiser = torch.optim.Adam(cnn.parameters(),
                                 lr=LEARNING_RATE)

    # train model
    train(cnn, train_dataloader, loss_fn, optimiser, device, EPOCHS)

    # save model
    torch.save(cnn.state_dict(), "cnn.pth")
    print("Trained feed forward net saved at cnn.pth")
