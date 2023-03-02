import os

import pandas as pd 
from tqdm import trange

from anuraset import AnuraSet 
from models import ResNetClassifier

import torch
import torchaudio

from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import Resize

from torchmetrics import MetricCollection
from torchmetrics.classification import (
                                        MultilabelF1Score, 
                                        # https://torchmetrics.readthedocs.io/en/stable/classification/f1_score.html#multilabelf1score
                                        MultilabelROC, 
                                        # https://torchmetrics.readthedocs.io/en/stable/classification/roc.html#multilabelroc 
                                        MultilabelAveragePrecision,
                                        # https://torchmetrics.readthedocs.io/en/stable/classification/average_precision.html#multilabelaverageprecision
                                        MultilabelPrecisionRecallCurve
                                        # https://torchmetrics.readthedocs.io/en/stable/classification/precision_recall_curve.html#multilabelprecisionrecallcurve
                                        )

DIR_VSCODE = "/mnt/batch/tasks/shared/LS_root/mounts/clusters/gpu-baseline/code/Users/jscanass/anuraset/"
os.chdir(DIR_VSCODE)

class_mapping = [
                'SPHSUR', 'BOABIS', 'SCIPER', 'DENNAH', 'LEPLAT', 'RHIICT', 'BOALEP',
                'BOAFAB', 'PHYCUV', 'DENMIN', 'ELABIC', 'BOAPRA', 'DENCRU', 'BOALUN',
                'BOAALB', 'PHYMAR', 'PITAZU', 'PHYSAU', 'LEPFUS', 'DENNAN', 'PHYALB',
                'LEPLAB', 'SCIFUS', 'BOARAN', 'SCIFUV', 'AMEPIC', 'LEPPOD', 'ADEDIP',
                'ELAMAT', 'PHYNAT', 'LEPELE', 'RHISCI', 'SCINAS', 'LEPNOT', 'ADEMAR',
                'BOAALM', 'PHYDIS', 'RHIORN', 'LEPFLA', 'SCIRIZ', 'DENELE', 'SCIALT'
                ]

def save_inferences(test_data, samples, preds, file_name):
    # make sure save directory exists; create if not
    os.makedirs(f'inferences/', exist_ok=True)

    df_inferences = test_data.annotations.copy()
    df_inferences.iloc[samples,8:] = preds
    df_inferences.to_csv(f'inferences/{file_name}.csv',index=False)    
    print(f'inferences saved in: inferences/baseline/{file_name}.csv')

def calculate_metrics(preds, targets, fn_metrics):

    #print(fn_metrics(preds, targets.long()))
    
    return fn_metrics(preds, targets.long())

def evaluate(model, data_loader, loss_fn, device):
    
    sample_idx_all = []
    preds_all = []
    sigmoid = nn.Sigmoid()
    
    num_batches = len(data_loader)
    metric_fn = MultilabelF1Score(num_labels=42).to(device)
    # set model to evaluation mode
    model.eval()
    
    # running averages # correct
    loss_total, metric_total = 0.0, 0.0     # for now, we just log the loss and overall accuracy (OA)

    progressBar = trange(len(data_loader), leave=False)
    with torch.no_grad():        
        for batch_idx, (input, target, index) in enumerate(data_loader):
            # put data and labels on device
            input, target = input.to(device), target.to(device)

            prediction = sigmoid(model(input))

            loss = loss_fn(prediction, target)

            # log statistics
            loss_total += loss.item()
            # log metrics
            metric = metric_fn(prediction, target)
            metric_total += metric.item()
            #collecting infereces
            sample_idx_all.extend(index.tolist())
            preds_all.extend(prediction.tolist())
            
            progressBar.set_description(
                '[Evaluation] Loss: {:.4f}; F1-score macro: {:.4f}'.format(
                    loss_total/(batch_idx+1),
                    metric_total/(batch_idx+1),
                )
            )
            progressBar.update(1)
        progressBar.close() 
    loss_total /= num_batches
    metric_total /= num_batches
            
    return sample_idx_all, preds_all
    
if __name__ == "__main__":
    
    ANNOTATIONS_FILE = 'baseline/metadata_sample1000.csv'
    AUDIO_DIR =  "/mnt/batch/tasks/shared/LS_root/mounts/clusters/gpu-baseline/code/Users/jscanass/anuraset_pruebas/data/datasets/anurasetv3/audio/"
    SAMPLE_RATE = 22050 
    NUM_SAMPLES = SAMPLE_RATE*3
    #ANNOTATIONS_FILE = "/mnt/batch/tasks/shared/LS_root/mounts/clusters/gpu-baseline/code/Users/jscanass/anuraset_pruebas/data/datasets/anurasetv3/metadata.csv" 
    BATCH_SIZE = 16
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    print(f"Using device {device}")

    # load back the model
    model_instance = ResNetClassifier(model_type='resnet50',
                            ).to(device)
    state_dict = torch.load("baseline/model_states/resnet50_4sites/resnet50_final.pth")
    model_instance.load_state_dict(state_dict)

    # load urban sound dataset dataset
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )
    test_transform = nn.Sequential(                           # Transforms. Here's where we could add data augmentation (see Björn's lecture on August 11).
            mel_spectrogram,                                            # convert to a spectrogram
            torchaudio.transforms.AmplitudeToDB(),
            Resize([224, 448]),
                                )

    test_data = AnuraSet(
            annotations_file=ANNOTATIONS_FILE, 
            audio_dir=AUDIO_DIR, 
            transformation=test_transform,
            train=False
                            )
    print(f"There are {len(test_data)} samples in the test set.")

    test_dataloader = DataLoader(test_data, 
                                 batch_size=BATCH_SIZE,
                                 shuffle=False)
    
    loss_fn = nn.BCEWithLogitsLoss()
    
    samples, preds = evaluate(model= model_instance, 
                                data_loader= test_dataloader,
                                loss_fn= loss_fn,
                                device=device)
    
    save_inferences(test_data, samples, preds,file_name='test')

    NUM_SAMPLES = 42
    
    metric_collection = MetricCollection([
        MultilabelF1Score(num_labels=NUM_SAMPLES, average=None, thresholds=0.5).to(device),
        MultilabelAveragePrecision(num_labels=NUM_SAMPLES, average=None, thresholds=None).to(device),
        MultilabelROC(num_labels=NUM_SAMPLES, thresholds=None).to(device),
        MultilabelPrecisionRecallCurve(num_labels=NUM_SAMPLES, thresholds=None).to(device)
                                    ])

    targets = test_data.annotations.iloc[samples,8:].values
    
    result_ev = calculate_metrics(torch.Tensor(preds).to(device), 
                                  torch.Tensor(targets).to(device), 
                                  metric_collection)
    
    print('Done!')    

   

                    
