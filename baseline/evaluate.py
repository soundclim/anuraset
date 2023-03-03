import os
import yaml
import argparse
from tqdm import trange

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

from anuraset import AnuraSet 
from models import ResNetClassifier

# How to modify when there are less classes?
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
    os.makedirs(f'baseline/results/{file_name}', exist_ok=True)

    df_inferences = test_data.annotations.copy()
    df_inferences.iloc[samples,8:] = preds
    df_inferences.to_csv(f'baseline/results/{file_name}inferences.csv',index=False)    
    print(f'Inferences saved in: baseline/results/{file_name}inferences.csv')

def save_metrics(cfg, metrics):
    # make sure save directory exists; create if not
    folder_name = cfg['folder_name']
    os.makedirs(f'baseline/results/{folder_name}', exist_ok=True)
    # add metris
    # also save config file if not present
    with open(f'baseline/model_states/{folder_name}config.yaml') as f:
        list_doc = yaml.safe_load(f)
        
    list_doc['MultilabelAveragePrecision'] = metrics['MultilabelAveragePrecision'].to('cpu').numpy().tolist()
    list_doc['MultilabelF1Score'] = metrics['MultilabelF1Score'].to('cpu').numpy().tolist()
    list_doc['class_mapping'] = class_mapping

    with open(f'baseline/results/{folder_name}metrics.yaml', "w") as f:
        yaml.dump(list_doc, f, default_flow_style=False)

def calculate_metrics(preds, targets, fn_metrics):
    #print(fn_metrics(preds, targets.long()))
    return fn_metrics(preds, targets.long())

def evaluate(model, data_loader, loss_fn, metric_fn, device):
    
    sample_idx_all = []
    preds_all = []
    sigmoid = nn.Sigmoid()
    
    num_batches = len(data_loader)
    # set model to evaluation mode
    model.eval()
    
    # running averages # correct
    loss_total, metric_total = 0.0, 0.0     # for now, we just log the loss and overall accuracy (OA)
    size = len(data_loader.dataset) 
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
                '[Evaluation] Loss: {:.4f}; F1-score macro: {:.4f} [{:>5d}/{:>5d}]'.format(
                    loss_total/(batch_idx+1),
                    metric_total/(batch_idx+1),
                    (batch_idx + 1) * len(input),
                    size
                )
            )
            progressBar.update(1)
        progressBar.close() 
    loss_total /= num_batches
    metric_total /= num_batches
            
    return sample_idx_all, preds_all
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Domain shift.')
    parser.add_argument('--config', help='Path to config file', default='configs/exp_resnet152.yaml')
    args = parser.parse_args()

    # load config
    print(f'Using config "{args.config}"')
    cfg = yaml.safe_load(open(args.config, 'r'))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = "cpu"
    print(f"Using device {device}")

    # Define Transformation

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        n_fft=512,
        hop_length=128,
        n_mels=128
    )
    test_transform = nn.Sequential(                           # Transforms. Here's where we could add data augmentation (see Björn's lecture on August 11).
            mel_spectrogram,                                            # convert to a spectrogram
            torchaudio.transforms.AmplitudeToDB(),
            Resize([224, 448]),
                                )

    ANNOTATIONS_FILE = os.path.join(
        cfg['data_root'],
        'metadata.csv'
    )
    
    AUDIO_DIR = os.path.join(
        cfg['data_root'],
        'audio'
    )
    
    test_data = AnuraSet(
            annotations_file=ANNOTATIONS_FILE, 
            audio_dir=AUDIO_DIR, 
            transformation=test_transform,
            train=False
                            )
    print(f"There are {len(test_data)} samples in the test set.")    

    test_dataloader = DataLoader(test_data, 
                                 batch_size=cfg['batch_size'],
                                 shuffle=False)
    
    multi_label=cfg['multilabel']   
    if multi_label:
        loss_fn = nn.BCEWithLogitsLoss()
    else:
        loss_fn = nn.CrossEntropyLoss()

    metric_fn = MultilabelF1Score(num_labels=cfg['num_classes']).to(device)
    
    # load back the model
    model_instance = ResNetClassifier(model_type=cfg['model_type'],
                            ).to(device)
    state_dict = torch.load('baseline/model_states/'+cfg['folder_name']+'final.pth')
    model_instance.load_state_dict(state_dict)

    samples, preds = evaluate(model_instance, 
                                test_dataloader,
                                loss_fn,
                                metric_fn,
                                device)
    
    save_inferences(test_data, samples, preds, file_name=cfg['folder_name'])
    
    metric_collection = MetricCollection([
        MultilabelF1Score(num_labels=cfg['num_classes'], average=None, thresholds=0.5).to(device),
        MultilabelAveragePrecision(num_labels=cfg['num_classes'], average=None, thresholds=None).to(device),
        MultilabelROC(num_labels=cfg['num_classes'], thresholds=None).to(device),
        MultilabelPrecisionRecallCurve(num_labels=cfg['num_classes'], thresholds=None).to(device)
                                    ])

    targets = test_data.annotations.iloc[samples,8:].values
    
    result_ev = calculate_metrics(torch.Tensor(preds).to(device), 
                                  torch.Tensor(targets).to(device), 
                                  metric_collection)
    
    save_metrics(cfg, result_ev)

    print('Done!')    

   

                    
