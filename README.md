·# AnuraSet: A large-scale acoustic multi-label dataset for neotropical anuran call classification in passive acoustic monitoring
<div align="center">
<img class="img-fluid" src="assets/dalle_frog.png" alt="img-verification" width="250" height="250">
</div>

We present a large-scale multi-species dataset of acoustics recordings of amphibians anuran from PAM recordings. The dataset comprises 27 hours of herpetologist annotations of 42 different species in different regions of Brazil. The classification task is unique and challenging due to the high species diversity, the long-tailed distribution, and frequent overlapping calls. The dataset, including raw recordings, preprocessing code, and baseline code, is made available to promote collaboration between machine learning researchers and ecologists in solving the classification challenges toward understanding the effects of global change on biodiversity.



## Download

The **Anuraset** is a labeled collection of 93k samples of 3-second-long passive acoustic monitoring recordings organized into 42 neotropical anurans species suitable for multi-label call classification. The dataset can be downloaded as a single .zip file (~10.5 GB):

**[Download Anuraset](https://zenodo.org/record/8056090/files/anuraset.zip?download=1)**

A more thorough description of the dataset is available in the original [paper](http://github.com).

Additionally, we open the raw data and all the annotations (weak and strong labels). You can download all the data in [Zenodo](https://zenodo.org/record/8056090).





## Installation instruction and reproduction of baseline results

1. Install [Conda](http://conda.io/)

2. Clone this repository

```bash
git clone https://github.com/soundclim/anuraset/
```

3. Create environment and install requirements

```bash
cd anuraset
conda create -n anuraset_env python=3.8 -y
conda activate anuraset_env
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
pip install -r requirements.txt
```

> **Notes**
> * The installation of dependencies where tested on Azure. If you want to run locally, you might have to change the way you install PyTorch. Check the [PyTorch official webpage](https://pytorch.org/get-started/locally/) for installation instruction on specific platforms.
> * For **macOS** you might need to install [chardet: The Universal Character Encoding Detector](https://pypi.org/project/chardet/) with pip.


4. Download the data directly from Zenodo 

5. Train 

```bash
python baseline/train.py --config baseline/configs/exp_resnet18.yaml
```

6. Inference

```bash
python baseline/evaluate.py --config  baseline/configs/exp_resnet18.yaml
```

7. Visualize results: Run notebook  (TODO)


## Citing this work (TODO)

..-

