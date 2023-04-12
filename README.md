# AnuraSet: A large-scale acoustic multi-label dataset for neotropical anuran call classification in passive acoustic monitoring
<div align="center">
<img class="img-fluid" src="assets/dalle_frog.png" alt="img-verification" width="250" height="250">
</div>

We present a large-scale multi-species dataset of acoustics recordings of amphibians anuran from PAM recordings. The dataset comprises 27 hours of herpetologist annotations of 42 different species in different regions of Brazil. The classification task is unique and challenging due to the high species diversity, the long-tailed distribution, and frequent overlapping calls. The dataset, including raw recordings, preprocessing code, and baseline code, is made available to promote collaboration between machine learning researchers and ecologists in solving the classification challenges toward understanding the effects of global change on biodiversity.




## TODO
- [x] Download dataset using link and fetcher
- [x] Baseline Code 
- [ ] Notebook with analysis of baseline results
- [ ] Preprocessing Code
- [ ] Links to download raw audio and annotations
- [ ] In deep explanation of data format
- [ ] Evaluation explanation
- [ ] Explicit assignation of LICENSE
- [ ] Add .bib with publication


## Download

The **Anuraset** is a labeled collection of 93k samples of 3-second-long passive acoustic monitoring recordings organized into 42 neotropical anurans species suitable for multi-label call classification. The dataset can be downloaded as a single .zip file (~10.5 GB):

**[Download Anuraset](https://chorus.blob.core.windows.net/public/anurasetv3.zip)**

A more thorough description of the dataset is available in the original [paper](http://github.com).

Additionally we open the [raw data](http://github.com) and the [annotations](http://github.com). (TODO, check [this](https://github.com/visipedia/caltech-fish-counting/blob/main/README.md#data-download)) 





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


4. Download dataset 

```bash
python datasets/fetcher.py
```

5. Train 

```bash
python baseline/train.py --config baseline/configs/exp_resnet18.yaml
```

6. Inference

```bash
python baseline/evaluate.py --config  baseline/configs/exp_resnet18.yaml
```

7. Visualize results: Run notebook  (TODO)


## Evaluation Procedure (TODO)

...

## Data Format (TODO)

### Audio
- `audio/*.wav`

  2000 audio recordings in WAV format (5 seconds, 44.1 kHz, mono) with the following naming convention:
  
  `{FOLD}-{CLIP_ID}-{TAKE}-{TARGET}.wav`
  
  - `{FOLD}` - index of the cross-validation fold,
  - `{CLIP_ID}` - ID of the original Freesound clip,
  - `{TAKE}` - letter disambiguating between different fragments from the same Freesound clip,
  - `{TARGET}` - class in numeric format [0, 49].
### Metadata

- `metadata.csv`

  CSV file with the following structure:
  
  | <sub>filename</sub> | <sub>fold</sub> | <sub>target</sub> | <sub>category</sub> | <sub>esc10</sub> | <sub>src_file</sub> | <sub>take</sub> |
  | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
  
  The `esc10` column indicates if a given file belongs to the *ESC-10* subset (10 selected classes, CC BY license).
  
- [`meta/esc50-human.xlsx`](meta/esc50-human.xlsx)

  Additional data pertaining to the crowdsourcing experiment (human classification accuracy).



## License (TODO)

...


## Citing this work (TODO)

..-

