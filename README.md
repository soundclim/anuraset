# AnuraSet: AnuraSet: A large-scale acoustic multi-label dataset for neotropical anuran call classification in passive acoustic monitoring

<!-- <img src="assets/examples.gif" width=100%> -->
![DALL·E 2022-10-23 22.58.22 -  frog shouting because of hot temperature and acting weird in the middle of a  rainforest , Henri Rousseau painting ](assets/dalle_frog.png)

The timing and intensity of calling activity in anuran amphibians, which has a central role in sexual selection and reproduction, are largely controlled by climatic conditions such as environmental temperature and humidity. Therefore, climate change is predicted to induce shifts in calling behavior and breeding phenology, species traits that can be tracked using passive acoustic monitoring (PAM). To construct robust algorithms that allow classifying species calls in a long-term monitoring program, it is fundamental to design adequate datasets and benchmarks in the wild.  We present a large-scale multi-species dataset of acoustics recordings of amphibians anuran from PAM recordings. The dataset comprises 27 hours of herpetologist annotations of 42 different species in different regions of Brazil. The classification task is unique and challenging due to the high species diversity, the long-tailed distribution, and frequent overlapping calls. We present a characterization of the challenges and a baseline model for the goals of the monitoring program. The dataset, including raw recordings, preprocessing code, and baseline code, is made available to promote collaboration between machine learning researchers and ecologists in solving the classification challenges toward understanding the effects of global change on biodiversity.

## Download

The **Anuraset** is a labeled collection of 93k samples of 3-second-long passive acoustic monitoring recordings organized into 42 neotropical anurans species suitable for multi-label call classification. The dataset can be downloaded as a single .zip file (~11 GB):

**[Download Anuraset](https://chorus.blob.core.windows.net/public/anurasetv3.zip)**

A more thorough description of the dataset is available in the original [paper](http://karol.piczak.com/papers/Piczak2015-ESC-Dataset.pdf) with some supplementary materials on GitHub: **[AnuraSet: AnuraSet: A large-scale acoustic multi-label dataset for neotropical anuran call classification in passive acoustic monitoring](https://github.com/karoldvl/paper-2015-esc-dataset)**.


## Reproduce baseline results

### Installation instructions

1. Install [Conda](http://conda.io/)

2. Create environment and install requirements

```bash
conda create -n anuraset_env python=3.8 -y
conda activate chorus_env
onda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
pip install -r requirements.txt
```
3. Download dataset 

```bash
python dataset/fetcher.py
```

4. Train (TODO:CHECK)

```bash
python baseline/train.py --config 
configs/exp_resnet18.yaml
```

5. Inference (TODO:CHECK)

```bash
python baseline/evaluate.py --config 
configs/exp_resnet18.yaml
```
6. Visualize results: Run notebook  (TODO)


## Benchmark Results

If you know of some other reference, you can message me or open a Pull Request directly.


| <sub>Title</sub> | <sub>Notes</sub> | <sub>F1-score (Macro)</sub> | <sub>Paper</sub> | <sub>Code</sub> |
| :--- | :--- | :--- | :--- | :--- |
| <sub>**HTS-AT: A Hierarchical Token-Semantic Audio Transformer for Sound Classification and Detection**</sub> | <sub>Transformer model with hierarchical structure and token-semantic modules</sub> | <sub>97.00%</sub> | <sub>[chen2022](https://arxiv.org/pdf/2202.00874.pdf)</sub> | <a href="https://github.com/RetroCirce/HTS-Audio-Transformer">:scroll:</a>   |
| <sub>**CLAP: Learning Audio Concepts From Natural Language Supervision**</sub> | <sub>CNN model pretrained by natural language supervision</sub> | <sub>96.70%</sub> | <sub>[elizalde2022](https://arxiv.org/pdf/2206.04769.pdf)</sub> |  |
| <sub>**AST: Audio Spectrogram Transformer**</sub> | <sub>Pure Attention Model Pretrained on AudioSet</sub> | <sub>95.70%</sub> | <sub>[gong2021](https://arxiv.org/pdf/2104.01778.pdf)</sub> | <a href="https://github.com/YuanGongND/ast">:scroll:</a> |
|  |  |
| <sub>**Audio Event and Scene Recognition: A Unified Approach using Strongly and Weakly Labeled Data**</sub> | <sub>Combination of weakly labeled data (YouTube) with strong labeling (ESC-10) for Acoustic Event Detection</sub> | <sub>N/A</sub> | <sub>[kumar2016a](https://arxiv.org/pdf/1611.04871.pdf)</sub> |  |


## Repository content

- [`audio/*.wav`](audio/)

  2000 audio recordings in WAV format (5 seconds, 44.1 kHz, mono) with the following naming convention:
  
  `{FOLD}-{CLIP_ID}-{TAKE}-{TARGET}.wav`
  
  - `{FOLD}` - index of the cross-validation fold,
  - `{CLIP_ID}` - ID of the original Freesound clip,
  - `{TAKE}` - letter disambiguating between different fragments from the same Freesound clip,
  - `{TARGET}` - class in numeric format [0, 49].

- [`meta/esc50.csv`](meta/esc50.csv)

  CSV file with the following structure:
  
  | <sub>filename</sub> | <sub>fold</sub> | <sub>target</sub> | <sub>category</sub> | <sub>esc10</sub> | <sub>src_file</sub> | <sub>take</sub> |
  | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
  
  The `esc10` column indicates if a given file belongs to the *ESC-10* subset (10 selected classes, CC BY license).
  
- [`meta/esc50-human.xlsx`](meta/esc50-human.xlsx)

  Additional data pertaining to the crowdsourcing experiment (human classification accuracy).


## License

The dataset is available under the terms of the [Creative Commons Attribution Non-Commercial license](http://creativecommons.org/licenses/by-nc/3.0/).

A smaller subset (clips tagged as *ESC-10*) is distributed under CC BY (Attribution).

Attributions for each clip are available in the [ LICENSE file](LICENSE).


## Citing

<a href="http://karol.piczak.com/papers/Piczak2015-ESC-Dataset.pdf"><img src="https://img.shields.io/badge/download%20paper-PDF-ff69b4.svg" alt="Download paper in PDF format" title="Download paper in PDF format" align="right" /></a>

If you find this dataset useful in an academic setting please cite:

> K. J. Piczak. **ESC: Dataset for Environmental Sound Classification**. *Proceedings of the 23rd Annual ACM Conference on Multimedia*, Brisbane, Australia, 2015.
> 
> [DOI: http://dx.doi.org/10.1145/2733373.2806390]

    @inproceedings{piczak2015dataset,
      title = {{ESC}: {Dataset} for {Environmental Sound Classification}},
      author = {Piczak, Karol J.},
      booktitle = {Proceedings of the 23rd {Annual ACM Conference} on {Multimedia}},
      date = {2015-10-13},
      url = {http://dl.acm.org/citation.cfm?doid=2733373.2806390},
      doi = {10.1145/2733373.2806390},
      location = {{Brisbane, Australia}},
      isbn = {978-1-4503-3459-4},
      publisher = {{ACM Press}},
      pages = {1015--1018}
    }

## Caveats

Please be aware of potential information leakage while training models on *ESC-50*, as some of the original Freesound recordings were already preprocessed in a manner that might be class dependent (mostly bandlimiting). Unfortunately, this issue went unnoticed when creating the original version of the dataset. Due to the number of methods already evaluated on *ESC-50*, no changes rectifying this issue will be made in order to preserve comparability.

