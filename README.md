# AnuraSet: A large-scale acoustic multi-label dataset for neotropical anuran call classification in passive acoustic monitoring
<div align="center">
<img class="img-fluid" src="assets/dalle_frog.png" alt="img-verification" width="250" height="250">
</div>

We present a large-scale multi-species dataset of acoustics recordings of amphibians anuran from PAM recordings. The dataset comprises 27 hours of herpetologist annotations of 42 different species in different regions of Brazil. The classification task is unique and challenging due to the high species diversity, the long-tailed distribution, and frequent overlapping calls. The dataset, including raw recordings, preprocessing code, and baseline code, is made available to promote collaboration between machine learning researchers and ecologists in solving the classification challenges toward understanding the effects of global change on biodiversity.



## Download

The **Anuraset** is a labeled collection of 93k samples of 3-second-long passive acoustic monitoring recordings organized into 42 neotropical anurans species suitable for multi-label call classification. The dataset can be downloaded as a single .zip file (~10.5 GB):

**[Download Anuraset](https://zenodo.org/record/8056090/files/anuraset.zip?download=1)**

A more thorough description of the dataset is available in the original [paper](https://arxiv.org/abs/2307.06860).

Additionally, we open the raw data and all the annotations (weak and strong labels). You can download all the data in [Zenodo](https://zenodo.org/record/8056090).





## Installation instruction and reproduction of baseline results

1. Install [Conda](http://conda.io/)

2. Clone this repository

```bash
git clone https://github.com/soundclim/anuraset/
```

3. Create an environment and install requirements

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


## Citing this work

If you find the AnuraSet useful for your research, please consider citing it as:

- Cañas, J. S., Toro-Gómez, M. P., Sugai, L. S. M., Restrepo, H. D. B., Rudas, J., Bautista, B. P., ... & Ulloa, J. S. (2023). AnuraSet: A dataset for benchmarking Neotropical anuran calls identification in passive acoustic monitoring. arXiv preprint arXiv:2307.06860.

## Acknowledgments
The authors acknowledge financial support from the intergovernmental Group on Earth Observations (GEO) and Microsoft, under the GEO-Microsoft Planetary Computer Programme (October 2021); São Paulo Research Foundation (FAPESP #2016/25358-3; #2019/18335-5); the National Council for Scientific and Technological Development (CNPq #302834/2020-6; #312338/2021-0, #307599/2021-3); National Institutes for Science and Technology (INCT) in Ecology, Evolution, and Biodiversity Conservation, supported by MCTIC/CNpq (proc. 465610/2014-5), FAPEG (proc. 201810267000023); CNPQ/MCTI/CONFAP-FAPS/PELD No 21/2020 (FAPESC 2021TR386); Comunidad de Madrid (2020-T1/AMB-20636, Atracción de Talento Investigador, Spain) and research projects funded by the European Commission (EAVESTROP–661408, Global Marie S. Curie fellowship, program H2020, EU); and the Ministerio de Economía, Industria y Competitividad (CGL2017-88764-R, MINECO/AEI/FEDER, Spain).


