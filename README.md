# Explainable-Neural-Networks

## Introduction
With the rapid advancement and increased usage of AI, it is important to understand how underlying models generate their outputs to allay fears and suspicion of unfair bias, and enable accountability which may be enforced by regulations and audits.
  
Understandability in AI can broadly be divided into the concepts of:
1. Interpretbability: Refers to intrinsic model parameters and weights that intuitively show how a model determines its generated output. This is a trait that is common in simpler models (eg. coefficients of a linear regressor).
2. Explainablity: Refers to how to take an ML model and explain the behavior in human terms. This concept often involves applying a surrogate to more complex models such as deep neural networks to unravel such blackboxes.

There are other ways to categorise "AI understandability", such as local / global, model-specific / agnostic, and their permutations. Specifically this repository explores incorporating local + model-agnostic explainability into neural networks for image classification of normal / bacterial / viral pneumonia scans.
  
The secondary objective of this repository is to develop pipelines that are interoperable with `Pytorch Lightning`, `Huggingface`, and `Captum` (explainability library). This is motivated by different pretrained model availabilities between Pytorch and Huggingface, specifically `Convnextv2` is only available in Huggingface at the time of developing this repo.
  
## Repo Structure
```
.
├── assets
│   └── images
├── conf
│   └── base
├── data
│   ├── processed
│   │   ├── predict
│   │   ├── test
│   │   │   ├── bacteria
│   │   │   ├── normal
│   │   │   └── virus
│   │   ├── train
│   │   │   ├── bacteria
│   │   │   ├── normal
│   │   │   └── virus
│   │   └── val
│   │       ├── bacteria
│   │       ├── normal
│   │       └── virus
│   └── raw
│       ├── bacteria
│       ├── normal
│       └── virus
├── docker
├── notebooks
└── src
```
  
## Data
This is a dataset of chest X-ray scans to identify whether the lungs are showing normal health, bacterial pneumonia, or viral pneumonia.
  
### Loading
Curl down and unzip the data to the existing repositry using the command:
```bash
# download using
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1VuaSBUw2MFTbobZ2ZcVjugVx-ey88xkF' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1VuaSBUw2MFTbobZ2ZcVjugVx-ey88xkF" -O pneumonia.zip && rm -rf /tmp/cookies.txt
# Unzip using
unzip -q pneumonia.zip .
```
Alternatively download the dataset from [here](https://drive.google.com/file/d/1AOd7h3OWTlBTQc8Gq-gbgIBCqPDxsO6S/view?usp=share_link).
  
### Wrangling
The data source had allocated very few images to the validation folder. Wrangling was done within the `xnn.ipynb` notebook using a combination of `pandas` and `shutils` to (1) reorganise all images according to their classes in the `data/raw` folder, then (2) split them into train / val / test subfolders in the ratio of 80:10:10. 3 images - 1 from each class - were randomly taken and put into the `data/predict` folder.
  
## EDA - FiftyOne
FiftyOne is an open-source tool developed by Voxel51 which visualises image datasets and enables exploration through tagging and filters.
  
From the wrangled data, `annotations_file.csv` was generated containing all the image metadata. This `notebooks/fiftyone.ipynb` contains cells that load this metadata file into a `pandas` dataframe. This notebook is copied into the image generated by `docker/fiftyone.DockerFile`, and will be executed to spin up the FiftyOne service that references images that are bind-mounted during `docker run`.

To launch FiftyOne, please follow the following steps:
  
1. Assuming you have already cloned this Github repo and `cd` into it, build the service using docker:
```bash
docker build -f docker/fiftyone.Dockerfile \
    -t fiftyone:0.1.0 \
    --platform linux/amd64 .  
```
2. Run the service using the command:
```bash
docker run -p 5151:5151 \
    --name fiftyone \
    -v ./data:/data \
    fiftyone:0.1.0    
```
3. Access the fiftyone dashboard using the link [http://localhost:5151](http://localhost:5151) which is hosted on your local machine.
4. Stop the service using the command:
```bash
docker container stop fiftyone   

```
5. To remove the container use the command:
```bash
docker rm fiftyone   
```
  
## Training Pipeline
Documentation WIP - refer to [src/train.py](src/train.py).
  
## Inference + Explainability Pipeline
Script and documentation WIP - refer to `Explainability` section of [notebooks/xnn.ipynb](notebooks/xnn.ipynb)
  
## Streamlit
Script WIP
  
## CI
### Pre-commit hook

More information on pre-commit hook [here](https://pre-commit.com/).
```bash
pre-commit install
```
  
### Github CI pipeline
WIP