# ResEmoteNet: Bridging Accuracy and Loss Reduction in Facial Emotion Recognition

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/resemotenet-bridging-accuracy-and-loss/facial-expression-recognition-on-affectnet)](https://paperswithcode.com/sota/facial-expression-recognition-on-affectnet?p=resemotenet-bridging-accuracy-and-loss)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/resemotenet-bridging-accuracy-and-loss/facial-expression-recognition-on-fer2013)](https://paperswithcode.com/sota/facial-expression-recognition-on-fer2013?p=resemotenet-bridging-accuracy-and-loss)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/resemotenet-bridging-accuracy-and-loss/facial-expression-recognition-on-raf-db)](https://paperswithcode.com/sota/facial-expression-recognition-on-raf-db?p=resemotenet-bridging-accuracy-and-loss)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/resemotenet-bridging-accuracy-and-loss/facial-expression-recognition-fer-on-expw)](https://paperswithcode.com/sota/facial-expression-recognition-fer-on-expw?p=resemotenet-bridging-accuracy-and-loss)

A new network that helps in extracting facial features and predict the emotion labels.

The emotion labels in this project are:
 - Happiness 😀
 - Surprise 😦
 - Anger 😠
 - Sadness ☹️
 - Disgust 🤢
 - Fear 😨
 - Neutral 😐


## Table of Content:

 - [Installation](#installation)
 - [Usage](#usage)
 - [Checkpoints](#checkpoints)
 - [Results](#results)
 - [License](#license)


## Installation

1. Create a Conda environment.
```bash
conda create --n "fer"
conda activate fer
```

2. Install Python v3.8 using Conda.
```bash
conda install python=3.8
```

3. Clone the repository.
```bash
git clone https://github.com/ArnabKumarRoy02/ResEmoteNet.git
```

4. Install the required libraries.
```bash
pip install -r requirement.txt
```

## Usage

Run the file.
```bash
cd train_files
python ResEmoteNet_train.py
```

## Checkpoints
All of the checkpoint models for FER2013, RAF-DB and AffectNet-7 can be found [here](https://drive.google.com/drive/folders/1Daxa6d1-XFxxpg6dyxYl4V-anfiHwtqK?usp=sharing).

## Results

 - FER2013:
   - Testing Accuracy: **79.79%** (SoTA - 76.82%)
 - CK+:
   - Testing Accuracy: **100%** (SoTA - 100%)
 - RAF-DB:
   - Testing Accuracy: **94.76%** (SoTA - 92.57%)
 - FERPlus:
   - Testing Accuracy: 91.64% (SoTA - **95.55%**)
 - AffectNet (7 emotions):
   - Testing Accuracy: **72.93%** (SoTA - 69.4%)
 - ExpW:
   - Testing Accuracy: **75.67%**

## License

This repository is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

Cite our paper:
```text
@ARTICLE{10812829,
  author={Roy, Arnab Kumar and Kathania, Hemant Kumar and Sharma, Adhitiya and Dey, Abhishek and Ansari, Md. Sarfaraj Alam},
  journal={IEEE Signal Processing Letters}, 
  title={ResEmoteNet: Bridging Accuracy and Loss Reduction in Facial Emotion Recognition}, 
  year={2024},
  pages={1-5},
  keywords={Emotion recognition;Feature extraction;Convolutional neural networks;Accuracy;Training;Computer architecture;Residual neural networks;Facial features;Face recognition;Facial Emotion Recognition;Convolutional Neural Network;Squeeze and Excitation Network;Residual Network},
  doi={10.1109/LSP.2024.3521321}
}
```
