# ResEmoteNet - Facial Emotion Recognition

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
 - [Dataset](#dataset)
 - [Usage](#usage)
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

## Dataset

Checkout the dataset for this repository [here](https://github.com/ArnabKumarRoy02/data/tree/e48496150560e3fc28c8977b121edc2f639dd1b6).

The complete dataset can also be found on [Kaggle](https://www.kaggle.com/datasets/arnabkumarroy02/four4all).

## Usage

Run the file.
```bash
cd train_files
python ResEmoteNet_train.py
```

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
   - Valance Estimation:
     - CCC: **0.785** (SoTA - 0.716)
     - RMSE: **0.249** (SoTA - 0.331)
   - Arousal Estimation:
     - CCC: **0.705** (SoTA - 0.642)
     - RMSE: **0.254** (SoTA - 0.305)

## License

This repository is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
