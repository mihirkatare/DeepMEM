# DeepMEM
[![GitHub Project](https://img.shields.io/badge/GitHub--blue?style=social&logo=GitHub)](https://github.com/mihirkatare/DeepMEM)
[![NSF Award Number](https://img.shields.io/badge/NSF-1836650-blue.svg)](https://nsf.gov/awardsearch/showAward?AWD_ID=1836650)

[![PyPI version](https://badge.fury.io/py/deepmem.svg)](https://badge.fury.io/py/deepmem)
[![Supported Python versions](https://img.shields.io/pypi/pyversions/deepmem.svg)](https://pypi.org/project/deepmem/)

[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/mihirkatare/DeepMEM/main.svg)](https://results.pre-commit.ci/latest/github/mihirkatare/DeepMEM/main)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This repository contains the code that was used for the IRIS-HEP Fellowship project: **[Deep Learning Implementations for Sustainable Matrix Element Method Calculations](https://iris-hep.org/fellows/mihirkatare.html)**.

IRIS-HEP Fellowship Presentation:
[Deep Learning for the Matrix Element Method](https://indico.cern.ch/event/1071423/contributions/4505210/attachments/2329865/3969981/Final%20Presentation%20-%20Mihir%20Katare.pdf)

---
## **Project Description**
The Matrix Element Method (MEM) is a powerful statistical analysis technique for experimental and simulated particle physics data. It has several benefits over black-box methods like neural networks, owing to its transparent and interpretable results. The drawback of MEM; however, is the significant amount of computationally intensive calculations involved in its execution, which impedes research that relies on it. This project aims to improve the viability of MEM, by implementing deep learning techniques to accurately and efficiently approximate MEM calculations - providing the much required speedup over the traditional approach, while preserving its interpretability. The implemented model can be used as a good approximation during the exploratory phase of research, and the full ME calculations can be used for the final runs, making the workflow for research involving MEM much more efficient.

---
## Installation

### From PyPI

`deepmem` is distributed on [PyPI](https://pypi.org/project/deepmem/) and can be installed in your Python virtual environment with `pip`

```console
$ python -m pip install deepmem
```

### From version control

`deepmem` can also be installed from source by cloning a version of this repository and then from the top level of the repository locally installing in your Python virtual environment with `pip`

```console
$ python -m pip install .
```

Developers will probably want to create an editable install so their code changes are automatically picked up without having to reinstall

```console
$ python -m pip install --editable .
```

---
## **Requirements for deployment**
The code is stable in python 3.8.5. Use the `requirements.txt` file to install the dependencies using `pip` or package installer of choice.

---
## **Explanation of inputs**
To run this code simply run `deepmem` with the required shell arguments and input file modifications.


The code takes two types of inputs:
- Through an JSON input file (e.g.: `input_files/input.json`): This is supposed to have inputs that do not require dynamimc user changes, i.e, during the training/optimization phase these will remain mostly constant.
- Through shell arguments (e.g. deepmem **`--device=0`**): These are inputs that would usually require dynamic user changes before running the code.

#### **Examples**:
Go through the input file at `input_files/input.json` and modify the paths to the data, where to save and load models, scalers, etc.
Then run the following code to train the model on the options in the `input_files/input.json` using the **0th numbered CUDA GPU** for **50 epochs**:

```console
$ deepmem --device=0 --epochs=50 --mode=train
```
Thereafter run the following code to run the testing phase once again using the options in the `input_files/input.json` using the **0th numbered CUDA GPU** for inference:
```console
$ deepmem --device=0 --mode=test
```
It should save a histogram with a visual explanation of the model performance in `post/histogram.png` (path can be changed in input file)
### **Shell Arguments Explanation**:

```console
$ deepmem --help
usage: deepmem [-h] [--loader LOADER] [--device DEVICE] [--epochs EPOCHS] [--inputfile INPUTFILE] [--mode MODE]

optional arguments:
  -h, --help            show this help message and exit
  --loader LOADER
  --device DEVICE
  --epochs EPOCHS
  --inputfile INPUTFILE
  --mode MODE           'train' or 'test'
```

Pass these to `deepmem` when running the code.
1. **--loader**: [Default: hybridMT] Which dataloader implementation to use out of [inbuilt, hybrid, hybridMT]
**AT THE MOMENT ONLY hybridMT is properly supported. It loads all the data into memory and can be use for reasonably sized datasets (works comfortably with ~300k events on DGX)**

2. **--device**: [Default: None] Which numbered cuda device to use for training. Using `None` will select the CPU instead (Not recommended)

3. **--epochs**: [Default: 10] Number of epochs to train for

4. **--inputfile**: [Default: input_files/input.json] Path to the input file

5. **--mode**: [Default: train] Whether to run in training mode or testing mode

### **Input File Options Explanation**:
- **input_paths :** &nbsp; [ *list of strings* ] Contains paths to the .root files of the inputs (events data)

- **output_paths :** &nbsp; [ *list of strings* ] Contains paths to the .root files of the outputs (weights) corresponding to the inputs above

- **input_tree :** &nbsp; [ *string* ] Specifies which tree/directory in the input file to parse

- **output_tree :** &nbsp; [ *string* ] Specifies which tree/directory in the output file to parse

- **prefixes :** &nbsp; [ *list of list of strings* ] Contains the prefixes to search for in the input file. Further explanation below

- **suffixes :** &nbsp; [ *list of list of strings* ] Contains the suffixes to search for in the input file. Further explanation below

*for example:*
If prefixes are [ ["lep1_"] , ["lep2_", "j1_"] , ["MET"] ] and suffixes are [ ["PT"] , ["PT", "Eta"] , ["", "_Phi"]] the dataloader wil load the following variables from the input file `lep1_PT, "lep2_PT, lep2_Eta, j1_PT, j1_Eta, MET, MET_Phi`. It is basically all combinations of prefixes and suffixes from lists at the same index.
- **dir_name_processing :** &nbsp; [ *boolean* ] [ **DEPRECATED** ]Further processing on the particle name string (if required) to identify particle directory.

- **weights :** &nbsp; [ *list of strings* ] Specifies the directory with weights associated with each event

- **batch_size :** &nbsp; [ *integer* ] Batch size hyperparameter for training. Relevant for `--loader=hybrid and --loader=hybridMT`

- **chunk_entries :** &nbsp; [ *integer* ] Specifies the number of entries in one loaded chunk. Only Relevant for `--loader=hybrid` **NOT RELEVANT FOR hybridMT**

- **shuffle :** &nbsp; [ *boolean* ] Whether the loaded data needs to be shuffled. Relevant for `--loader=hybrid and --loader=hybridMT`

- **split :** &nbsp; [ *list of 3 ints* ] Training, Validation and Testing Split
**KEEP THIS AT 8,1,1 for now since stability of other splits is still being determined**

- **LearningRate :** &nbsp; [ *float* ] Initial learning rate of the model

- **save_lowest_loss_model_at :** &nbsp; [ *string* ] Where to save model that achieved lowest loss during training

- **save_scaler_at :** &nbsp; [ *string* ] Where to save scaler fit to the training data (during training)

- **load_saved_model_from :** &nbsp; [ *string* ] Where to load model that will be used during testing phase

- **load_scaler_from :** &nbsp; [ *string* ] Where to load scaler that will be used during testing phase (should be the same scaler that was saved during training phase)

- **save_testing_histogram_at :** &nbsp; [ *string* ] Where to save final output histogram
