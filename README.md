# DeepMEM
[![NSF Award Number](https://img.shields.io/badge/NSF-1836650-blue.svg)](https://nsf.gov/awardsearch/showAward?AWD_ID=1836650)

[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/mihirkatare/DeepMEM/main.svg)](https://results.pre-commit.ci/latest/github/mihirkatare/DeepMEM/main)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This repository contains the code that was used for the IRIS-HEP Fellowship project: **[Deep Learning Implementations for Sustainable Matrix Element Method Calculations](https://iris-hep.org/fellows/mihirkatare.html)**.


---
## **Project Description**
The Matrix Element Method (MEM) is a powerful statistical analysis technique for experimental and simulated particle physics data. It has several benefits over black-box methods like neural networks, owing to its transparent and interpretable results. The drawback of MEM; however, is the significant amount of computationally intensive calculations involved in its execution, which impedes research that relies on it. This project aims to improve the viability of MEM, by implementing deep learning techniques to accurately and efficiently approximate MEM calculations - providing the much required speedup over the traditional approach, while preserving its interpretability. The implemented model can be used as a good approximation during the exploratory phase of research, and the full ME calculations can be used for the final runs, making the workflow for research involving MEM much more efficient.
