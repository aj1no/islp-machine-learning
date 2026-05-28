# ISLP Machine Learning Exercises

*[Ler em Português](README.pt-br.md)*

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.8-11557c?style=flat-square)](https://matplotlib.org/)
[![License MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![CI](https://img.shields.io/github/actions/workflow/status/aj1no/islp-machine-learning/ci.yml?branch=main&label=CI&style=flat-square)](https://github.com/aj1no/islp-machine-learning/actions)

This repository contains solved classification exercises using Machine Learning models in Python with `scikit-learn` and the `ISLP` textbook package. The source code explores different classification algorithms and the effects on statistical accuracy when separating Train/Test data vs using the full dataset.

---

## Code Overview

The main script `exercicios_logistica.py` consists of 3 practical parts:

1. **Random Forest and Breast Cancer Dataset:** Loads native `sklearn` data, trains the model using an 80%/20% split, plots a confusion matrix heatmap, and outputs a complete classification report.
2. **Logistic Regression with Train/Test Split:** Re-implements 4 classic exercises from the `ISLP` library, dividing the modeling with independent samples and validating on unseen slices of the table using binarization matrix notation (`get_dummies`).
   - `ISLP::Default` (Target: *student*)
   - `ISLP::Smarket` (Target: *Direction*)
   - `ISLP::Weekly` (Target: *Direction*)
   - `ISLP::Caravan` (Target: *Purchase*)
3. **Basic Logistic Regression:** Removes `train_test_split` to train and verify accuracy on the entire dataset, illustrating the statistical concept of "training error underestimation" when predicting samples the model has already seen.

---

## Libraries & Installation

To run this project, make sure you have Python 3.10+ and execute the command below in your terminal to fetch the core requirements:

```bash
pip install ISLP pandas scikit-learn matplotlib seaborn
```

Once done, run the main file:
```bash
python exercicios_logistica.py
```
*(Note that the Logistic Regression reports from Part 2 and 3 will print in the terminal only after closing the Matplotlib figure window that opens in Part 1).*
