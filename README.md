# MSS - the Multi-Sensor Study

This is the code to accompany our pre-print: 

Uncovering personalised glucose responses and circadian rhythms from multiple wearable biosensors with Bayesian dynamical modelling (2022) Phillips NE, Collet TH*, Naef F*. 

The goal of this computational method is to learn interpretable, personal parameters from wearable time series data to reveal circadian rhythms and physiological responses to external stressors such as meals and physical activity. The package MSS uses [TensorFlow Probability](https://www.tensorflow.org/probability).

## Overview

In our study we measured food and drink ingestion, glucose dynamics, physical activity, heart rate (HR) and heart rate variability (HRV) in 25 healthy participants over 14 days.

We subdivide the problem of analysing the multiple signals by creating three successive mathematical models, where the models include different subsets of variables.

- Model 1: food + drink events, glucose CGM, circadian
- Model 2: physical activity, HR, HRV, circadian
- Model 3: food + drink events, glucose CGM, physical activity, HR, HRV, circadian

![modelsoverview.png](images/modelsoverview.png)

## Tutorial

These three different models are implemented as part of the MSS package. Please see the tutorial Jupyter Notebook `tutorials/tutorial.ipynb` to see how to perform inference using these models.

## Installation instructions

Please run the following code in the terminal to install the MSS package

```
git clone https://github.com/Naef-lab/MultiSensor && cd MultiSensor
conda env create -f environment.yml
conda activate MultiSensor
pip install .
```