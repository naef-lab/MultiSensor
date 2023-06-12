# Datasets for the Multi-Sensor Study (MSS)

Here are the datasets collected in our paper: 

Uncovering personalised glucose responses and circadian rhythms from multiple wearable biosensors with Bayesian dynamical modelling (2023) Phillips NE, Collet TH\*, Naef F\*. 

## Overview of data

The data is structured into different folders that correspond to the different models used in the article.

- Model 1: Model1 data (food and glucose)
- Model 2: Model2 data (Actiheart data)
- Model 3: Model3 data (glucose actiheart integrated)

Within each folder, each different file represents the data for one study participant.

Please find below a description of the variables

## Time variables in common across all data types

For each dataset, time is encoded using the following variables:

- weekday: represents the day of the week (where Monday=0,...,Sunday=6)
- days: represents the number of days since the experiment began. In most cases this starts at 0, but note that in some cases where devices fell off and were replaced etc., the first day is not always zero.
- hours: the hour of day of the specific recording (from 0-24 hours)
- abs_time_days: time measured as the total number of days since the beginning of the experiment, where 0 is counted as midnight preceding the first measurement (e.g. abs_time_days=0.5 represents 12:00 on the first day of the experiment).
- abs_time_hours: time measured as the total number of hours since the beginning of the experiment, where 0 is counted as midnight preceding the first measurement (e.g. abs_time_hours =12.0 represents 12:00 on the first day of the experiment).

## Model1 data (food and glucose)

For IDs 3, 5, 7, 8 and 25, there are two glucose sensors. The two different sensors are distinguished in the filenames by the number (either 1 or 2) after the hyphen.

### food_MSSXX-1.xlsx 

This contains the food data recorded with the smartphone application

Each line represents a consumed item that was recorded with the smartphone app. For the purposes of anonymity, only the timestamp is provided, and the text annotation has been removed. However, items with the same text annotation have the same value of 'food_item_index'

- food_item_index: denotes which ingestion events have the same free text entry. E.g. if two items have food_item_index=3, this is because they have the same annotation label e.g. coffee

### glucose_MSSXX-1.csv

The CGM data for the participant. Units are mmol/L

- Historic Glucose (mmol/L): the original glucose value returned by the CGM device.
- Trend: the nonparametric regression fit with Gaussian processes (GPs) to remove the long-term trends observed in the data
- Detrended: the CGM glucose values after correcting for the long term trend. This is used in the analysis.

## Model2 data (Actiheart data)

The physical and heart activity data used in the study, as measured with the Actiheart device.

### actiheart_MSSXX.csv

- Activity: the activity counts as measured with the Actiheart device
- BPM: the heart rate (beats per minute) as measured with the Actiheart device
- RMSSD: the heart rate variability (RMSSD, root mean square of successive differences between normal heartbeats (ms)) as measured with the Actiheart device
- Comments: comments on data quality (exported with Actiheart data)
- Quality: signal quality (0-1). We use a cut-off at 0.8.

## Model3 data (glucose actiheart integrated)

The glucose, food and Actiheart data combined and projected onto 15 min intervals.

### glucose_actiheart_integrated_MSSXX-1.csv

- Activity: the activity counts as measured with the Actiheart device
- BPM: the heart rate (beats per minute) as measured with the Actiheart device
- RMSSD: the heart rate variability (RMSSD, root mean square of successive differences between normal heartbeats (ms)) as measured with the Actiheart device
- Detrended: the CGM glucose values after correcting for the long term trend. This is used in the analysis.
- mask: determines whether data will be masked (i.e. skipped) during model training. This is used as an input into the model log likelihood calculation.

### food_MSSXX-1.xlsx 

This contains the food data recorded with a smartphone application

Each line represents a consumed item that was recorded with the smartphone app. For the purposes of anonymity, only the timestamp is provided, and the text annotation has been removed. However, items with the same text annotation have the same value of 'food_item_index'

- food_item_index: denotes which ingestion events have the same free text entry. E.g. if two items have food_item_index=3, this is because they have the same annotation label e.g. coffee

## Citation

Uncovering personalised glucose responses and circadian rhythms from multiple wearable biosensors with Bayesian dynamical modelling (2023) Phillips NE, Collet TH\*, Naef F\*. 
