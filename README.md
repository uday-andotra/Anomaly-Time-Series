# TimeSeries Anomaly Detection

## Overview

This repository implements and compares anomaly detection techniques for time series data using the Numenta Anomaly Benchmark (NAB) dataset, specifically the Twitter volume mentions for IBM (sampled every 5 minutes from February to April 2015). The project explores probabilistic and forecasting models, including:

- **Time Series Decomposition** (using Twitter's AnomalyDetection library in R).
- **Hidden Markov Models (HMM)** (implemented in R with depmixS4).
- **Autoencoders** (implemented in Python with Keras/TensorFlow).

The dataset contains 15,893 observations, and anomalies are benchmarked against ARTime (a top-performing detector from NAB, identifying ~1,590 anomalies). The methods detect point and subsequence anomalies, with results visualized and compared in the included report.

Key findings:
- Decomposition: Detects 539 anomalies (focuses on residuals via ESD test).
- HMM: Detects 776 anomalies (using forward-backward algorithm with 8 states).
- Autoencoder: Detects 781 anomalies (using MSE reconstruction error threshold).

The repository includes code for reproduction, the raw dataset, and a detailed PDF report.

## Table of Contents

- [Overview](#overview)
- [Files](#files)
- [Requirements](#requirements)
- [Installation and Setup](#installation-and-setup)
- [Usage](#usage)
- [Methods](#methods)
- [Results](#results)
- [References](#references)

## Files

- **`ProjectAppliedTimeSeries_Decomp_HMM.Rmd`**: R Markdown file implementing time series decomposition and HMM for anomaly detection. Includes data loading, visualization, and anomaly plotting.
- **`ProjectAppliedTimeSeries_Autoencoder.ipynb`**: Jupyter Notebook implementing an autoencoder model in Python. Includes data normalization, model training, and anomaly visualization.
- **`Anomaly Detection in Time Series.pdf`**: Comprehensive report detailing the introduction, data, methods (decomposition, HMM, autoencoder), results, conclusions, and references.
- **`Twitter_volume_IBM.csv`**: Raw NAB dataset (timestamped Twitter mentions for IBM).

## Requirements

### R Dependencies
- tseries
- AnomalyDetection (from GitHub: twitter/AnomalyDetection)
- depmixS4

### Python Dependencies
- numpy
- pandas
- keras (or tensorflow.keras)
- matplotlib

No additional installations beyond these are needed, as the code uses pre-installed libraries in standard environments.

## Installation and Setup

1. Clone the repository:
   ```
   git clone https://github.com/uday-andotra/TimeSeries-AnomalyDetection.git
   cd TimeSeries-AnomalyDetection
   ```

2. For R: Install dependencies if not already present:
   ```
   install.packages(c("tseries", "depmixS4"))
   devtools::install_github("twitter/AnomalyDetection")
   ```

3. For Python: Install dependencies via pip:
   ```
   pip install numpy pandas keras tensorflow matplotlib
   ```

4. The dataset is loaded directly from URLs in the code, but a local copy (`Twitter_volume_IBM.csv`) is provided for offline use.

## Usage

### Running the R Markdown (Decomposition and HMM)
- Open `ProjectAppliedTimeSeries_Decomp_HMM.Rmd` in RStudio.
- Knit the document to HTML for visualizations and results.
- Key outputs: Plots of decomposed series, estimated HMM states, and detected anomalies.

### Running the Jupyter Notebook (Autoencoder)
- Open `ProjectAppliedTimeSeries_Autoencoder.ipynb` in Jupyter Notebook or Google Colab.
- Run all cells sequentially.
- Key outputs: Training loss plot, reconstructed data, and anomaly scatter plot.
- Note: The notebook detects 781 anomalies based on a 95th percentile MSE threshold.

### Viewing the Report
- Open `Anomaly Detection in Time Series.pdf` for a full explanation, including figures and comparisons.

## Methods

### Time Series Decomposition
- Decomposes the series into trend, seasonal, and random components.
- Uses Twitter's AnomalyDetection library with ESD test on residuals to flag anomalies (max_anoms=0.05).
- Effective for local/global outliers but less so for subsequences.

### Hidden Markov Model (HMM)
- Models the data as a Gaussian HMM with 8 hidden states.
- Fits using depmixS4 and detects anomalies via residuals exceeding a 95th quantile threshold.
- Captures sequential dependencies but sensitive to state count and parameter optimization.

### Autoencoder
- Neural network with an encoding layer (64 neurons, ReLU) and decoding layer (linear).
- Trained for 50 epochs on normalized sequences (length=288).
- Anomalies flagged where reconstruction MSE > 95th percentile.
- Handles both point and subsequence anomalies well, without assuming prior distributions.

For mathematical details and architectures, refer to the PDF report.

## Results

| Method                  | Anomalies Detected | Strengths                          | Weaknesses                        |
|-------------------------|--------------------|------------------------------------|-----------------------------------|
| Decomposition (AnomalyDetection) | 539               | Efficient for outliers; uses robust stats | Misses subsequence anomalies     |
| HMM (Forward-Backward) | 776               | Handles sequential data; probabilistic | Biased toward extremities; optimization challenges |
| Autoencoder (MSE)      | 781               | Captures point/subsequence anomalies; no priors | Less efficient for very long sequences |

Visualizations in the code show anomalies marked on the time series plot. Autoencoder performs best in aligning with ARTime benchmarks, detecting clusters like the April 2015 event.

## References

1. A review on outlier/anomaly detection in time series data (arXiv:2002.04236).
2. Numenta Anomaly Benchmark (NAB) GitHub: https://github.com/numenta/NAB.
3. Twitter AnomalyDetection GitHub: https://github.com/twitter/AnomalyDetection.
4. Multivariate time series anomaly detection: A framework of Hidden Markov Models (DOI: 10.1016/j.asoc.2017.06.035).
5. Intro to Autoencoders (TensorFlow Core): https://www.tensorflow.org/tutorials/generative/autoencoder.
6. Autoencoders for Anomaly Detection in an Industrial Multivariate Time Series Dataset (DOI: 10.3390/engproc2022016023).
