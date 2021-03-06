# Interval-HCHO-Concentration-Estimation
<center class="half">
    <img src="https://github.com/dingyizhe2000/Interval-HCHO-Concentration-Estimation/blob/main/images/U.png" width="450"/><img src="https://github.com/dingyizhe2000/Interval-HCHO-Concentration-Estimation/blob/main/images/L.png" width="450"/> </center>
    
    Interval Predictions of Upper and Lower Bounds for Global 2019 HCHO Surface Concentration Distribution

This is the PyTorch implementation for our work--Mapping 2019 Global Surface HCHO Distribution and Confidential Interval by Satellite Observation of Sentinel-5P and Neural Network Model. With the usage of quality-driven interval estimation algorithm([High-Quality Prediction Intervals for Deep Learning](https://github.com/TeaPearce/Deep_Learning_Prediction_Intervals)), we manage to give the point and interval global HCHO surface concentration distribution in 2019.

Contact:  Bohan Jin [(2018200684@ruc.edu.cn)](2018200684@ruc.edu.cn); Yizhe Ding [(1810015@mail.nankai.edu.cn)](1810015@mail.nankai.edu.cn)

## Model Structure
<center class="half">
    <img src="https://github.com/dingyizhe2000/Interval-HCHO-Concentration-Estimation/blob/main/images/model.png" width="450"/></center>

Notice that ReLU activations in the last block are disabled.

## Code Files
Our project contains 4 code files:
- dataset.py
- function.py
- interval.ipynb
- point.ipynb

Hyperparameters are included in our files and you can run interval.ipynb and point.ipynb to reproduce our results.

<center class="half">
    <img src="https://github.com/dingyizhe2000/Interval-HCHO-Concentration-Estimation/blob/main/images/P.png" width="450"/></center>

    Point Estimation of Global 2019 HCHO Surface Concentration Distribution

## Training Set and Results
You can download the data from the website mentioned in our paper to train the network.  
We have also provided our [results](https://drive.google.com/file/d/10A2VIEHm22DF_gyCufV-pbgUdYYhNJKf/view) in .tif format for downloading.
