### Modelling the regional climate feedback of wildfires around Amazon Rainforest with deep learning approaches

---------------------------

This repository presents the archive of code implementation for my MDaSc (Master in Data Science) graduation project at the University of Hong Kong (HKU). The aim of the project was to predict 1) the occurrences of wildfires for the next day; 2) short-term (in days) and long-term (in months) post-wildfire meteorological impacts, including temperature and precipitations, with remote sensing imagery. All images were collected from Google Earth Engine. The deep learning model structure chosen was a hybrid of recurrent and convolutional designs to adapt for the spatial time series inputs. The encoder comprised stacked bi-directional ConvLSTM layers, bridged by a multi-head attention layer (as in the design of Transformers), and connected to the decoder for de-convolultion to reconstruct the output image (next-day or next-month prediction).

  
- Part I Data Engineering
    - Google Earth Engine API
    - Multi-resolution and Multi-channel inputs (wildfire events, ERA5 weather variables, NDVI, land covers, elevation, etc.)
    - Convolutional Variational Autoencoder (C-VAE)
- Part II Research Modelling
    - Pixelwise binary classification on wildfire events and Regression on long-term andshort-term weather
    - Bi-directional Convolutional LSTM (BiConvLSTM) Encoder
    - Multi-head attention mechanism as adopted in Transformer models
- Part III Generative Adversarial Learning
    - Conditional GAN (Generator + Critic models)
    - Generator with mixture of random noises and BiConvLSTM-encoded features
  
