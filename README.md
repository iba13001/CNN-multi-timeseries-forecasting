# CNN-multi-timeseries-forecasting
Dataset class: pass multiple time series to data loaders
CNN function: predicts targets based on the passed features. The function is multivariate that can handle continuous, categorical and temporal features. 
Layers of CNN:
- CNN layer for sequentional predictions
- Embedding layer for categorical and temporal features
- Linear layer for continuous and categorical features to find their effect on target without sequence.
