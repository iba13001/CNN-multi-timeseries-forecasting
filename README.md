# CNN-multi-timeseries-multivariate-forecasting
- Dataset class: pass multiple time series to data loaders
- CNN class: predicts targets based on the passed features. The function is multivariate that can handle continuous, categorical and temporal features for multiple timeseries. Each series has an index ts_id.
  Layers of CNN:
  - CNN layer for sequentional predictions
  - Embedding layer for categorical and temporal features
  - Linear layer for continuous and categorical features to find their effect on target without sequence.
