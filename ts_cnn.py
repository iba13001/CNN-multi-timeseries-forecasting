import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import pytorch_lightning as pl
from torch import nn

class TimeSeriesDataset(Dataset):
    def __init__(self, df, sequence_length, target_length, continuous_features, categorical_features):
        """
        Args:
            df (pd.DataFrame): DataFrame containing columns ['ts_id', 'time_step', 'value', 'LY_AMC_PERIOD', 'LY_AMC_WEEK', 'LY_AMC_DAY'].
            sequence_length (int): Length of input sequences.
            target_length (int): Length of output sequences (forecast horizon).
        """
        self.df = df
        self.sequence_length = sequence_length
        self.target_length = target_length
        self.continuous_features = continuous_features
        self.categorical_features = categorical_features
        
        # Group by ts_id and store the groups
        self.ts_groups = df.groupby('ts_id')
        
        # Generate sequence indices for each ts_id
        self.sequences = []
        for ts_id, group in self.ts_groups:
            series_length = len(group)
            #print(ts_id, series_length)
            for idx in range(series_length - sequence_length - target_length + 1):
                self.sequences.append((ts_id, idx))
            #print(ts_id, idx)
            
    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        ts_id, seq_idx = self.sequences[idx]
        group = self.ts_groups.get_group(ts_id)
        
        
        # Get input (x), temporal features, and target (y)
        x = group.iloc[seq_idx : seq_idx + self.sequence_length][s_metric].values
        y = group.iloc[seq_idx + self.sequence_length : seq_idx + self.sequence_length + self.target_length][s_metric].values
        
        # Sequence Features
        continuous_seq = group.iloc[seq_idx : seq_idx + self.sequence_length][self.continuous_features].values
        categorical_seq = group.iloc[seq_idx : seq_idx + self.sequence_length][self.categorical_features].values
        
        # Target features
        continuous = group.iloc[seq_idx + self.sequence_length : seq_idx + self.sequence_length + self.target_length][self.continuous_features].values
        categorical = group.iloc[seq_idx + self.sequence_length : seq_idx + self.sequence_length + self.target_length][self.categorical_features].values
 
        # Convert to tensors
        x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(-1)
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)
        continuous_seq_tensor = torch.tensor(continuous_seq, dtype=torch.float32)  # (sequence_length, num_continuous_features)
        categorical_seq_tensor = torch.tensor(categorical_seq, dtype=torch.long)
        continuous_tensor = torch.tensor(continuous, dtype=torch.float32)  # (sequence_length, num_continuous_features)
        categorical_tensor = torch.tensor(categorical, dtype=torch.long)
        
        return {'input': x_tensor, 'target': y_tensor, 
                #'AMC_YEAR_CAT': year_tensor, 'LY_AMC_PERIOD': period_tensor, 'LY_AMC_WEEK': week_tensor, 'LY_AMC_DAY': day_tensor,
                
                'ts_id': torch.tensor([ts_id]),
                'target_ts_id' : torch.tensor([ts_id]),
                'continuous_seq' : continuous_seq_tensor,
                'categorical_seq' : categorical_seq_tensor,
                'categorical': categorical_tensor, 
                'continuous': continuous_tensor,
               }

class CNNForecastingModel(pl.LightningModule):
    def __init__(self, input_dim, continuous_dim, categorical_dims, hidden_dim, output_dim, num_layers, sequence_length, target_length, 
                 ts_id_vocab_size, 
                 embed_dim_ts, 
                 embed_dims_cat,
                 lr=1e-3
                ):
        
        super(CNNForecastingModel, self).__init__()
        self.target_length = target_length
        self.sequence_length = sequence_length
        self.save_hyperparameters()
        
        # Embeddings for IDs and categorical features
        self.ts_id_embedding = nn.Embedding(ts_id_vocab_size, embed_dim_ts)
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(num_categories, embed_dim_cat)
            for num_categories, embed_dim_cat in zip(categorical_dims, embed_dims_cat)
        ])

        self.embedding_dim = sum(embed_dims_cat) + embed_dim_ts

        # CNN for sequence modeling
        self.conv1 = nn.Conv1d(input_dim + self.embedding_dim + continuous_dim, hidden_dim, kernel_size=3, padding=1)  # First Conv layer
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)  # Second Conv layer
        self.pool = nn.AdaptiveAvgPool1d(1)  # Pooling layer
        self.fc_seq = nn.Linear(hidden_dim, self.target_length)  # Fully connected for sequence output

        # Sequential for target feature modeling
        hidden_dim_cat = 128  # Define the hidden dimension for the non-linear layer
        self.fc_cat = nn.Sequential(
            nn.Linear((self.embedding_dim + continuous_dim) * self.target_length, hidden_dim_cat),
            nn.ReLU(),
            nn.Linear(hidden_dim_cat, self.target_length)
        )

        # Combine sequence and categorical predictions
        self.fc_combined = nn.Linear(self.target_length * 2, self.target_length)
        self.lr = lr

    def forward(self, x, ts_id, continuous_seq, categorical_seq, target_ts_id, continuous, categorical):
        
        # Seq embeded features
        ts_id_embed_seq = self.ts_id_embedding(ts_id.repeat(1, self.sequence_length))
        embedded_cat_features_seq = [
            embedding(categorical_seq[:, :, i]) for i, embedding in enumerate(self.cat_embeddings)
        ]
        cat_features_embed_seq = torch.cat(embedded_cat_features_seq, dim=-1)
        target_features_seq = torch.cat([ts_id_embed_seq, continuous_seq, cat_features_embed_seq], dim=-1)
        x = torch.cat([x, target_features_seq], dim=-1)
        
        # Process sequence data with CNN
        x = x.permute(0, 2, 1)  # Change shape to (batch_size, input_dim, sequence_length) for Conv1d
        x = torch.relu(self.conv1(x))  # First Conv layer
        x = torch.relu(self.conv2(x))  # Second Conv layer
        x = self.pool(x).squeeze(-1)  # Pooling and flattening
        seq_prediction = self.fc_seq(x)

        # Process Target features
        target_ts_id_embed = self.ts_id_embedding(target_ts_id.repeat(1, self.target_length))
        embedded_cat_features = [
            embedding(categorical[:, :, i]) for i, embedding in enumerate(self.cat_embeddings)
        ]
        cat_features_embed = torch.cat(embedded_cat_features, dim=-1)
        target_features = torch.cat([target_ts_id_embed, continuous, cat_features_embed], dim=-1)
        target_features_batch_size, horizon, num_features = target_features.size()
        target_features_prediction = self.fc_cat(target_features.view(target_features_batch_size, -1))

        # Combine predictions
        combined_input = torch.cat([seq_prediction, target_features_prediction], dim=-1)
        combined_prediction = self.fc_combined(combined_input)

        return combined_prediction

    def training_step(self, batch, batch_idx):
        inputs, targets, ts_id = batch['input'], batch['target'].squeeze(-1), batch['ts_id']
        target_ts_id = batch['target_ts_id']
        continuous_seq = batch['continuous_seq']
        categorical_seq = batch['categorical_seq']
        continuous = batch['continuous']
        categorical = batch['categorical']
        predictions = self(inputs, ts_id, continuous_seq, categorical_seq, target_ts_id, continuous, categorical)
        loss = self.weighted_loss(predictions, targets)
        #loss = self.hubber(predictions, targets, 0.2)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets, ts_id = batch['input'], batch['target'].squeeze(-1), batch['ts_id']
        target_ts_id = batch['target_ts_id']
        continuous_seq = batch['continuous_seq']
        categorical_seq = batch['categorical_seq']
        continuous = batch['continuous']
        categorical = batch['categorical']
        predictions = self(inputs, ts_id, continuous_seq, categorical_seq, target_ts_id, continuous, categorical)
        val_loss = self.weighted_loss(predictions, targets)
        #val_loss = self.hubber(predictions, targets, 0.2)
        self.log("val_loss", val_loss, prog_bar=True)
        
        # Calculate MAPE
        mape = self.calculate_mape(predictions, targets)
        # Log the MAPE for validation
        self.log("val_mape", mape, prog_bar=True)
        
        return {"val_loss": val_loss}

    def calculate_mape(self, predictions, targets):
        # Avoid division by zero by adding a small epsilon
        weights = torch.where(targets == 0, 0, 1.0)  # Lower weight for zero values
        epsilon = 1e-6
        abs_percentage_error = torch.abs((targets - predictions) / (targets + epsilon)) * weights
        mape = torch.mean(abs_percentage_error) * 100
        return mape

    def weighted_loss(self, predictions, targets):
        weights = torch.where(targets == 0, 0.1, 1.0)  # Lower weight for zero values
        loss_fn = nn.MSELoss(reduction='none')
        loss = loss_fn(predictions, targets)
        weighted_loss = (loss * weights).mean()
        return weighted_loss
    
    def hubber(self, predictions, targets, delta):
        self.delta = delta
        # Create a mask to ignore zero targets
        mask = targets != 0  # True for non-zero targets, False for zero targets

        # Calculate the residuals (difference between predictions and targets)
        residuals = predictions - targets

        # Apply the Huber Loss formula element-wise
        quadratic = 0.5 * residuals**2
        linear = self.delta * (torch.abs(residuals) - 0.5 * self.delta)
        huber = torch.where(torch.abs(residuals) <= self.delta, quadratic, linear)

        # Apply the mask to ignore zero targets
        masked_huber = huber * mask

        # Return the mean loss over the valid (non-zero) targets
        return masked_huber.sum() / mask.sum()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
