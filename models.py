import math
import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_devices):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.predict_time = nn.Linear(hidden_size, 1)
        self.predict_device = nn.Linear(hidden_size, num_devices)
        self.predict_event = nn.Linear(hidden_size, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h0 = torch.zeros(1 * self.num_layers, x.size(0), self.hidden_size).to(device=x.device)
        c0 = torch.zeros(1 * self.num_layers, x.size(0), self.hidden_size).to(device=x.device)

        features, _ = self.lstm(x)
        # features = features[:, -self.num_hidden_seq_for_out:]
        features = features[:, -1]
        #features = features.reshape(features.size(0), -1)

        time_predicted = self.sigmoid(self.predict_time(features))
        device_predicted = self.predict_device(features)
        event_predicted = self.sigmoid(self.predict_event(features))

        return time_predicted, device_predicted, event_predicted

class CNNModel(nn.Module):
    def __init__(self, in_channels, hidden_size, num_devices):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(
                        in_channels=in_channels,
                        out_channels=32,
                        kernel_size=3,
                        stride=1,
                        padding=1
        )
        self.conv2 = nn.Conv1d(
                        in_channels=32,
                        out_channels=hidden_size,
                        kernel_size=5,
                        stride=1,
                        padding=2
        )

        self.predict_time = nn.Linear(hidden_size, 1)
        self.predict_device = nn.Linear(hidden_size, num_devices)
        self.predict_event = nn.Linear(hidden_size, 1)

        self.dropout = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.permute(0, 2, 1)                  # (batch, seq_len, input_dim) -> ((batch, input_dim, seq_len)
        features = self.conv2(self.conv1(x))    # (batch, input_dim, seq_len) -> (batch, hidden_size, seq_len)

        features = features[:, :, -1]           # (batch, hidden_size, seq_len) -> (batch, hidden_size)

        time_predicted = self.sigmoid(self.predict_time(features))
        device_predicted = self.predict_device(features)
        event_predicted = self.sigmoid(self.predict_event(features))

        return time_predicted, device_predicted, event_predicted


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=1000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):

        return x + self.pe[:, :x.size(1)]


class TransformerModel(nn.Module):
    def __init__(self, input_size, d_model, nhead, dim_feedforward, dropout, num_layers, num_devices):
        super().__init__()
        self.d_model = d_model
        self.expand_dim = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.predict_time = nn.Linear(d_model, 1)
        self.predict_device = nn.Linear(d_model, num_devices)
        self.predict_event = nn.Linear(d_model, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, src, src_mask=None):
        """
        Arguments:
            src: Tensor, shape ``[batch_size, seq_len, input_dim]``
            src_mask: Tensor, shape ``[seq_len, seq_len]``

        Returns:
            output Tensor of shape ``[batch_size, output_dim]``
        """
        src = self.expand_dim(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        features = self.encoder(src, src_mask)
        features = features[:, -1]

        time_predicted = self.sigmoid(self.predict_time(features))
        device_predicted = self.predict_device(features)
        event_predicted = self.sigmoid(self.predict_event(features))

        return time_predicted, device_predicted, event_predicted

