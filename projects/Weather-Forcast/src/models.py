"""
models.py -- Model definitions
===============================
- BiLSTM:      Bidirectional LSTM (captures forward + backward context)
- LSTMAttn:    LSTM + self-attention pooling
- LSTMModel2:  Two-layer LSTM + Dropout
- Transformer: PositionalEncoding + Encoder + mean pool
- Ensemble:    Wraps multiple models, averages predictions

Factory: build_model(cfg) returns the requested model.
"""

import numpy as np
import torch
import torch.nn as nn


# ─── BiLSTM ───────────────────────────────────────────────────

class BiLSTM(nn.Module):
    """Bidirectional LSTM: hidden_dim//2 per direction, concat => hidden_dim"""

    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim // 2, num_layers=num_layers,
            batch_first=True, bidirectional=True, dropout=dropout
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.norm(out[:, -1, :])
        out = self.dropout(out)
        return self.fc(out).squeeze(-1)


# ─── LSTM + Attention ──────────────────────────────────────────

class LSTMAttn(nn.Module):
    """LSTM with multi-head self-attention pooling over timesteps"""

    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2, nhead=4):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=dropout
        )
        self.attn = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        attn_out, _ = self.attn(out, out, out)
        out = self.norm(out + attn_out)    # residual
        out = out.mean(dim=1)              # pool all timesteps
        out = self.dropout(out)
        return self.fc(out).squeeze(-1)


# ─── Two-layer LSTM (improved) ─────────────────────────────────

class LSTMModel2(nn.Module):
    """Two-layer LSTM + LayerNorm + Dropout + deeper FC"""

    def __init__(self, input_dim, hidden_dim=64, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=2,
            batch_first=True, dropout=dropout
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.norm(out[:, -1, :])
        out = self.dropout(out)
        return self.fc(out).squeeze(-1)


# ─── Transformer ───────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    """Transformer Encoder: input proj -> pos enc -> encoder -> mean pool -> FC"""

    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2,
                 dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.fc = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.fc(x).squeeze(-1)


# ─── Ensemble ──────────────────────────────────────────────────

class Ensemble(nn.Module):
    """Wraps a list of pre-trained models, averages their predictions."""

    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x):
        preds = torch.stack([m(x) for m in self.models], dim=0)
        return preds.mean(dim=0)


# ─── Factory ───────────────────────────────────────────────────

def build_model(cfg):
    name = cfg["model"]["name"]
    d = cfg["model"]

    if name == "bilstm":
        return BiLSTM(d["input_dim"], d["hidden_dim"], d["num_layers"], d["dropout"])
    elif name == "lstmattn":
        return LSTMAttn(d["input_dim"], d["hidden_dim"], d["num_layers"], d["dropout"], d["nhead"])
    elif name == "lstm2":
        return LSTMModel2(d["input_dim"], d["hidden_dim"], d["dropout"])
    elif name == "transformer":
        return TransformerModel(d["input_dim"], d["d_model"], d["nhead"],
                                d["num_layers"], d["dim_feedforward"], d["dropout"])
    elif name == "ensemble":
        raise ValueError("Use build_ensemble() to create ensemble models")
    raise ValueError(f"Unknown model: {name}")


def build_ensemble(cfg, ckpt_dir):
    """Build ensemble from individual trained checkpoints."""
    models = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for mname in ["bilstm", "lstm2", "transformer"]:
        m = build_model({**cfg, "model": {**cfg["model"], "name": mname}})
        path = os.path.join(ckpt_dir, f"{mname}_best.pt")
        if os.path.exists(path):
            m.load_state_dict(torch.load(path, map_location=device))
            m.eval()
            models.append(m)
    if len(models) < 2:
        raise FileNotFoundError("Need at least 2 trained models for ensemble")
    return Ensemble(models).to(device)

import os  # noqa: E402


def build_loss(name):
    if name == "mae":
        return nn.L1Loss()
    elif name == "mse":
        return nn.MSELoss()
    raise ValueError(f"Unknown loss: {name}")
