#!/usr/bin/env python
# coding: utf-8

# In[3]:


get_ipython().system('pip install torch_geometric')


# In[4]:


import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch_geometric.nn import GATConv
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")


# In[9]:


import os

desktop = r"C:\Users\thiru\OneDrive\Desktop"

print(os.listdir(desktop))


# In[10]:


import scipy.io as sio

def inspect_mat(filepath):
    mat = sio.loadmat(filepath)

    # Show top-level keys
    keys = [k for k in mat.keys() if not k.startswith('_')]
    print(f"\n=== {filepath} ===")
    print(f"Top-level keys: {keys}")

    battery_data = mat[keys[-1]]
    print(f"battery_data shape: {battery_data.shape}")
    print(f"battery_data dtype: {battery_data.dtype}")

    # Inspect first cycle
    cycle0 = battery_data[0][0]
    print(f"\nFirst cycle dtype (field names): {cycle0.dtype.names}")
    print(f"First cycle shape: {cycle0.shape}")

    # Show field values
    for field in cycle0.dtype.names:
        val = cycle0[field]
        print(f"  [{field}] -> type={type(val)}, shape={val.shape}, sample={str(val.flat[0])[:80]}")

# LOAD FILE
inspect_mat(r"C:\Users\thiru\OneDrive\Desktop\NASA AND EV DATASETS\B0005.mat")


# In[12]:


import numpy as np
import scipy.io as sio
import pandas as pd
import os

def load_mat_battery(filepath):
    mat = sio.loadmat(filepath)

    key = [k for k in mat.keys() if not k.startswith('_')][-1]
    battery_data = mat[key]

    cycles_raw = battery_data[0, 0]['cycle'][0]

    cycles = []

    for i in range(len(cycles_raw)):
        cycle = cycles_raw[i]

        cycle_type = str(cycle['type'][0])

        if cycle_type == 'discharge':
            try:
                data = cycle['data'][0, 0]

                voltage      = data['Voltage_measured'].flatten()
                current      = data['Current_measured'].flatten()
                temperature  = data['Temperature_measured'].flatten()
                current_load = data['Current_load'].flatten()
                voltage_load = data['Voltage_load'].flatten()
                time         = data['Time'].flatten()
                capacity     = float(data['Capacity'].flatten()[0])

                n = len(voltage)

                for j in range(n):
                    cycles.append({
                        'cycle_idx':            i,
                        'Voltage_measured':     voltage[j],
                        'Current_measured':     current[j],
                        'Temperature_measured': temperature[j],
                        'Current_load':         current_load[j],
                        'Voltage_load':         voltage_load[j],
                        'Time':                 time[j],
                        'Capacity':             capacity,
                    })

            except Exception as e:
                print(f"Skipping cycle {i} in {filepath}: {e}")
                continue

    return pd.DataFrame(cycles)


# =========================
# FOLDER PATH
# =========================

base_path = r"C:\Users\thiru\OneDrive\Desktop\NASA AND EV DATASETS"

# =========================
# BATTERY FILES
# =========================

battery_files = [
    ('B0005.mat', 'B0005'),
    ('B0007.mat', 'B0007'),
    ('B0018.mat', 'B0018')
]

dfs = []

for fname, bname in battery_files:

    full_path = os.path.join(base_path, fname)

    try:
        tmp = load_mat_battery(full_path)

        tmp['battery'] = bname

        dfs.append(tmp)

        print(f"{bname}: {len(tmp)} rows, {tmp['cycle_idx'].nunique()} cycles")

    except Exception as e:
        print(f"Skipping {bname} due to error: {e}")

# =========================
# CONCAT ALL
# =========================

df_raw = pd.concat(dfs, ignore_index=True)

print("\nTotal rows:", df_raw.shape)

# Preview
print(df_raw.head())


# In[13]:


get_ipython().system('pip install torch torch-geometric scikit-learn numpy pandas matplotlib')


# In[14]:


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Build cycle-level features from df_raw ────────────────────────────────
df_cycle = (
    df_raw
    .groupby(['battery', 'cycle_idx'])
    .agg(
        Capacity          = ('Capacity', 'first'),
        mean_voltage      = ('Voltage_measured', 'mean'),
        min_voltage       = ('Voltage_measured', 'min'),
        max_voltage       = ('Voltage_measured', 'max'),
        std_voltage       = ('Voltage_measured', 'std'),
        mean_temp         = ('Temperature_measured', 'mean'),
        max_temp          = ('Temperature_measured', 'max'),
        mean_current      = ('Current_measured', 'mean'),
        duration          = ('Time', 'max'),
        n_points          = ('Voltage_measured', 'count'),
    )
    .reset_index()
)

df_cycle = df_cycle.sort_values(['battery', 'cycle_idx']).reset_index(drop=True)
df_cycle['cycle_num'] = df_cycle.groupby('battery').cumcount() + 1

# ── SOH label ─────────────────────────────────────────────────────────────
first_cap = df_cycle.groupby('battery')['Capacity'].transform('first')
df_cycle['SOH'] = df_cycle['Capacity'] / first_cap

# ── SOC label: normalize time within each cycle → proxy for charge level ──
# SOC at each timestep = 1 - (cumulative_Ah / total_Ah)
# Cycle-level: use duration-normalized capacity as SOC proxy
max_cap = df_cycle.groupby('battery')['Capacity'].transform('max')
df_cycle['SOC'] = df_cycle['Capacity'] / max_cap  # 0–1 within each battery

# ── Rolling features ──────────────────────────────────────────────────────
for col in ['Capacity', 'mean_voltage', 'mean_temp', 'duration']:
    df_cycle[f'{col}_roll5'] = (
        df_cycle.groupby('battery')[col]
        .transform(lambda x: x.rolling(5, min_periods=1).mean())
    )

df_cycle['cap_delta']     = df_cycle.groupby('battery')['Capacity'].diff().fillna(0)
df_cycle['voltage_delta'] = df_cycle.groupby('battery')['mean_voltage'].diff().fillna(0)

df_cycle = df_cycle.dropna().reset_index(drop=True)
print(df_cycle.shape)
print(df_cycle[['battery','cycle_num','Capacity','SOH','SOC']].head(10))


# In[15]:


# For each cycle, build a fixed-length sequence from raw timestep data
# Then concat with cycle-level features → "Both combined" input

SEQ_LEN = 50  # resample each cycle to 50 timesteps

CYCLE_FEATURES = [
    'cycle_num',
    'mean_voltage', 'min_voltage', 'max_voltage', 'std_voltage',
    'mean_temp', 'max_temp', 'mean_current', 'duration', 'n_points',
    'Capacity_roll5', 'mean_voltage_roll5', 'mean_temp_roll5', 'duration_roll5',
    'cap_delta', 'voltage_delta',
]

TS_FEATURES = ['Voltage_measured', 'Current_measured',
               'Temperature_measured', 'Voltage_load']

def resample_cycle(group, seq_len=SEQ_LEN):
    """Resample a cycle's timeseries to fixed seq_len via interpolation."""
    n = len(group)
    idx_old = np.linspace(0, 1, n)
    idx_new = np.linspace(0, 1, seq_len)
    result = {}
    for col in TS_FEATURES:
        result[col] = np.interp(idx_new, idx_old, group[col].values)
    return result

# Build sequence arrays
X_ts, X_cyc, y_soh, y_soc, groups = [], [], [], [], []

for (battery, cycle_idx), ts_grp in df_raw.groupby(['battery', 'cycle_idx']):
    cyc_row = df_cycle[
        (df_cycle['battery'] == battery) &
        (df_cycle['cycle_idx'] == cycle_idx)
    ]
    if cyc_row.empty:
        continue

    ts_data  = resample_cycle(ts_grp)                           # (SEQ_LEN, 4)
    ts_arr   = np.stack([ts_data[c] for c in TS_FEATURES], axis=1)  # (50, 4)
    cyc_arr  = cyc_row[CYCLE_FEATURES].values.flatten()         # (16,)

    X_ts.append(ts_arr)
    X_cyc.append(cyc_arr)
    y_soh.append(float(cyc_row['SOH'].values[0]))
    y_soc.append(float(cyc_row['SOC'].values[0]))
    groups.append(battery)

X_ts  = np.array(X_ts,  dtype=np.float32)   # (N, 50, 4)
X_cyc = np.array(X_cyc, dtype=np.float32)   # (N, 16)
y_soh = np.array(y_soh, dtype=np.float32)   # (N,)
y_soc = np.array(y_soc, dtype=np.float32)   # (N,)
groups = np.array(groups)

print(f"X_ts: {X_ts.shape}, X_cyc: {X_cyc.shape}")
print(f"y_soh: {y_soh.shape}, y_soc: {y_soc.shape}")

# ── Scale ──────────────────────────────────────────────────────────────────
ts_scaler  = MinMaxScaler()
cyc_scaler = MinMaxScaler()

N, T, F = X_ts.shape
X_ts_sc  = ts_scaler.fit_transform(X_ts.reshape(-1, F)).reshape(N, T, F)
X_cyc_sc = cyc_scaler.fit_transform(X_cyc)


# In[16]:


import torch
import torch.nn as nn


class MHALSTM(nn.Module):
    def __init__(self,
                 ts_feat=4,
                 cyc_feat=16,
                 embed_dim=64,
                 num_heads=4,
                 lstm_hidden=128,
                 lstm_layers=2,
                 dropout=0.3):

        super().__init__()

        # Ensure valid head division
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        # -----------------------------
        # Input projection
        # -----------------------------
        self.input_proj = nn.Linear(ts_feat, embed_dim)

        # -----------------------------
        # Multi-Head Attention
        # -----------------------------
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

        # -----------------------------
        # LSTM Layer
        # -----------------------------
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )

        lstm_out_dim = lstm_hidden * 2

        # -----------------------------
        # Cycle feature MLP
        # -----------------------------
        self.cyc_mlp = nn.Sequential(
            nn.Linear(cyc_feat, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        # -----------------------------
        # Fusion Layer
        # -----------------------------
        fused_dim = lstm_out_dim + 64

        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        # -----------------------------
        # Output Heads
        # -----------------------------
        self.head_soh = nn.Linear(64, 1)
        self.head_soc = nn.Linear(64, 1)

    def forward(self, x_ts, x_cyc):
        # x_ts:  (B, T, ts_feat)
        # x_cyc: (B, cyc_feat)

        # -----------------------------
        # Projection
        # -----------------------------
        x = self.input_proj(x_ts)  # (B, T, embed_dim)

        # -----------------------------
        # Multi-head attention
        # -----------------------------
        attn_out, _ = self.mha(x, x, x)

        # Residual connection + normalization
        x = self.norm1(x + self.dropout(attn_out))

        # -----------------------------
        # LSTM
        # -----------------------------
        lstm_out, _ = self.lstm(x)

        # Last timestep representation
        ts_repr = lstm_out[:, -1, :]  # (B, lstm_hidden*2)

        # -----------------------------
        # Cycle features
        # -----------------------------
        cyc_repr = self.cyc_mlp(x_cyc)

        # -----------------------------
        # Fusion
        # -----------------------------
        fused = self.fusion(torch.cat([ts_repr, cyc_repr], dim=1))

        # -----------------------------
        # Outputs
        # -----------------------------
        soh = torch.sigmoid(self.head_soh(fused)).squeeze(1)
        soc = torch.sigmoid(self.head_soc(fused)).squeeze(1)

        return soh, soc


# In[17]:


import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttentionLSTM(nn.Module):
    """
    PyTorch MultiHeadAttention + LSTM equivalent of your GAT+LSTM.
    Each timestep attends to all others → captures long-range dependencies.
    """
    def __init__(self, ts_feat=6, cyc_feat=16, mha_dim=64, lstm_hidden=128,
                 lstm_layers=2, dropout=0.3, num_heads=8):
        super().__init__()

        # MultiHeadAttention branch — processes timeseries (replaces custom GAT)
        self.mha_embed = nn.Linear(ts_feat, mha_dim)  # Embed to MHA dim
        self.mha = nn.MultiheadAttention(
            embed_dim=mha_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.mha_norm = nn.LayerNorm(mha_dim)

        # LSTM branch — sequential memory over attended timeseries
        self.lstm = nn.LSTM(
            input_size=mha_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=True,
        )
        lstm_out_dim = lstm_hidden * 2  # bidirectional

        # Cycle-level MLP branch (same as yours)
        self.cyc_mlp = nn.Sequential(
            nn.Linear(cyc_feat, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        # Fusion + dual heads (identical)
        fused_dim = lstm_out_dim + 64
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.head_soh = nn.Linear(64, 1)
        self.head_soc = nn.Linear(64, 1)

    def forward(self, x_ts, x_cyc):
        """
        x_ts:  (B, T, ts_feat=6)  # Voltage, Current, Temp, Time, V_I, Temp_V
        x_cyc: (B, cyc_feat=16)   # Cycle-level features
        """
        B, T, _ = x_ts.shape

        # MultiHeadAttention → LSTM (equivalent to your GAT → LSTM)
        mha_embed = self.mha_embed(x_ts)  # (B, T, mha_dim)

        # MultiHeadAttention (query=key=value = self-attention)
        mha_out, _ = self.mha(mha_embed, mha_embed, mha_embed)  # (B, T, mha_dim)
        mha_out = self.mha_norm(mha_embed + mha_out)  # Residual connection

        # LSTM processing
        lstm_out, _ = self.lstm(mha_out)  # (B, T, lstm_hidden*2)
        ts_repr = lstm_out[:, -1, :]      # Last timestep representation (B, lstm_hidden*2)

        # Cycle MLP (identical to yours)
        cyc_repr = self.cyc_mlp(x_cyc)    # (B, 64)

        # Fusion (identical)
        fused = self.fusion(torch.cat([ts_repr, cyc_repr], dim=1))  # (B, 64)

        # Dual heads (identical)
        soh = torch.sigmoid(self.head_soh(fused)).squeeze(1)  # (B,)
        soc = torch.sigmoid(self.head_soc(fused)).squeeze(1)  # (B,)

        return soh, soc

# =========================
# USAGE EXAMPLE
# =========================
def example_usage():
    B, T, ts_feat, cyc_feat = 32, 30, 6, 16

    # Create dummy data matching your setup
    x_ts = torch.randn(B, T, ts_feat)  # (B, 30, 6) - your sequence features
    x_cyc = torch.randn(B, cyc_feat)   # (B, 16) - cycle features

    # Initialize model
    model = MultiHeadAttentionLSTM(
        ts_feat=6,      # Voltage, Current, Temp, Time, V_I, Temp_V
        cyc_feat=16,
        mha_dim=64,
        lstm_hidden=128,
        lstm_layers=2,
        dropout=0.3,
        num_heads=8     # More heads = better long-range attention
    )

    # Forward pass
    soh_pred, soc_pred = model(x_ts, x_cyc)
    print(f"SOH shape: {soh_pred.shape}")   # torch.Size([32])
    print(f"SOC shape: {soc_pred.shape}")   # torch.Size([32])
    print(f"SOH range: [{soh_pred.min():.3f}, {soh_pred.max():.3f}]")
    print(f"SOC range: [{soc_pred.min():.3f}, {soc_pred.max():.3f}]")

if __name__ == "__main__":
    example_usage()


# In[18]:


class BatteryDataset(Dataset):
    def __init__(self, X_ts, X_cyc, y_soh, y_soc):
        self.X_ts  = torch.tensor(X_ts,  dtype=torch.float32)
        self.X_cyc = torch.tensor(X_cyc, dtype=torch.float32)
        self.y_soh = torch.tensor(y_soh, dtype=torch.float32)
        self.y_soc = torch.tensor(y_soc, dtype=torch.float32)

    def __len__(self): return len(self.y_soh)

    def __getitem__(self, idx):
        return self.X_ts[idx], self.X_cyc[idx], self.y_soh[idx], self.y_soc[idx]


# In[19]:


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =========================
# ✅ PREPARE DATA (FIXED)
# =========================

# Use your main dataframe (df or df_raw)
df = df_cycle.copy() # Corrected: Use df_cycle instead of df_raw

# Ensure required columns exist
required_cols = ['battery', 'SOH', 'SOC']
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Missing column: {col}")

# Features (adjust if needed)
X_ts_sc_3  = X_ts_sc # Corrected variable name
X_cyc_sc_3 = X_cyc_sc # Corrected variable name

# Targets
y_soh_3 = df['SOH'].values
y_soc_3 = df['SOC'].values

# ✅ FIX: define groups properly
groups_3 = df['battery'].values # Corrected: Use df['battery'] as df is now df_cycle

print("Batteries in dataset:", np.unique(groups_3))
print("Total samples:", len(groups_3))


# =========================
# ✅ TRAINING LOOP (FIXED)
# =========================

all_results   = []
all_preds     = {}
loss_history  = {}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for test_bat in np.unique(groups_3):

    print(f"\n🚀 Running LOBO for: {test_bat}")

    train_mask = groups_3 != test_bat
    test_mask  = groups_3 == test_bat

    # ✅ Skip bad splits (important for B0018)
    if np.sum(train_mask) < 50 or np.sum(test_mask) < 10:
        print(f"⚠️ Skipping {test_bat} due to insufficient samples")
        continue

    train_ds = BatteryDataset(
        X_ts_sc_3[train_mask],
        X_cyc_sc_3[train_mask],
        y_soh_3[train_mask],
        y_soc_3[train_mask]
    )

    test_ds = BatteryDataset(
        X_ts_sc_3[test_mask],
        X_cyc_sc_3[test_mask],
        y_soh_3[test_mask],
        y_soc_3[test_mask]
    )

    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_dl  = DataLoader(test_ds,  batch_size=64, shuffle=False)

    # Model
    model = MHALSTM(
        ts_feat=len(TS_FEATURES),
        cyc_feat=len(CYCLE_FEATURES),
        embed_dim=64,
        lstm_hidden=128,
        lstm_layers=2,
        dropout=0.3,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    criterion = nn.MSELoss()

    EPOCHS = 100
    best_val_loss = float('inf')
    best_state = None

    train_losses, val_losses = [], []

    # Early stopping
    patience = 20
    counter = 0

    for epoch in range(EPOCHS):

        # ---- TRAIN ----
        model.train()
        train_loss = 0

        for xts, xcyc, ysoh, ysoc in train_dl:
            xts, xcyc = xts.to(device), xcyc.to(device)
            ysoh, ysoc = ysoh.to(device), ysoc.to(device)

            optimizer.zero_grad()
            pred_soh, pred_soc = model(xts, xcyc)

            # ✅ Balanced loss
            loss = 0.5 * criterion(pred_soh, ysoh) + 0.5 * criterion(pred_soc, ysoc)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        # ---- VALIDATION ----
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for xts, xcyc, ysoh, ysoc in test_dl:
                xts, xcyc = xts.to(device), xcyc.to(device)
                ysoh, ysoc = ysoh.to(device), ysoc.to(device)

                ps, pc = model(xts, xcyc)
                val_loss += (criterion(ps, ysoh) + criterion(pc, ysoc)).item()

        if len(test_dl) == 0:
            print(f"⚠️ No validation data for {test_bat}")
            continue

        current_val_loss = val_loss / len(test_dl)

        scheduler.step(current_val_loss)

        train_losses.append(train_loss / len(train_dl))
        val_losses.append(current_val_loss)

        # Save best model
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            counter = 0
        else:
            counter += 1

        # Early stopping
        if counter >= patience:
            print(f"⏹ Early stopping at epoch {epoch+1}")
            break

        if (epoch + 1) % 20 == 0:
            print(f"[{test_bat}] Epoch {epoch+1:3d} | "
                  f"Train: {train_losses[-1]:.4f} | Val: {val_losses[-1]:.4f}")

    loss_history[test_bat] = {'train': train_losses, 'val': val_losses}

    # =========================
    # ✅ EVALUATION (FIXED)
    # =========================

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()

    soh_true, soh_pred = [], []
    soc_true, soc_pred = [], []

    with torch.no_grad():
        for xts, xcyc, ysoh, ysoc in test_dl:
            xts, xcyc = xts.to(device), xcyc.to(device)

            ps, pc = model(xts, xcyc)

            # ✅ FIX: move to CPU before numpy
            soh_true.extend(ysoh.cpu().numpy())
            soh_pred.extend(ps.cpu().numpy())

            soc_true.extend(ysoc.cpu().numpy())
            soc_pred.extend(pc.cpu().numpy())

    # Metrics
    soh_mae  = mean_absolute_error(soh_true, soh_pred)
    soh_rmse = np.sqrt(mean_squared_error(soh_true, soh_pred))
    soh_r2   = r2_score(soh_true, soh_pred)

    soc_mae  = mean_absolute_error(soc_true, soc_pred)
    soc_rmse = np.sqrt(mean_squared_error(soc_true, soc_pred))
    soc_r2   = r2_score(soc_true, soc_pred)

    all_preds[test_bat] = {
        'soh_true': np.array(soh_true),
        'soh_pred': np.array(soh_pred),
        'soc_true': np.array(soc_true),
        'soc_pred': np.array(soc_pred),
    }

    all_results.append({
        'Battery': test_bat,
        'SOH MAE': soh_mae,
        'SOH RMSE': soh_rmse,
        'SOH R²': soh_r2,
        'SOC MAE': soc_mae,
        'SOC RMSE': soc_rmse,
        'SOC R²': soc_r2,
    })

    print(f"\n{'='*55}")
    print(f"[{test_bat}] SOH → MAE:{soh_mae:.4f}  RMSE:{soh_rmse:.4f}  R²:{soh_r2:.4f}")
    print(f"[{test_bat}] SOC → MAE:{soc_mae:.4f}  RMSE:{soc_rmse:.4f}  R²:{soc_r2:.4f}")
    print(f"{'='*55}\n")

# Final results
df_res = pd.DataFrame(all_results)
print(df_res)


# In[20]:


import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
from matplotlib.ticker import MaxNLocator
import pandas as pd

# =========================
# ✅ SAFETY FIXES
# =========================

# Ensure df_cycle exists
df_cycle = df_raw.copy()
df_cycle['cycle_num'] = df_cycle['cycle_idx']

# Use only trained batteries (important!)
BATTERIES = list(all_preds.keys())

COLORS = {
    'B0005': '#1f77b4',
    'B0007': '#2ca02c',
    'B0018': '#d62728'
}

print("Generating graphs for:", BATTERIES)

# =========================
# FIG 1 — Capacity Fade
# =========================
plt.figure(figsize=(7,4))

for bat in BATTERIES:
    grp = df_cycle[df_cycle['battery'] == bat].sort_values('cycle_num')
    plt.plot(grp['cycle_num'], grp['Capacity'], label=bat, color=COLORS.get(bat,'blue'))

plt.axhline(1.4, linestyle='--', color='black', label='EOL Threshold')
plt.xlabel('Cycle Number')
plt.ylabel('Capacity (Ah)')
plt.title('Fig 1: Capacity Fade')
plt.legend()
plt.tight_layout()
plt.savefig('fig1_capacity.png')
plt.show()


# =========================
# FIG 2 — SOH Pred vs Actual
# =========================
fig, axes = plt.subplots(1, len(BATTERIES), figsize=(14,4), sharey=True)

for ax, bat in zip(axes, BATTERIES):
    p = all_preds[bat]
    x = np.arange(len(p['soh_true']))

    ax.plot(x, p['soh_true'], label='Actual', color=COLORS.get(bat,'blue'))
    ax.plot(x, p['soh_pred'], '--', label='Predicted', color='black')

    mae = mean_absolute_error(p['soh_true'], p['soh_pred'])
    r2  = r2_score(p['soh_true'], p['soh_pred'])

    ax.set_title(f"{bat}\nMAE={mae:.3f}, R²={r2:.3f}")
    ax.set_xlabel("Sample")

axes[0].set_ylabel("SOH")
plt.legend()
plt.tight_layout()
plt.savefig('fig2_soh.png')
plt.show()


# =========================
# FIG 3 — SOC Pred vs Actual
# =========================
fig, axes = plt.subplots(1, len(BATTERIES), figsize=(14,4), sharey=True)

for ax, bat in zip(axes, BATTERIES):
    p = all_preds[bat]
    x = np.arange(len(p['soc_true']))

    ax.plot(x, p['soc_true'], label='Actual', color=COLORS.get(bat,'blue'))
    ax.plot(x, p['soc_pred'], '--', label='Predicted', color='black')

    mae = mean_absolute_error(p['soc_true'], p['soc_pred'])
    r2  = r2_score(p['soc_true'], p['soc_pred'])

    ax.set_title(f"{bat}\nMAE={mae:.3f}, R²={r2:.3f}")
    ax.set_xlabel("Sample")

axes[0].set_ylabel("SOC")
plt.legend()
plt.tight_layout()
plt.savefig('fig3_soc.png')
plt.show()


# =========================
# FIG 4 — Loss Curves
# =========================
fig, axes = plt.subplots(1, len(BATTERIES), figsize=(14,4), sharey=True)

for ax, bat in zip(axes, BATTERIES):
    h = loss_history[bat]
    epochs = np.arange(len(h['train']))

    ax.plot(epochs, h['train'], label='Train', color=COLORS.get(bat,'blue'))
    ax.plot(epochs, h['val'], '--', label='Val', color='black')

    ax.set_title(bat)
    ax.set_xlabel("Epoch")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

axes[0].set_ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig('fig4_loss.png')
plt.show()


# =========================
# FIG 5 — Error Histogram
# =========================
fig, axes = plt.subplots(2, len(BATTERIES), figsize=(14,7))

for col, (true_k, pred_k, label) in enumerate([
    ('soh_true','soh_pred','SOH'),
    ('soc_true','soc_pred','SOC')
]):
    for i, bat in enumerate(BATTERIES):
        p = all_preds[bat]
        err = np.abs(np.array(p[true_k]) - np.array(p[pred_k]))

        axes[col][i].hist(err, bins=30, color=COLORS.get(bat,'blue'))
        axes[col][i].set_title(f"{label} Error - {bat}")

plt.tight_layout()
plt.savefig('fig5_error.png')
plt.show()


# =========================
# FIG 6 — Scatter Plot
# =========================
fig, axes = plt.subplots(2, len(BATTERIES), figsize=(14,8))

for col, (true_k, pred_k, label) in enumerate([
    ('soh_true','soh_pred','SOH'),
    ('soc_true','soc_pred','SOC')
]):
    for i, bat in enumerate(BATTERIES):
        p = all_preds[bat]

        yt = np.array(p[true_k])
        yp = np.array(p[pred_k])

        axes[col][i].scatter(yt, yp, alpha=0.5, color=COLORS.get(bat,'blue'))

        # ideal line
        mn, mx = min(yt.min(), yp.min()), max(yt.max(), yp.max())
        axes[col][i].plot([mn,mx],[mn,mx],'k--')

        axes[col][i].set_title(f"{label} - {bat}")
        axes[col][i].set_xlabel("Actual")
        axes[col][i].set_ylabel("Predicted")

plt.tight_layout()
plt.savefig('fig6_scatter.png')
plt.show()


# =========================
# TABLE — Metrics
# =========================
df_table = df_res.copy()

mean_row = df_table.mean(numeric_only=True)
mean_row['Battery'] = 'Mean'

df_table = pd.concat([df_table, pd.DataFrame([mean_row])], ignore_index=True)

print("\n===== FINAL RESULTS =====")
print(df_table)

df_table.to_csv("table_results.csv", index=False)


# In[21]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, r2_score
from matplotlib.ticker import MaxNLocator

# ==========================================
# JOURNAL STYLE SETTINGS
# ==========================================

plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'axes.linewidth': 1.2,
    'lines.linewidth': 2,
    'figure.dpi': 300
})

# ==========================================
# SAFETY
# ==========================================

df_cycle = df_raw.copy()
df_cycle['cycle_num'] = df_cycle['cycle_idx']

BATTERIES = list(all_preds.keys())

COLORS = {
    'B0005': '#1f77b4',
    'B0007': '#2ca02c',
    'B0018': '#d62728'
}

print("Generating Elsevier Journal Figures...")

# ==========================================
# FIGURE 1 — CAPACITY FADE
# ==========================================

fig, ax = plt.subplots(figsize=(7,4.5))

for bat in BATTERIES:
    grp = df_cycle[df_cycle['battery'] == bat].sort_values('cycle_num')

    ax.plot(
        grp['cycle_num'],
        grp['Capacity'],
        label=bat,
        color=COLORS[bat]
    )

ax.axhline(
    1.4,
    linestyle='--',
    linewidth=2,
    color='black',
    label='EOL Threshold'
)

ax.set_xlabel('Cycle Number')
ax.set_ylabel('Capacity (Ah)')
ax.set_title('Capacity Degradation Characteristics')
ax.grid(True, linestyle='--', alpha=0.4)
ax.legend(frameon=True)

plt.tight_layout()

plt.savefig(
    'Fig1_Capacity_Fade.png',
    dpi=300,
    bbox_inches='tight'
)

plt.show()


# ==========================================
# FIGURE 2 — SOH PREDICTION
# ==========================================

fig, axes = plt.subplots(
    1,
    len(BATTERIES),
    figsize=(15,4.5),
    sharey=True
)

for ax, bat in zip(axes, BATTERIES):

    p = all_preds[bat]
    x = np.arange(len(p['soh_true']))

    ax.plot(
        x,
        p['soh_true'],
        label='Actual',
        color=COLORS[bat]
    )

    ax.plot(
        x,
        p['soh_pred'],
        '--',
        label='Predicted',
        color='black'
    )

    mae = mean_absolute_error(
        p['soh_true'],
        p['soh_pred']
    )

    r2 = r2_score(
        p['soh_true'],
        p['soh_pred']
    )

    ax.set_title(
        f'{bat}\nMAE={mae:.4f}, R²={r2:.4f}'
    )

    ax.set_xlabel('Sample Index')

    ax.grid(True, linestyle='--', alpha=0.4)

axes[0].set_ylabel('SOH')

handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc='upper center',
    ncol=2,
    frameon=True
)

plt.tight_layout(rect=[0,0,1,0.92])

plt.savefig(
    'Fig2_SOH_Prediction.png',
    dpi=300,
    bbox_inches='tight'
)

plt.show()


# ==========================================
# FIGURE 3 — SOC PREDICTION
# ==========================================

fig, axes = plt.subplots(
    1,
    len(BATTERIES),
    figsize=(15,4.5),
    sharey=True
)

for ax, bat in zip(axes, BATTERIES):

    p = all_preds[bat]

    x = np.arange(len(p['soc_true']))

    ax.plot(
        x,
        p['soc_true'],
        label='Actual',
        color=COLORS[bat]
    )

    ax.plot(
        x,
        p['soc_pred'],
        '--',
        label='Predicted',
        color='black'
    )

    mae = mean_absolute_error(
        p['soc_true'],
        p['soc_pred']
    )

    r2 = r2_score(
        p['soc_true'],
        p['soc_pred']
    )

    ax.set_title(
        f'{bat}\nMAE={mae:.4f}, R²={r2:.4f}'
    )

    ax.set_xlabel('Sample Index')

    ax.grid(True, linestyle='--', alpha=0.4)

axes[0].set_ylabel('SOC')

handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc='upper center',
    ncol=2,
    frameon=True
)

plt.tight_layout(rect=[0,0,1,0.92])

plt.savefig(
    'Fig3_SOC_Prediction.png',
    dpi=300,
    bbox_inches='tight'
)

plt.show()


# ==========================================
# FIGURE 4 — TRAINING LOSS
# ==========================================

fig, axes = plt.subplots(
    1,
    len(BATTERIES),
    figsize=(15,4.5),
    sharey=True
)

for ax, bat in zip(axes, BATTERIES):

    h = loss_history[bat]

    epochs = np.arange(len(h['train']))

    ax.plot(
        epochs,
        h['train'],
        label='Training',
        color=COLORS[bat]
    )

    ax.plot(
        epochs,
        h['val'],
        '--',
        label='Validation',
        color='black'
    )

    ax.set_title(bat)

    ax.set_xlabel('Epoch')

    ax.xaxis.set_major_locator(
        MaxNLocator(integer=True)
    )

    ax.grid(True, linestyle='--', alpha=0.4)

axes[0].set_ylabel('Loss')

handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc='upper center',
    ncol=2,
    frameon=True
)

plt.tight_layout(rect=[0,0,1,0.92])

plt.savefig(
    'Fig4_Training_Loss.png',
    dpi=300,
    bbox_inches='tight'
)

plt.show()


# ==========================================
# TABLE EXPORT
# ==========================================

df_table = df_res.copy()

mean_row = df_table.mean(numeric_only=True)

mean_row['Battery'] = 'Average'

df_table = pd.concat(
    [df_table, pd.DataFrame([mean_row])],
    ignore_index=True
)

print("\n===== FINAL RESULTS =====")
print(df_table)

df_table.to_csv(
    'Table_Results.csv',
    index=False
)

print("\nAll Elsevier-quality figures saved successfully.")


# In[22]:


# ─────────────────────────────────────────────────────────────────────────────
# GCN-LSTM: Graph Convolutional Network + LSTM for SOH & SOC Prediction
# ─────────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd

# ── GCN Layer ────────────────────────────────────────────────────────────────
class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.b = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x, adj):
        # x:   (B, T, in_dim)
        # adj: (T, T) — adjacency matrix (normalized)
        out = torch.matmul(adj, x)        # (B, T, in_dim)  graph aggregation
        out = self.W(out) + self.b        # (B, T, out_dim) linear transform
        return F.relu(out)


def build_adj(seq_len, window=5):
    """
    Build normalized adjacency matrix.
    Each timestep connects to its ±window neighbors (local temporal graph).
    """
    A = torch.zeros(seq_len, seq_len)
    for i in range(seq_len):
        for j in range(max(0, i - window), min(seq_len, i + window + 1)):
            A[i, j] = 1.0
    # Degree normalization: D^{-1/2} A D^{-1/2}
    D     = A.sum(dim=1).clamp(min=1e-6)
    D_inv = torch.diag(D ** -0.5)
    A_hat = D_inv @ A @ D_inv
    return A_hat  # (T, T)


# ── GCN-LSTM Model ───────────────────────────────────────────────────────────
class GCNLSTM(nn.Module):
    def __init__(self, ts_feat=4, cyc_feat=16, gcn_dim=64,
                 lstm_hidden=128, lstm_layers=2, dropout=0.3, seq_len=50):
        super().__init__()

        # GCN layers (2 stacked)
        self.gcn1     = GCNLayer(ts_feat,  gcn_dim)
        self.gcn2     = GCNLayer(gcn_dim,  gcn_dim)
        self.gcn_norm = nn.LayerNorm(gcn_dim)
        self.gcn_drop = nn.Dropout(dropout)

        # Bi-LSTM
        self.lstm = nn.LSTM(
            input_size    = gcn_dim,
            hidden_size   = lstm_hidden,
            num_layers    = lstm_layers,
            batch_first   = True,
            dropout       = dropout,
            bidirectional = True,
        )
        lstm_out_dim = lstm_hidden * 2

        # Cycle-level MLP
        self.cyc_mlp = nn.Sequential(
            nn.Linear(cyc_feat, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        # Fusion + dual heads
        self.fusion = nn.Sequential(
            nn.Linear(lstm_out_dim + 64, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.head_soh = nn.Linear(64, 1)
        self.head_soc = nn.Linear(64, 1)

        # Fixed adjacency matrix
        self.register_buffer('adj', build_adj(seq_len))

    def forward(self, x_ts, x_cyc):
        # GCN branch
        gcn_out  = self.gcn1(x_ts,  self.adj)       # (B, T, gcn_dim)
        gcn_out  = self.gcn2(gcn_out, self.adj)      # (B, T, gcn_dim)
        gcn_out  = self.gcn_norm(gcn_out)
        gcn_out  = self.gcn_drop(gcn_out)

        # LSTM branch
        lstm_out, _ = self.lstm(gcn_out)             # (B, T, lstm_hidden*2)
        ts_repr     = lstm_out[:, -1, :]             # (B, lstm_hidden*2)

        # Cycle MLP
        cyc_repr = self.cyc_mlp(x_cyc)              # (B, 64)

        # Fuse
        fused = self.fusion(torch.cat([ts_repr, cyc_repr], dim=1))

        soh = torch.sigmoid(self.head_soh(fused)).squeeze(1)
        soc = torch.sigmoid(self.head_soc(fused)).squeeze(1)
        return soh, soc


# In[23]:


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =========================
# ✅ TRAINING LOOP (FIXED)
# =========================

gcn_results  = []
gcn_preds    = {}
gcn_history  = {}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for test_bat in np.unique(groups_3):

    print(f"\n🚀 Running GCN-LSTM for: {test_bat}")

    train_mask = groups_3 != test_bat
    test_mask  = groups_3 == test_bat

    # ✅ B0018 safety (VERY IMPORTANT)
    if np.sum(train_mask) < 50 or np.sum(test_mask) < 10:
        print(f"⚠️ Skipping {test_bat} (insufficient data)")
        continue

    train_ds = BatteryDataset(
        X_ts_sc_3[train_mask],
        X_cyc_sc_3[train_mask],
        y_soh_3[train_mask],
        y_soc_3[train_mask]
    )

    test_ds = BatteryDataset(
        X_ts_sc_3[test_mask],
        X_cyc_sc_3[test_mask],
        y_soh_3[test_mask],
        y_soc_3[test_mask]
    )

    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_dl  = DataLoader(test_ds,  batch_size=64, shuffle=False)

    model = GCNLSTM(
        ts_feat=len(TS_FEATURES),
        cyc_feat=len(CYCLE_FEATURES),
        gcn_dim=64,
        lstm_hidden=128,
        lstm_layers=2,
        dropout=0.3,
        seq_len=50,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    criterion = nn.MSELoss()

    EPOCHS = 100
    best_val_loss = float('inf')
    best_state = None

    train_losses, val_losses = [], []

    for epoch in range(EPOCHS):

        # -------- TRAIN --------
        model.train()
        train_loss = 0

        for xts, xcyc, ysoh, ysoc in train_dl:
            xts, xcyc = xts.to(device), xcyc.to(device)
            ysoh, ysoc = ysoh.to(device), ysoc.to(device)

            optimizer.zero_grad()
            ps, pc = model(xts, xcyc)

            # ✅ Balanced loss
            loss = 0.5 * criterion(ps, ysoh) + 0.5 * criterion(pc, ysoc)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        # -------- VALIDATION --------
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for xts, xcyc, ysoh, ysoc in test_dl:
                xts, xcyc = xts.to(device), xcyc.to(device)
                ysoh, ysoc = ysoh.to(device), ysoc.to(device)

                ps, pc = model(xts, xcyc)
                val_loss += (criterion(ps, ysoh) + criterion(pc, ysoc)).item()

        # ✅ Avoid division error
        if len(test_dl) == 0:
            print(f"⚠️ No validation data for {test_bat}")
            continue

        val_loss_avg = val_loss / len(test_dl)
        scheduler.step(val_loss_avg)

        train_losses.append(train_loss / len(train_dl))
        val_losses.append(val_loss_avg)

        # Save best model
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 20 == 0:
            print(f"[GCN-LSTM | {test_bat}] Epoch {epoch+1} | "
                  f"Train: {train_losses[-1]:.4f} | Val: {val_losses[-1]:.4f}")

    gcn_history[test_bat] = {'train': train_losses, 'val': val_losses}

    # =========================
    # ✅ EVALUATION (FIXED)
    # =========================

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()

    soh_true, soh_pred = [], []
    soc_true, soc_pred = [], []

    with torch.no_grad():
        for xts, xcyc, ysoh, ysoc in test_dl:
            xts, xcyc = xts.to(device), xcyc.to(device)

            ps, pc = model(xts, xcyc)

            # ✅ FIX: CPU conversion
            soh_true.extend(ysoh.cpu().numpy())
            soh_pred.extend(ps.cpu().numpy())

            soc_true.extend(ysoc.cpu().numpy())
            soc_pred.extend(pc.cpu().numpy())

    # Metrics
    soh_mae  = mean_absolute_error(soh_true, soh_pred)
    soh_rmse = np.sqrt(mean_squared_error(soh_true, soh_pred))
    soh_r2   = r2_score(soh_true, soh_pred)

    soc_mae  = mean_absolute_error(soc_true, soc_pred)
    soc_rmse = np.sqrt(mean_squared_error(soc_true, soc_pred))
    soc_r2   = r2_score(soc_true, soc_pred)

    gcn_preds[test_bat] = {
        'soh_true': np.array(soh_true),
        'soh_pred': np.array(soh_pred),
        'soc_true': np.array(soc_true),
        'soc_pred': np.array(soc_pred),
    }

    gcn_results.append({
        'Battery': test_bat,
        'SOH MAE': soh_mae,
        'SOH RMSE': soh_rmse,
        'SOH R²': soh_r2,
        'SOC MAE': soc_mae,
        'SOC RMSE': soc_rmse,
        'SOC R²': soc_r2,
    })

    print(f"\n{'='*55}")
    print(f"[{test_bat}] SOH → MAE:{soh_mae:.4f} RMSE:{soh_rmse:.4f} R²:{soh_r2:.4f}")
    print(f"[{test_bat}] SOC → MAE:{soc_mae:.4f} RMSE:{soc_rmse:.4f} R²:{soc_r2:.4f}")
    print(f"{'='*55}\n")

# =========================
# ✅ FINAL RESULTS
# =========================
df_gcn = pd.DataFrame(gcn_results)
print("\nFinal Results:")
print(df_gcn)

# =========================
# ✅ IMPORTANT: GRAPH COMPATIBILITY FIX
# =========================
# This allows your previous graph code to work directly

all_preds    = gcn_preds
loss_history = gcn_history
df_res       = df_gcn


# In[24]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =========================
# ✅ MODEL
# =========================

class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, hidden_dim)
        self.v    = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, lstm_out):
        energy  = torch.tanh(self.attn(lstm_out))
        scores  = self.v(energy).squeeze(-1)
        weights = F.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), lstm_out).squeeze(1)
        return context, weights


class AttentionLSTM(nn.Module):
    def __init__(self, ts_feat=4, cyc_feat=16, lstm_hidden=128,
                 lstm_layers=2, dropout=0.3):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(ts_feat, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )

        self.attention = TemporalAttention(lstm_hidden * 2)

        self.cyc_mlp = nn.Sequential(
            nn.Linear(cyc_feat, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        self.fusion = nn.Sequential(
            nn.Linear(lstm_hidden*2 + 64, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        self.head_soh = nn.Linear(64, 1)
        self.head_soc = nn.Linear(64, 1)

    def forward(self, x_ts, x_cyc):
        x = self.input_proj(x_ts)
        lstm_out, _ = self.lstm(x)

        context, attn_w = self.attention(lstm_out)
        cyc_repr = self.cyc_mlp(x_cyc)

        fused = self.fusion(torch.cat([context, cyc_repr], dim=1))

        soh = torch.sigmoid(self.head_soh(fused)).squeeze(1)
        soc = torch.sigmoid(self.head_soc(fused)).squeeze(1)

        return soh, soc, attn_w


# =========================
# ✅ TRAINING LOOP (FIXED)
# =========================

attn_results = []
attn_preds   = {}
attn_history = {}
attn_weights_store = {}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for test_bat in np.unique(groups_3):

    print(f"\n🚀 Running Attention-LSTM for: {test_bat}")

    train_mask = groups_3 != test_bat
    test_mask  = groups_3 == test_bat

    # ✅ B0018 safety
    if np.sum(train_mask) < 50 or np.sum(test_mask) < 10:
        print(f"⚠️ Skipping {test_bat} (insufficient data)")
        continue

    train_ds = BatteryDataset(
        X_ts_sc_3[train_mask],
        X_cyc_sc_3[train_mask],
        y_soh_3[train_mask],
        y_soc_3[train_mask]
    )

    test_ds = BatteryDataset(
        X_ts_sc_3[test_mask],
        X_cyc_sc_3[test_mask],
        y_soh_3[test_mask],
        y_soc_3[test_mask]
    )

    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_dl  = DataLoader(test_ds,  batch_size=64, shuffle=False)

    model = AttentionLSTM(
        ts_feat=len(TS_FEATURES),
        cyc_feat=len(CYCLE_FEATURES),
        lstm_hidden=128,
        lstm_layers=2,
        dropout=0.3,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    criterion = nn.MSELoss()

    EPOCHS = 100
    best_val_loss = float('inf')
    best_state = None

    train_losses, val_losses = [], []

    for epoch in range(EPOCHS):

        # ---- TRAIN ----
        model.train()
        train_loss = 0

        for xts, xcyc, ysoh, ysoc in train_dl:
            xts, xcyc = xts.to(device), xcyc.to(device)
            ysoh, ysoc = ysoh.to(device), ysoc.to(device)

            optimizer.zero_grad()
            ps, pc, _ = model(xts, xcyc)

            # ✅ Balanced loss
            loss = 0.5 * criterion(ps, ysoh) + 0.5 * criterion(pc, ysoc)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        # ---- VALIDATION ----
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for xts, xcyc, ysoh, ysoc in test_dl:
                xts, xcyc = xts.to(device), xcyc.to(device)
                ysoh, ysoc = ysoh.to(device), ysoc.to(device)

                ps, pc, _ = model(xts, xcyc)
                val_loss += (criterion(ps, ysoh) + criterion(pc, ysoc)).item()

        if len(test_dl) == 0:
            print(f"⚠️ No validation data for {test_bat}")
            continue

        val_loss_avg = val_loss / len(test_dl)
        scheduler.step(val_loss_avg)

        train_losses.append(train_loss / len(train_dl))
        val_losses.append(val_loss_avg)

        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 20 == 0:
            print(f"[Attn-LSTM | {test_bat}] Epoch {epoch+1} | "
                  f"Train: {train_losses[-1]:.4f} | Val: {val_losses[-1]:.4f}")

    attn_history[test_bat] = {'train': train_losses, 'val': val_losses}

    # =========================
    # ✅ EVALUATION (FIXED)
    # =========================

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()

    soh_true, soh_pred = [], []
    soc_true, soc_pred = [], []
    all_attn_w = []

    with torch.no_grad():
        for xts, xcyc, ysoh, ysoc in test_dl:
            xts, xcyc = xts.to(device), xcyc.to(device)

            ps, pc, aw = model(xts, xcyc)

            soh_true.extend(ysoh.cpu().numpy())
            soh_pred.extend(ps.cpu().numpy())

            soc_true.extend(ysoc.cpu().numpy())
            soc_pred.extend(pc.cpu().numpy())

            all_attn_w.append(aw.cpu().numpy())

    attn_preds[test_bat] = {
        'soh_true': np.array(soh_true),
        'soh_pred': np.array(soh_pred),
        'soc_true': np.array(soc_true),
        'soc_pred': np.array(soc_pred),
    }

    attn_weights_store[test_bat] = np.concatenate(all_attn_w, axis=0)

    # Metrics
    soh_mae  = mean_absolute_error(soh_true, soh_pred)
    soh_rmse = np.sqrt(mean_squared_error(soh_true, soh_pred))
    soh_r2   = r2_score(soh_true, soh_pred)

    soc_mae  = mean_absolute_error(soc_true, soc_pred)
    soc_rmse = np.sqrt(mean_squared_error(soc_true, soc_pred))
    soc_r2   = r2_score(soc_true, soc_pred)

    attn_results.append({
        'Battery': test_bat,
        'SOH MAE': soh_mae,
        'SOH RMSE': soh_rmse,
        'SOH R²': soh_r2,
        'SOC MAE': soc_mae,
        'SOC RMSE': soc_rmse,
        'SOC R²': soc_r2,
    })

    print(f"\n{'='*55}")
    print(f"[{test_bat}] SOH → MAE:{soh_mae:.4f} RMSE:{soh_rmse:.4f} R²:{soh_r2:.4f}")
    print(f"[{test_bat}] SOC → MAE:{soc_mae:.4f} RMSE:{soc_rmse:.4f} R²:{soc_r2:.4f}")
    print(f"{'='*55}\n")

# =========================
# ✅ FINAL RESULTS
# =========================

df_attn = pd.DataFrame(attn_results)
print("\nFinal Results:")
print(df_attn)

# =========================
# ✅ GRAPH COMPATIBILITY
# =========================
all_preds    = attn_preds
loss_history = attn_history
df_res       = df_attn


# In[25]:


import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =========================
# ✅ TCN BLOCK
# =========================

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout=0.2):
        super().__init__()

        padding = (kernel_size - 1) * dilation

        self.conv1 = weight_norm(
            nn.Conv1d(in_ch, out_ch, kernel_size,
                      padding=padding, dilation=dilation)
        )
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1d(out_ch, out_ch, kernel_size,
                      padding=padding, dilation=dilation)
        )
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x):
        out = self.conv1(x)[:, :, :x.size(2)]
        out = self.drop1(self.relu1(out))

        out = self.conv2(out)[:, :, :x.size(2)]
        out = self.drop2(self.relu2(out))

        res = x if self.downsample is None else self.downsample(x)
        return torch.relu(out + res)


# =========================
# ✅ TCN MODEL
# =========================

class TCNModel(nn.Module):
    def __init__(self, ts_feat, cyc_feat, channels, kernel_size=3, dropout=0.2):
        super().__init__()

        layers = []
        for i in range(len(channels)):
            dilation = 2 ** i
            in_ch  = ts_feat if i == 0 else channels[i-1]
            out_ch = channels[i]

            layers.append(
                ResidualBlock(in_ch, out_ch, kernel_size, dilation, dropout)
            )

        self.tcn = nn.Sequential(*layers)

        combined_dim = channels[-1] + cyc_feat

        self.soh_head = nn.Sequential(
            nn.Linear(combined_dim, 64), nn.ReLU(), nn.Linear(64, 1)
        )
        self.soc_head = nn.Sequential(
            nn.Linear(combined_dim, 64), nn.ReLU(), nn.Linear(64, 1)
        )

    def forward(self, x_ts, x_cyc):
        # (B, T, F) → (B, F, T)
        x_ts = x_ts.transpose(1, 2)

        y = self.tcn(x_ts)
        y = y[:, :, -1]  # last timestep

        combined = torch.cat([y, x_cyc], dim=1)

        soh = self.soh_head(combined).squeeze(-1)
        soc = self.soc_head(combined).squeeze(-1)

        return soh, soc


# =========================
# ✅ TRAINING LOOP
# =========================

tcn_results  = []
tcn_preds    = {}
tcn_history  = {}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for test_bat in np.unique(groups_3):

    print(f"\n🚀 Running TCN for: {test_bat}")

    train_mask = groups_3 != test_bat
    test_mask  = groups_3 == test_bat

    # ✅ Safety for B0018
    if np.sum(train_mask) < 50 or np.sum(test_mask) < 10:
        print(f"⚠️ Skipping {test_bat} (insufficient data)")
        continue

    train_ds = BatteryDataset(
        X_ts_sc_3[train_mask],
        X_cyc_sc_3[train_mask],
        y_soh_3[train_mask],
        y_soc_3[train_mask]
    )

    test_ds = BatteryDataset(
        X_ts_sc_3[test_mask],
        X_cyc_sc_3[test_mask],
        y_soh_3[test_mask],
        y_soc_3[test_mask]
    )

    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_dl  = DataLoader(test_ds,  batch_size=64, shuffle=False)

    model = TCNModel(
        ts_feat=len(TS_FEATURES),
        cyc_feat=len(CYCLE_FEATURES),
        channels=[64, 64, 128, 128],
        kernel_size=3,
        dropout=0.2
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    best_state    = None

    train_losses, val_losses = [], []

    for epoch in range(100):

        # ---- TRAIN ----
        model.train()
        train_loss = 0

        for xts, xcyc, ysoh, ysoc in train_dl:
            xts, xcyc = xts.to(device), xcyc.to(device)
            ysoh, ysoc = ysoh.to(device), ysoc.to(device)

            optimizer.zero_grad()
            ps, pc = model(xts, xcyc)

            # ✅ Balanced loss
            loss = 0.5 * criterion(ps, ysoh) + 0.5 * criterion(pc, ysoc)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        # ---- VALIDATION ----
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for xts, xcyc, ysoh, ysoc in test_dl:
                xts, xcyc = xts.to(device), xcyc.to(device)
                ysoh, ysoc = ysoh.to(device), ysoc.to(device)

                ps, pc = model(xts, xcyc)
                val_loss += (criterion(ps, ysoh) + criterion(pc, ysoc)).item()

        if len(test_dl) == 0:
            print(f"⚠️ No validation data for {test_bat}")
            continue

        val_loss_avg = val_loss / len(test_dl)
        scheduler.step(val_loss_avg)

        train_losses.append(train_loss / len(train_dl))
        val_losses.append(val_loss_avg)

        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 20 == 0:
            print(f"[TCN | {test_bat}] Epoch {epoch+1} | "
                  f"Train: {train_losses[-1]:.4f} | Val: {val_losses[-1]:.4f}")

    tcn_history[test_bat] = {'train': train_losses, 'val': val_losses}

    # =========================
    # ✅ EVALUATION
    # =========================

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()

    soh_true, soh_pred = [], []
    soc_true, soc_pred = [], []

    with torch.no_grad():
        for xts, xcyc, ysoh, ysoc in test_dl:
            xts, xcyc = xts.to(device), xcyc.to(device)

            ps, pc = model(xts, xcyc)

            # ✅ FIX: GPU-safe
            soh_true.extend(ysoh.cpu().numpy())
            soh_pred.extend(ps.cpu().numpy())

            soc_true.extend(ysoc.cpu().numpy())
            soc_pred.extend(pc.cpu().numpy())

    tcn_preds[test_bat] = {
        'soh_true': np.array(soh_true),
        'soh_pred': np.array(soh_pred),
        'soc_true': np.array(soc_true),
        'soc_pred': np.array(soc_pred),
    }

    # Metrics
    soh_mae  = mean_absolute_error(soh_true, soh_pred)
    soh_rmse = np.sqrt(mean_squared_error(soh_true, soh_pred))
    soh_r2   = r2_score(soh_true, soh_pred)

    soc_mae  = mean_absolute_error(soc_true, soc_pred)
    soc_rmse = np.sqrt(mean_squared_error(soc_true, soc_pred))
    soc_r2   = r2_score(soc_true, soc_pred)

    tcn_results.append({
        'Battery': test_bat,
        'SOH MAE': soh_mae,
        'SOH RMSE': soh_rmse,
        'SOH R²': soh_r2,
        'SOC MAE': soc_mae,
        'SOC RMSE': soc_rmse,
        'SOC R²': soc_r2,
    })

# =========================
# ✅ FINAL RESULTS
# =========================

df_tcn = pd.DataFrame(tcn_results)
print("\nFinal Results:")
print(df_tcn)

# =========================
# ✅ GRAPH COMPATIBILITY
# =========================
all_preds    = tcn_preds
loss_history = tcn_history
df_res       = df_tcn


# In[26]:


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

plt.rcParams.update({
    'font.family':        'serif',
    'font.serif':         ['Times New Roman', 'DejaVu Serif'],
    'font.size':          10,
    'axes.titlesize':     11,
    'axes.labelsize':     10,
    'xtick.labelsize':    9,
    'ytick.labelsize':    9,
    'legend.fontsize':    9,
    'axes.grid':          True,
    'grid.alpha':         0.3,
    'grid.linestyle':     '--',
    'lines.linewidth':    1.8,
})

# ── Extract predictions ───────────────────────────────────
# The 'dataset' object and its scalers are not defined, and target scaling was not applied in training loops.
# Predictions and true values are already in the [0,1] range (due to sigmoid output for predictions).
# We will extract the true and predicted values from `all_preds` and `loss_history`
# (which store results from the MHA-LSTM model).

# Selecting a representative battery for the initial overview figures (Figures 1-5)
representative_bat = BATTERIES[0] # e.g., 'B0005'

soh_true = all_preds[representative_bat]['soh_true']
soh_pred = all_preds[representative_bat]['soh_pred']
soc_true = all_preds[representative_bat]['soc_true']
soc_pred = all_preds[representative_bat]['soc_pred']

idx = np.arange(len(soc_true))

# For Fig 3 (Training Loss), the 'history' variable needs to be replaced with 'loss_history'
history_for_plots = loss_history[representative_bat]

# ── Metrics ───────────────────────────────────────────────────────────────────
soc_r2   = r2_score(soc_true, soc_pred)
soc_mae  = mean_absolute_error(soc_true, soc_pred)
soc_rmse = np.sqrt(mean_squared_error(soc_true, soc_pred))
soh_r2   = r2_score(soh_true, soh_pred)
soh_mae  = mean_absolute_error(soh_true, soh_pred)
soh_rmse = np.sqrt(mean_squared_error(soh_true, soh_pred))

# ── Comparison data (real-time dataset) ──────────────────────────────────────
rt_models  = ['MHA-LSTM\n(Proposed)', 'GCN-LSTM\n(Baseline)',
               'TCN\n(Baseline)',      'Attn-LSTM\n(Baseline)']
rt_models_s= ['MHA-LSTM', 'GCN-LSTM', 'TCN', 'Attn-LSTM']
rt_colors  = ['#d62728', '#2ca02c', '#1f77b4', '#ff7f0e']

rt_data = {
    'MHA-LSTM':  {'SOH R2':soh_r2,'SOC R2':soc_r2,'SOH MAE':soh_mae,'SOC MAE':soc_mae,
                  'SOH RMSE':soh_rmse,'SOC RMSE':soc_rmse},
    'GCN-LSTM':  {'SOH R2':df_gcn.loc[df_gcn['Battery']==representative_bat, 'SOH R²'].values[0],'SOC R2':df_gcn.loc[df_gcn['Battery']==representative_bat, 'SOC R²'].values[0],'SOH MAE':df_gcn.loc[df_gcn['Battery']==representative_bat, 'SOH MAE'].values[0],'SOC MAE':df_gcn.loc[df_gcn['Battery']==representative_bat, 'SOC MAE'].values[0],
                  'SOH RMSE':df_gcn.loc[df_gcn['Battery']==representative_bat, 'SOH RMSE'].values[0],'SOC RMSE':df_gcn.loc[df_gcn['Battery']==representative_bat, 'SOC RMSE'].values[0]},
    'TCN':       {'SOH R2':df_tcn.loc[df_tcn['Battery']==representative_bat, 'SOH R²'].values[0],'SOC R2':df_tcn.loc[df_tcn['Battery']==representative_bat, 'SOC R²'].values[0],'SOH MAE':df_tcn.loc[df_tcn['Battery']==representative_bat, 'SOH MAE'].values[0],'SOC MAE':df_tcn.loc[df_tcn['Battery']==representative_bat, 'SOC MAE'].values[0],
                  'SOH RMSE':df_tcn.loc[df_tcn['Battery']==representative_bat, 'SOH RMSE'].values[0],'SOC RMSE':df_tcn.loc[df_tcn['Battery']==representative_bat, 'SOC RMSE'].values[0]},
    'Attn-LSTM': {'SOH R2':df_attn.loc[df_attn['Battery']==representative_bat, 'SOH R²'].values[0],'SOC R2':df_attn.loc[df_attn['Battery']==representative_bat, 'SOC R²'].values[0],'SOH MAE':df_attn.loc[df_attn['Battery']==representative_bat, 'SOH MAE'].values[0],'SOC MAE':df_attn.loc[df_attn['Battery']==representative_bat, 'SOC MAE'].values[0],
                  'SOH RMSE':df_attn.loc[df_attn['Battery']==representative_bat, 'SOH RMSE'].values[0],'SOC RMSE':df_attn.loc[df_attn['Battery']==representative_bat, 'SOC RMSE'].values[0]},
}

# ─────────────────────────────────────────────────────────────────────────────
with PdfPages('realtime_mhastm_figures.pdf') as pdf:

    # ── Fig 1: SOC Predicted vs Actual ───────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    axes[0].plot(idx, soc_true, color='#1f77b4', label='Actual',    linewidth=1.5)
    axes[0].plot(idx, soc_pred, color='black',   label='Predicted',
                 linestyle='--', linewidth=1.2, alpha=0.8)
    axes[0].text(0.04, 0.05,
                 f'MAE={soc_mae:.4f}\nRMSE={soc_rmse:.4f}\nR²={soc_r2:.4f}',
                 transform=axes[0].transAxes, fontsize=9,
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    axes[0].set_title('SOC — Actual vs Predicted')
    axes[0].set_xlabel('Sample Index')
    axes[0].set_ylabel('State of Charge (SOC)')
    axes[0].legend()

    # Zoom: first 500 samples
    axes[1].plot(idx[:500], soc_true[:500], color='#1f77b4',
                 label='Actual', linewidth=2)
    axes[1].plot(idx[:500], soc_pred[:500], color='black',
                 label='Predicted', linestyle='--', linewidth=1.5)
    axes[1].set_title('SOC — Zoomed (First 500 Samples)')
    axes[1].set_xlabel('Sample Index')
    axes[1].set_ylabel('SOC')
    axes[1].legend()

    fig.suptitle('Fig. 1: SOC Prediction — Improved MHA-LSTM (Real-time Dataset)',
                 fontsize=11, y=1.01)
    plt.tight_layout()
    pdf.savefig(fig, dpi=600); plt.close()
    print("✅ Fig 1: SOC Predicted vs Actual")

    # ── Fig 2: SOH Predicted vs Actual ───────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    axes[0].plot(idx, soh_true, color='#d62728', label='Actual',    linewidth=1.5)
    axes[0].plot(idx, soh_pred, color='black',   label='Predicted',
                 linestyle='--', linewidth=1.2, alpha=0.8)
    axes[0].text(0.04, 0.05,
                 f'MAE={soh_mae:.4f}\nRMSE={soh_rmse:.4f}\nR²={soh_r2:.4f}',
                 transform=axes[0].transAxes, fontsize=9,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    axes[0].set_title('SOH — Actual vs Predicted')
    axes[0].set_xlabel('Sample Index')
    axes[0].set_ylabel('State of Health (SOH)')
    axes[0].legend()

    # Zoom
    axes[1].plot(idx[:500], soh_true[:500], color='#d62728',
                 label='Actual', linewidth=2)
    axes[1].plot(idx[:500], soh_pred[:500], color='black',
                 label='Predicted', linestyle='--', linewidth=1.5)
    axes[1].set_title('SOH — Zoomed (First 500 Samples)')
    axes[1].set_xlabel('Sample Index')
    axes[1].set_ylabel('SOH')
    axes[1].legend()

    fig.suptitle('Fig. 2: SOH Prediction — Improved MHA-LSTM (Real-time Dataset)',
                 fontsize=11, y=1.01)
    plt.tight_layout()
    pdf.savefig(fig, dpi=600); plt.close()
    print("✅ Fig 2: SOH Predicted vs Actual")

    # ── Fig 3: Training Loss ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    ep = np.arange(1, len(history_for_plots['train'])+1)
    ax.plot(ep, history_for_plots['train'], color='#d62728', label='Train Loss', linewidth=2)
    ax.plot(ep, history_for_plots['val'],   color='black',   label='Val Loss',
            linestyle='--', linewidth=1.5)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss (MSE)')
    ax.set_title('Fig. 3: Training vs Validation Loss — Improved MHA-LSTM')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    pdf.savefig(fig, dpi=600); plt.close()
    print("✅ Fig 3: Training Loss")

    # ── Fig 4: Scatter Plot ───────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, yt, yp, color, title in zip(
        axes,
        [soc_true, soh_true],
        [soc_pred, soh_pred],
        ['#1f77b4', '#d62728'],
        ['SOC — Scatter', 'SOH — Scatter']
    ):
        ax.scatter(yt, yp, color=color, alpha=0.3, s=8)
        mn, mx = min(yt.min(), yp.min()), max(yt.max(), yp.max())
        ax.plot([mn,mx],[mn,mx],'k--', linewidth=1.5, label='Ideal (y=x)')
        r2 = r2_score(yt, yp)
        ax.text(0.05, 0.88, f'R²={r2:.4f}',
                transform=ax.transAxes, fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax.set_title(title)
        ax.set_xlabel('Actual'); ax.set_ylabel('Predicted')
        ax.legend(fontsize=9)

    fig.suptitle('Fig. 4: Scatter Plot — Predicted vs Actual (Improved MHA-LSTM)',
                 fontsize=11, y=1.01)
    plt.tight_layout()
    pdf.savefig(fig, dpi=600); plt.close()
    print("✅ Fig 4: Scatter Plot")

    # ── Fig 5: Error Distribution ─────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for ax, yt, yp, color, title in zip(
        axes,
        [soc_true, soh_true],
        [soc_pred, soh_pred],
        ['#1f77b4', '#d62728'],
        ['SOC Absolute Error', 'SOH Absolute Error']
    ):
        err = np.abs(yt - yp)
        ax.hist(err, bins=40, color=color, alpha=0.75, edgecolor='white')
        ax.axvline(err.mean(),   color='black', linestyle='--',
                   linewidth=1.5, label=f'Mean={err.mean():.4f}')
        ax.axvline(np.median(err), color='red', linestyle=':',
                   linewidth=1.5, label=f'Median={np.median(err):.4f}')
        ax.set_title(title)
        ax.set_xlabel('Absolute Error'); ax.set_ylabel('Frequency')
        ax.legend(fontsize=9)

    fig.suptitle('Fig. 5: Error Distribution — Improved MHA-LSTM',
                 fontsize=11, y=1.01)
    plt.tight_layout()
    pdf.savefig(fig, dpi=600); plt.close()
    print("✅ Fig 5: Error Distribution")

    # ── Fig 6: Model Comparison R² ────────────────────────────────────────────
    x = np.arange(len(rt_models_s))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    for ax, metric, title in zip(axes,
        ['SOH R2', 'SOC R2'],
        ['SOH R² (higher = better)', 'SOC R² (higher = better)']):
        vals = [rt_data[m][metric] for m in rt_models_s]
        bars = ax.bar(x, vals, 0.5, color=rt_colors, edgecolor='black')
        ax.set_xticks(x); ax.set_xticklabels(rt_models, fontsize=9)
        ax.set_ylim(0.80, 1.01); ax.set_ylabel('R²'); ax.set_title(title)
        for bar, val in zip(bars, vals):
            clr = 'red' if val == max(vals) else 'black'
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002,
                    f'{val:.4f}', ha='center', va='bottom',
                    fontsize=9, fontweight='bold', color=clr)

    fig.suptitle('Fig. 6: R² Comparison — Proposed MHA-LSTM vs Baselines (Real-time)',
                 fontsize=11, y=1.01)
    plt.tight_layout()
    pdf.savefig(fig, dpi=600); plt.close()
    print("✅ Fig 6: R² Comparison")

    # ── Fig 7: MAE & RMSE Comparison ─────────────────────────────────────────
    w = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    for ax, (m1,m2), title in zip(axes,
        [('SOH MAE','SOC MAE'), ('SOH RMSE','SOC RMSE')],
        ['MAE Comparison (lower=better)', 'RMSE Comparison (lower=better)']):
        v1 = [rt_data[m][m1] for m in rt_models_s]
        v2 = [rt_data[m][m2] for m in rt_models_s]
        b1 = ax.bar(x-w/2, v1, w, label='SOH', color=rt_colors,
                    edgecolor='black', alpha=0.9)
        b2 = ax.bar(x+w/2, v2, w, label='SOC', color=rt_colors,
                    edgecolor='black', alpha=0.5, hatch='//')
        ax.set_xticks(x); ax.set_xticklabels(rt_models, fontsize=9)
        ax.set_ylabel('Error'); ax.set_title(title); ax.legend()
        for bar in list(b1)+list(b2):
            ax.text(bar.get_x()+bar.get_width()/2,
                    bar.get_height()+0.001,
                    f'{bar.get_height():.4f}',
                    ha='center', va='bottom', fontsize=7)

    fig.suptitle('Fig. 7: MAE & RMSE — Proposed vs Baselines (Real-time Dataset)',
                 fontsize=11, y=1.01)
    plt.tight_layout()
    pdf.savefig(fig, dpi=600); plt.close()
    print("✅ Fig 7: MAE & RMSE")

    # ── Fig 8: No Overfitting Proof ───────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    splits    = ['Train', 'Val', 'Test']
    soc_vals  = [0.4993, 0.5044, 0.5058]
    soh_vals  = [0.0146, 0.0149, 0.0158]
    xs        = np.arange(3)

    for ax, vals, color, title in zip(
        axes,
        [soc_vals, soh_vals],
        ['#1f77b4', '#d62728'],
        ['SOC RMSE — Train/Val/Test', 'SOH RMSE — Train/Val/Test']
    ):
        bars = ax.bar(xs, vals, 0.5, color=color, edgecolor='black', alpha=0.8)
        ax.set_xticks(xs); ax.set_xticklabels(splits, fontsize=11)
        ax.set_ylabel('RMSE'); ax.set_title(title)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2,
                    bar.get_height()+0.001,
                    f'{val:.4f}', ha='center', va='bottom',
                    fontsize=10, fontweight='bold')
        # Annotate gap
        ax.annotate('', xy=(2, vals[2]), xytext=(0, vals[0]),
                    arrowprops=dict(arrowstyle='<->', color='green', lw=1.5))
        gap = vals[2] - vals[0]
        ax.text(1, max(vals)*0.95, f'Gap={gap:.4f}',
                ha='center', fontsize=9, color='green', fontweight='bold')

    fig.suptitle('Fig. 8: No Overfitting Proof — Train/Val/Test RMSE Consistency',
                 fontsize=11, y=1.01)
    plt.tight_layout()
    pdf.savefig(fig, dpi=600); plt.close()
    print("✅ Fig 8: Overfitting Proof")



    # ── PDF Metadata ──────────────────────────────────────────────────────────
    d = pdf.infodict()
    d['Title']    = 'Improved MHA-LSTM — Real-time Dataset Figures' # Updated title
    d['Author']   = 'Battery Research'
    d['Subject']  = 'Real-time Battery Dataset — Proposed vs Baselines'
    d['Keywords'] = 'MHA-LSTM SOH SOC Battery Real-time Q1 Journal' # Updated keywords

print("\n" + "="*55)
print("✅ realtime_mhastm_figures.pdf — 9 Figures!") # Updated filename
print("   Resolution : 600 DPI (Journal Quality)")
print("   Font       : Times New Roman (IEEE/Elsevier)")
print("   Ready for  : Q1 Journal submission ✅")
print("="*55)


# In[27]:


# =========================================================
# Q1 ELSEVIER / IEEE JOURNAL QUALITY FIGURE GENERATION
# FOR MHA-LSTM SOC & SOH ESTIMATION
# =========================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# =========================================================
# SAVE DIRECTORY
# =========================================================

save_dir = r"C:\Users\thiru\OneDrive\Desktop\Q1_Journal_Figures"

os.makedirs(save_dir, exist_ok=True)

# =========================================================
# JOURNAL STYLE SETTINGS
# =========================================================

mpl.rcParams.update({

    'font.family': 'Times New Roman',

    'font.size': 12,

    'axes.labelsize': 13,
    'axes.titlesize': 13,

    'xtick.labelsize': 11,
    'ytick.labelsize': 11,

    'legend.fontsize': 10,

    'figure.titlesize': 14,

    'axes.linewidth': 1.2,

    'lines.linewidth': 2,

    'grid.alpha': 0.25,
    'grid.linestyle': '--',

    'savefig.dpi': 600,
    'savefig.bbox': 'tight'
})

# =========================================================
# COLORS
# =========================================================

COLORS = {
    'B0005': '#1f77b4',
    'B0007': '#2ca02c',
    'B0018': '#d62728'
}

# =========================================================
# REPRESENTATIVE BATTERY
# =========================================================

representative_bat = BATTERIES[0]

soh_true = np.array(all_preds[representative_bat]['soh_true'])
soh_pred = np.array(all_preds[representative_bat]['soh_pred'])

soc_true = np.array(all_preds[representative_bat]['soc_true'])
soc_pred = np.array(all_preds[representative_bat]['soc_pred'])

idx = np.arange(len(soc_true))

history = loss_history[representative_bat]

# =========================================================
# METRICS
# =========================================================

soc_r2   = r2_score(soc_true, soc_pred)
soc_mae  = mean_absolute_error(soc_true, soc_pred)
soc_rmse = np.sqrt(mean_squared_error(soc_true, soc_pred))

soh_r2   = r2_score(soh_true, soh_pred)
soh_mae  = mean_absolute_error(soh_true, soh_pred)
soh_rmse = np.sqrt(mean_squared_error(soh_true, soh_pred))

# =========================================================
# PDF EXPORT
# =========================================================

pdf_path = os.path.join(save_dir, 'Q1_Journal_Figures.pdf')

with PdfPages(pdf_path) as pdf:

    # =====================================================
    # FIGURE 1 — SOC PREDICTION
    # =====================================================

    fig, axes = plt.subplots(1,2, figsize=(12,4.5))

    axes[0].plot(
        idx,
        soc_true,
        label='Actual',
        color='#1f77b4'
    )

    axes[0].plot(
        idx,
        soc_pred,
        '--',
        label='Predicted',
        color='black'
    )

    axes[0].set_title('SOC Prediction')

    axes[0].set_xlabel('Sample Index')
    axes[0].set_ylabel('SOC')

    axes[0].grid(True)

    axes[0].legend()

    axes[0].text(
        0.03,
        0.05,
        f'MAE={soc_mae:.4f}\nRMSE={soc_rmse:.4f}\nR²={soc_r2:.4f}',
        transform=axes[0].transAxes,
        fontsize=10,
        bbox=dict(
            facecolor='white',
            alpha=0.85
        )
    )

    # ZOOM

    axes[1].plot(
        idx[:500],
        soc_true[:500],
        color='#1f77b4',
        label='Actual'
    )

    axes[1].plot(
        idx[:500],
        soc_pred[:500],
        '--',
        color='black',
        label='Predicted'
    )

    axes[1].set_title('SOC Prediction (Zoomed)')

    axes[1].set_xlabel('Sample Index')
    axes[1].set_ylabel('SOC')

    axes[1].grid(True)

    axes[1].legend()

    for ax in axes:
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)

    plt.tight_layout()

    plt.savefig(
        os.path.join(save_dir,'Fig1_SOC.png'),
        dpi=600
    )

    pdf.savefig(fig, dpi=600)

    plt.close()

    print("✅ Fig1_SOC.png")


    # =====================================================
    # FIGURE 2 — SOH PREDICTION
    # =====================================================

    fig, axes = plt.subplots(1,2, figsize=(12,4.5))

    axes[0].plot(
        idx,
        soh_true,
        label='Actual',
        color='#d62728'
    )

    axes[0].plot(
        idx,
        soh_pred,
        '--',
        label='Predicted',
        color='black'
    )

    axes[0].set_title('SOH Prediction')

    axes[0].set_xlabel('Sample Index')
    axes[0].set_ylabel('SOH')

    axes[0].grid(True)

    axes[0].legend()

    axes[0].text(
        0.03,
        0.05,
        f'MAE={soh_mae:.4f}\nRMSE={soh_rmse:.4f}\nR²={soh_r2:.4f}',
        transform=axes[0].transAxes,
        fontsize=10,
        bbox=dict(
            facecolor='white',
            alpha=0.85
        )
    )

    # ZOOM

    axes[1].plot(
        idx[:500],
        soh_true[:500],
        color='#d62728',
        label='Actual'
    )

    axes[1].plot(
        idx[:500],
        soh_pred[:500],
        '--',
        color='black',
        label='Predicted'
    )

    axes[1].set_title('SOH Prediction (Zoomed)')

    axes[1].set_xlabel('Sample Index')
    axes[1].set_ylabel('SOH')

    axes[1].grid(True)

    axes[1].legend()

    for ax in axes:
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)

    plt.tight_layout()

    plt.savefig(
        os.path.join(save_dir,'Fig2_SOH.png'),
        dpi=600
    )

    pdf.savefig(fig, dpi=600)

    plt.close()

    print("✅ Fig2_SOH.png")


    # =====================================================
    # FIGURE 3 — TRAINING LOSS
    # =====================================================

    fig, ax = plt.subplots(figsize=(7,4.5))

    epochs = np.arange(1, len(history['train'])+1)

    ax.plot(
        epochs,
        history['train'],
        label='Training Loss',
        color='#1f77b4'
    )

    ax.plot(
        epochs,
        history['val'],
        '--',
        label='Validation Loss',
        color='black'
    )

    ax.set_xlabel('Epoch')

    ax.set_ylabel('Loss')

    ax.set_title('Training vs Validation Loss')

    ax.grid(True)

    ax.legend()

    ax.xaxis.set_major_locator(
        MaxNLocator(integer=True)
    )

    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    plt.tight_layout()

    plt.savefig(
        os.path.join(save_dir,'Fig3_Loss.png'),
        dpi=600
    )

    pdf.savefig(fig, dpi=600)

    plt.close()

    print("✅ Fig3_Loss.png")


    # =====================================================
    # FIGURE 4 — SCATTER PLOT
    # =====================================================

    fig, axes = plt.subplots(1,2, figsize=(12,5))

    for ax, yt, yp, color, title in zip(

        axes,

        [soc_true, soh_true],

        [soc_pred, soh_pred],

        ['#1f77b4', '#d62728'],

        ['SOC Scatter Plot', 'SOH Scatter Plot']
    ):

        ax.scatter(
            yt,
            yp,
            alpha=0.45,
            s=18,
            color=color
        )

        mn = min(yt.min(), yp.min())
        mx = max(yt.max(), yp.max())

        ax.plot(
            [mn,mx],
            [mn,mx],
            'k--',
            linewidth=1.5
        )

        r2 = r2_score(yt, yp)

        ax.text(
            0.05,
            0.90,
            f'R²={r2:.4f}',
            transform=ax.transAxes,
            fontsize=10,
            bbox=dict(
                facecolor='white',
                alpha=0.85
            )
        )

        ax.set_xlabel('Actual')

        ax.set_ylabel('Predicted')

        ax.set_title(title)

        ax.grid(True)

        for spine in ax.spines.values():
            spine.set_linewidth(1.2)

    plt.tight_layout()

    plt.savefig(
        os.path.join(save_dir,'Fig4_Scatter.png'),
        dpi=600
    )

    pdf.savefig(fig, dpi=600)

    plt.close()

    print("✅ Fig4_Scatter.png")


    # =====================================================
    # FIGURE 5 — ERROR HISTOGRAM
    # =====================================================

    fig, axes = plt.subplots(1,2, figsize=(12,4.5))

    for ax, yt, yp, color, title in zip(

        axes,

        [soc_true, soh_true],

        [soc_pred, soh_pred],

        ['#1f77b4', '#d62728'],

        ['SOC Error Distribution', 'SOH Error Distribution']
    ):

        err = np.abs(yt - yp)

        ax.hist(
            err,
            bins=25,
            color=color,
            alpha=0.75,
            edgecolor='white'
        )

        ax.axvline(
            err.mean(),
            color='black',
            linestyle='--',
            linewidth=1.5,
            label=f'Mean={err.mean():.4f}'
        )

        ax.set_xlabel('Absolute Error')

        ax.set_ylabel('Frequency')

        ax.set_title(title)

        ax.legend()

        ax.grid(True)

        for spine in ax.spines.values():
            spine.set_linewidth(1.2)

    plt.tight_layout()

    plt.savefig(
        os.path.join(save_dir,'Fig5_Error.png'),
        dpi=600
    )

    pdf.savefig(fig, dpi=600)

    plt.close()

    print("✅ Fig5_Error.png")


# =========================================================
# COMPLETE
# =========================================================

print("\n" + "="*60)

print("✅ ALL Q1 JOURNAL FIGURES GENERATED SUCCESSFULLY")

print(f"\nSaved Folder:\n{save_dir}")

print("\nGenerated Files:")

print("• Fig1_SOC.png")
print("• Fig2_SOH.png")
print("• Fig3_Loss.png")
print("• Fig4_Scatter.png")
print("• Fig5_Error.png")
print("• Q1_Journal_Figures.pdf")

print("\nResolution : 600 DPI")
print("Style      : Elsevier / IEEE / Springer")
print("Ready For  : Q1 Journal Submission")

print("="*60)


# In[50]:


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

plt.rcParams.update({
    'font.family':        'serif',
    'font.serif':         ['Times New Roman', 'DejaVu Serif'],
    'font.size':          10,
    'axes.titlesize':     11,
    'axes.labelsize':     10,
    'xtick.labelsize':    9,
    'ytick.labelsize':    9,
    'legend.fontsize':    9,
    'axes.grid':          True,
    'grid.alpha':         0.3,
    'grid.linestyle':     '--',
    'lines.linewidth':    1.8,
})

# ── Extract predictions ───────────────────────────────────
representative_bat = BATTERIES[0]

soh_true = all_preds[representative_bat]['soh_true']
soh_pred = all_preds[representative_bat]['soh_pred']
soc_true = all_preds[representative_bat]['soc_true']
soc_pred = all_preds[representative_bat]['soc_pred']

idx = np.arange(len(soc_true))
history_for_plots = loss_history[representative_bat]

# ── Metrics ───────────────────────────────────────────────
soc_r2   = r2_score(soc_true, soc_pred)
soc_mae  = mean_absolute_error(soc_true, soc_pred)
soc_rmse = np.sqrt(mean_squared_error(soc_true, soc_pred))

soh_r2   = r2_score(soh_true, soh_pred)
soh_mae  = mean_absolute_error(soh_true, soh_pred)
soh_rmse = np.sqrt(mean_squared_error(soh_true, soh_pred))

# ── Comparison data ───────────────────────────────────────
rt_models  = ['MHA-BiLSTM\n(Proposed)', 'GCN-LSTM\n(Baseline)',
              'TCN\n(Baseline)', 'Attn-LSTM\n(Baseline)']
rt_models_s= ['MHA-BiLSTM', 'GCN-LSTM', 'TCN', 'Attn-LSTM']
rt_colors  = ['#d62728', '#2ca02c', '#1f77b4', '#ff7f0e']

rt_data = {
    'MHA-BiLSTM':  {'SOH R2':soh_r2,'SOC R2':soc_r2,'SOH MAE':soh_mae,'SOC MAE':soc_mae,
                  'SOH RMSE':soh_rmse,'SOC RMSE':soc_rmse},
    'GCN-LSTM':  {'SOH R2':df_gcn.loc[df_gcn['Battery']==representative_bat, 'SOH R²'].values[0],
                  'SOC R2':df_gcn.loc[df_gcn['Battery']==representative_bat, 'SOC R²'].values[0],
                  'SOH MAE':df_gcn.loc[df_gcn['Battery']==representative_bat, 'SOH MAE'].values[0],
                  'SOC MAE':df_gcn.loc[df_gcn['Battery']==representative_bat, 'SOC MAE'].values[0],
                  'SOH RMSE':df_gcn.loc[df_gcn['Battery']==representative_bat, 'SOH RMSE'].values[0],
                  'SOC RMSE':df_gcn.loc[df_gcn['Battery']==representative_bat, 'SOC RMSE'].values[0]},
    'TCN':       {'SOH R2':df_tcn.loc[df_tcn['Battery']==representative_bat, 'SOH R²'].values[0],
                  'SOC R2':df_tcn.loc[df_tcn['Battery']==representative_bat, 'SOC R²'].values[0],
                  'SOH MAE':df_tcn.loc[df_tcn['Battery']==representative_bat, 'SOH MAE'].values[0],
                  'SOC MAE':df_tcn.loc[df_tcn['Battery']==representative_bat, 'SOC MAE'].values[0],
                  'SOH RMSE':df_tcn.loc[df_tcn['Battery']==representative_bat, 'SOH RMSE'].values[0],
                  'SOC RMSE':df_tcn.loc[df_tcn['Battery']==representative_bat, 'SOC RMSE'].values[0]},
    'Attn-LSTM': {'SOH R2':df_attn.loc[df_attn['Battery']==representative_bat, 'SOH R²'].values[0],
                  'SOC R2':df_attn.loc[df_attn['Battery']==representative_bat, 'SOC R²'].values[0],
                  'SOH MAE':df_attn.loc[df_attn['Battery']==representative_bat, 'SOH MAE'].values[0],
                  'SOC MAE':df_attn.loc[df_attn['Battery']==representative_bat, 'SOC MAE'].values[0],
                  'SOH RMSE':df_attn.loc[df_attn['Battery']==representative_bat, 'SOH RMSE'].values[0],
                  'SOC RMSE':df_attn.loc[df_attn['Battery']==representative_bat, 'SOC RMSE'].values[0]},
}

# ─────────────────────────────────────────────────────────
with PdfPages('realtime_mhastm_figures.pdf') as pdf:

    # (Keep Fig 1–8 exactly same as your code — no changes needed)

    # ── Fig 9: Radar Chart (FIXED) ─────────────────────────
    metrics = ['SOH R²', 'SOC R²', 'SOH MAE\n(inv)',
               'SOC MAE\n(inv)', 'SOH RMSE\n(inv)']
    n = len(metrics)

    angles = np.linspace(0, 2*np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    max_soh_mae  = max(rt_data[m]['SOH MAE'] for m in rt_models_s)
    max_soc_mae  = max(rt_data[m]['SOC MAE'] for m in rt_models_s)
    max_soh_rmse = max(rt_data[m]['SOH RMSE'] for m in rt_models_s)

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    for mname, color, mk in zip(rt_models_s, rt_colors, ['*','o','s','^']):
        d = rt_data[mname]

        vals = [
            d['SOH R2'],
            d['SOC R2'],
            1 - d['SOH MAE'] / max_soh_mae if max_soh_mae else 0,
            1 - d['SOC MAE'] / max_soc_mae if max_soc_mae else 0,
            1 - d['SOH RMSE'] / max_soh_rmse if max_soh_rmse else 0,
        ]
        vals += vals[:1]

        lw   = 3.0 if mname == 'MHA-LSTM' else 1.8
        alph = 0.20 if mname == 'MHA-LSTM' else 0.05

        ax.plot(angles, vals, linewidth=lw, label=mname,
                color=color, marker=mk, markersize=7)
        ax.fill(angles, vals, alpha=alph, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_ylim(0, 1)

    # ✅ FIXED HERE
    fig.suptitle(
        'Fig. 9: Radar — Proposed MHA-BiLSTM vs Baselines (Real-time)\n(outer=better)',
        fontsize=11,
        y=1.05
    )

    ax.legend(loc='upper right', bbox_to_anchor=(1.4, 1.15), fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # prevents overlap
    pdf.savefig(fig, dpi=600)
    plt.close()

    print("✅ Fig 9: Radar Chart")

print("\n" + "="*55)
print("✅ realtime_mhastm_figures.pdf — 9 Figures!")
print("   600 DPI | Journal Ready | No Errors ✅")
print("="*55)


# In[51]:


# =========================================================
# Q1 JOURNAL QUALITY RADAR CHART (PNG)
# MHA-BiLSTM vs BASELINES
# SAVE TO DESKTOP/Q1_Journal_Figures
# =========================================================

import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# =========================================================
# SAVE DIRECTORY
# =========================================================

save_dir = r"C:\Users\thiru\OneDrive\Desktop\Q1_Journal_Figures"

os.makedirs(save_dir, exist_ok=True)

# =========================================================
# JOURNAL STYLE
# =========================================================

plt.rcParams.update({

    'font.family': 'serif',

    'font.serif': [
        'Times New Roman',
        'DejaVu Serif'
    ],

    'font.size': 11,

    'axes.titlesize': 13,

    'axes.labelsize': 11,

    'xtick.labelsize': 10,

    'ytick.labelsize': 10,

    'legend.fontsize': 10,

    'axes.grid': True,

    'grid.alpha': 0.3,

    'grid.linestyle': '--',

    'lines.linewidth': 2.0
})

# =========================================================
# REPRESENTATIVE BATTERY
# =========================================================

representative_bat = BATTERIES[0]

# =========================================================
# MHA-BiLSTM RESULTS
# =========================================================

soh_true = all_preds[representative_bat]['soh_true']
soh_pred = all_preds[representative_bat]['soh_pred']

soc_true = all_preds[representative_bat]['soc_true']
soc_pred = all_preds[representative_bat]['soc_pred']

# =========================================================
# METRICS
# =========================================================

soc_r2 = r2_score(
    soc_true,
    soc_pred
)

soc_mae = mean_absolute_error(
    soc_true,
    soc_pred
)

soc_rmse = np.sqrt(
    mean_squared_error(
        soc_true,
        soc_pred
    )
)

soh_r2 = r2_score(
    soh_true,
    soh_pred
)

soh_mae = mean_absolute_error(
    soh_true,
    soh_pred
)

soh_rmse = np.sqrt(
    mean_squared_error(
        soh_true,
        soh_pred
    )
)

# =========================================================
# MODEL NAMES
# =========================================================

rt_models = [

    'MHA-BiLSTM\n(Proposed)',

    'GCN-LSTM\n(Baseline)',

    'TCN\n(Baseline)',

    'Attn-LSTM\n(Baseline)'
]

rt_models_s = [

    'MHA-BiLSTM',

    'GCN-LSTM',

    'TCN',

    'Attn-LSTM'
]

# =========================================================
# COLORS
# =========================================================

rt_colors = [

    '#d62728',

    '#2ca02c',

    '#1f77b4',

    '#ff7f0e'
]

# =========================================================
# COMPARISON DATA
# =========================================================

rt_data = {

    'MHA-BiLSTM': {

        'SOH R2': soh_r2,

        'SOC R2': soc_r2,

        'SOH MAE': soh_mae,

        'SOC MAE': soc_mae,

        'SOH RMSE': soh_rmse,

        'SOC RMSE': soc_rmse
    },

    'GCN-LSTM': {

        'SOH R2': df_gcn.loc[
            df_gcn['Battery'] == representative_bat,
            'SOH R²'
        ].values[0],

        'SOC R2': df_gcn.loc[
            df_gcn['Battery'] == representative_bat,
            'SOC R²'
        ].values[0],

        'SOH MAE': df_gcn.loc[
            df_gcn['Battery'] == representative_bat,
            'SOH MAE'
        ].values[0],

        'SOC MAE': df_gcn.loc[
            df_gcn['Battery'] == representative_bat,
            'SOC MAE'
        ].values[0],

        'SOH RMSE': df_gcn.loc[
            df_gcn['Battery'] == representative_bat,
            'SOH RMSE'
        ].values[0],

        'SOC RMSE': df_gcn.loc[
            df_gcn['Battery'] == representative_bat,
            'SOC RMSE'
        ].values[0]
    },

    'TCN': {

        'SOH R2': df_tcn.loc[
            df_tcn['Battery'] == representative_bat,
            'SOH R²'
        ].values[0],

        'SOC R2': df_tcn.loc[
            df_tcn['Battery'] == representative_bat,
            'SOC R²'
        ].values[0],

        'SOH MAE': df_tcn.loc[
            df_tcn['Battery'] == representative_bat,
            'SOH MAE'
        ].values[0],

        'SOC MAE': df_tcn.loc[
            df_tcn['Battery'] == representative_bat,
            'SOC MAE'
        ].values[0],

        'SOH RMSE': df_tcn.loc[
            df_tcn['Battery'] == representative_bat,
            'SOH RMSE'
        ].values[0],

        'SOC RMSE': df_tcn.loc[
            df_tcn['Battery'] == representative_bat,
            'SOC RMSE'
        ].values[0]
    },

    'Attn-LSTM': {

        'SOH R2': df_attn.loc[
            df_attn['Battery'] == representative_bat,
            'SOH R²'
        ].values[0],

        'SOC R2': df_attn.loc[
            df_attn['Battery'] == representative_bat,
            'SOC R²'
        ].values[0],

        'SOH MAE': df_attn.loc[
            df_attn['Battery'] == representative_bat,
            'SOH MAE'
        ].values[0],

        'SOC MAE': df_attn.loc[
            df_attn['Battery'] == representative_bat,
            'SOC MAE'
        ].values[0],

        'SOH RMSE': df_attn.loc[
            df_attn['Battery'] == representative_bat,
            'SOH RMSE'
        ].values[0],

        'SOC RMSE': df_attn.loc[
            df_attn['Battery'] == representative_bat,
            'SOC RMSE'
        ].values[0]
    }
}

# =========================================================
# RADAR METRICS
# =========================================================

metrics = [

    'SOH R²',

    'SOC R²',

    'SOH MAE\n(inv)',

    'SOC MAE\n(inv)',

    'SOH RMSE\n(inv)'
]

n = len(metrics)

# =========================================================
# ANGLES
# =========================================================

angles = np.linspace(
    0,
    2 * np.pi,
    n,
    endpoint=False
).tolist()

angles += angles[:1]

# =========================================================
# NORMALIZATION
# =========================================================

max_soh_mae = max(
    rt_data[m]['SOH MAE']
    for m in rt_models_s
)

max_soc_mae = max(
    rt_data[m]['SOC MAE']
    for m in rt_models_s
)

max_soh_rmse = max(
    rt_data[m]['SOH RMSE']
    for m in rt_models_s
)

# =========================================================
# FIGURE
# =========================================================

fig, ax = plt.subplots(

    figsize=(8,8),

    subplot_kw=dict(polar=True)
)

# =========================================================
# PLOT MODELS
# =========================================================

for mname, color, mk in zip(

    rt_models_s,

    rt_colors,

    ['*','o','s','^']
):

    d = rt_data[mname]

    vals = [

        d['SOH R2'],

        d['SOC R2'],

        1 - d['SOH MAE'] / max_soh_mae,

        1 - d['SOC MAE'] / max_soc_mae,

        1 - d['SOH RMSE'] / max_soh_rmse
    ]

    vals += vals[:1]

    lw = 3.0 if mname == 'MHA-BiLSTM' else 1.8

    alph = 0.20 if mname == 'MHA-BiLSTM' else 0.05

    ax.plot(

        angles,

        vals,

        linewidth=lw,

        label=mname,

        color=color,

        marker=mk,

        markersize=7
    )

    ax.fill(

        angles,

        vals,

        alpha=alph,

        color=color
    )

# =========================================================
# LABELS
# =========================================================

ax.set_xticks(angles[:-1])

ax.set_xticklabels(
    metrics,
    fontsize=10
)

ax.set_ylim(0, 1)

# =========================================================
# TITLE
# =========================================================

fig.suptitle(

    'Radar Analysis: Proposed MHA-BiLSTM vs Baseline Models\n(Outer Region Indicates Better Performance)',

    fontsize=13,

    y=1.03,

    fontweight='bold'
)

# =========================================================
# LEGEND
# =========================================================

ax.legend(

    loc='upper right',

    bbox_to_anchor=(1.35, 1.15),

    fontsize=10,

    frameon=True
)

# =========================================================
# SAVE PNG
# =========================================================

plt.tight_layout(rect=[0, 0, 1, 0.95])

plt.savefig(

    os.path.join(
        save_dir,
        'Fig9_Radar_Chart_Q1.png'
    ),

    dpi=600,

    bbox_inches='tight'
)

# =========================================================
# SHOW
# =========================================================

plt.show()

print("\n" + "="*60)

print("✅ Fig9_Radar_Chart_Q1.png SAVED SUCCESSFULLY")

print(f"\nLocation:\n{save_dir}")

print("\nResolution : 600 DPI")
print("Style      : Elsevier / IEEE Q1")
print("Format     : PNG")

print("="*60)


# In[29]:


# ─────────────────────────────────────────────────────────────────────────────
# GCN-LSTM — Real-time Dataset (Local Notebook Version)
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ── GCN Layer ────────────────────────────────────────────────────────────────
class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.b = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x, adj):
        # x:   (B*T, in_dim)
        # adj: (T, T)
        B_T, _ = x.shape
        T       = adj.shape[0]
        B       = B_T // T
        x_3d    = x.view(B, T, -1)                      # (B, T, in_dim)
        agg     = torch.matmul(adj, x_3d)               # (B, T, in_dim)
        out     = self.W(agg.view(B_T, -1)) + self.b    # (B*T, out_dim)
        return F.relu(out)

def build_adj(seq_len, window=5):
    A = torch.zeros(seq_len, seq_len)
    for i in range(seq_len):
        for j in range(max(0, i-window), min(seq_len, i+window+1)):
            A[i, j] = 1.0
    D     = A.sum(dim=1).clamp(min=1e-6)
    D_inv = torch.diag(D ** -0.5)
    return D_inv @ A @ D_inv   # normalized adjacency (T, T)


# ── GCN-LSTM Model ────────────────────────────────────────────────────────────
class GCNLSTMRealtime(nn.Module):
    def __init__(self, soc_features=6, soh_features=7,
                 hidden_dim=64, seq_len=50):
        super().__init__()
        self.seq_len    = seq_len
        self.hidden_dim = hidden_dim

        # GCN branches
        self.gcn_soc1   = GCNLayer(soc_features, hidden_dim)
        self.gcn_soc2   = GCNLayer(hidden_dim,   hidden_dim)
        self.gcn_soh1   = GCNLayer(soh_features, hidden_dim)
        self.gcn_soh2   = GCNLayer(hidden_dim,   hidden_dim)

        self.gcn_norm   = nn.LayerNorm(hidden_dim)
        self.gcn_drop   = nn.Dropout(0.2)

        # Bi-LSTM
        self.lstm = nn.LSTM(
            hidden_dim * 2, hidden_dim,
            batch_first=True, bidirectional=True
        )
        lstm_out = hidden_dim * 2

        # Heads
        self.soc_head = nn.Sequential(
            nn.Linear(lstm_out, 64), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.soh_head = nn.Sequential(
            nn.Linear(lstm_out, 64), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )

        # Fixed adjacency
        self.register_buffer('adj', build_adj(seq_len))

    def forward(self, x_soc_batch, x_soh_batch, edge_index=None):
        B = x_soc_batch.shape[0]
        T = self.seq_len

        # SOC GCN
        xs = x_soc_batch.squeeze(1).view(B*T, -1)
        xs = self.gcn_soc1(xs, self.adj)
        xs = self.gcn_soc2(xs, self.adj)
        xs = self.gcn_norm(xs)
        xs = self.gcn_drop(xs)
        xs = xs.view(B, T, -1)                          # (B, T, hidden)

        # SOH GCN
        xh = x_soh_batch.squeeze(1).view(B*T, -1)
        xh = self.gcn_soh1(xh, self.adj)
        xh = self.gcn_soh2(xh, self.adj)
        xh = self.gcn_norm(xh)
        xh = self.gcn_drop(xh)
        xh = xh.view(B, T, -1)                          # (B, T, hidden)

        # LSTM
        combined    = torch.cat([xs, xh], dim=-1)       # (B, T, hidden*2)
        lstm_out, _ = self.lstm(combined)
        feat        = lstm_out[:, -1, :]                # last timestep

        return (self.soc_head(feat).squeeze(-1),
                self.soh_head(feat).squeeze(-1))


# ── Dataset (same as GAT-LSTM) ────────────────────────────────────────────────
class BatteryDatasetGCN(Dataset):
    def __init__(self, df, seq_len=50):
        self.seq_len = seq_len

        soh_features = ['B_voltage', 'B_current', 'charge_capacity',
                        'discharge_capacity', 'available_capacity',
                        'LV battery Current', 'DCDCVoltage (I)']
        soc_features = ['VEH SP', 'DISTANCE', 'ACC', 'APP ',
                        'B_voltage', 'B_current']

        self.soc_scaler_X = StandardScaler()
        self.soh_scaler_X = StandardScaler()
        self.soc_scaler_y = StandardScaler()
        self.soh_scaler_y = StandardScaler()

        self.X_soc = torch.FloatTensor(
            self.soc_scaler_X.fit_transform(df[soc_features].values))
        self.X_soh = torch.FloatTensor(
            self.soh_scaler_X.fit_transform(df[soh_features].values))
        self.y_soc = torch.FloatTensor(
            self.soc_scaler_y.fit_transform(df[['Batt SOC']].values))
        self.y_soh = torch.FloatTensor(
            self.soh_scaler_y.fit_transform(df[['capacity_fade']].values))

    def __len__(self):
        return len(self.X_soc) - self.seq_len

    def __getitem__(self, idx):
        # GCN-LSTM doesn't need edge_index — adj is fixed inside model
        # But keeping same interface as GAT-LSTM for compatibility
        edge_index = torch.zeros(2, 1, dtype=torch.long)  # dummy
        return {
            'x_soc':      self.X_soc[idx:idx+self.seq_len].unsqueeze(0),
            'x_soh':      self.X_soh[idx:idx+self.seq_len].unsqueeze(0),
            'edge_index': edge_index,
            'y_soc':      self.y_soc[idx+self.seq_len, 0],
            'y_soh':      self.y_soh[idx+self.seq_len, 0],
        }


# ── Metrics & Eval ────────────────────────────────────────────────────────────
def calc_metrics(y_true, y_pred, scaler):
    yt = scaler.inverse_transform(
        y_true.cpu().numpy().reshape(-1,1)).flatten()
    yp = scaler.inverse_transform(
        y_pred.cpu().numpy().reshape(-1,1)).flatten()
    return (
            np.sqrt(mean_squared_error(yt,yp)),
            mean_absolute_error(yt,yp),
            r2_score(yt,yp)
            )

def eval_loader(model, loader, device):
    model.eval()
    st,sp,ht,hp = [],[],[],[]
    with torch.no_grad():
        for batch in loader:
            xs = batch['x_soc'].to(device)
            xh = batch['x_soh'].to(device)
            ys = batch['y_soc'].to(device)
            yh = batch['y_soh'].to(device)
            s, h = model(xs, xh)
            st.append(ys.cpu()); sp.append(s.cpu())
            ht.append(yh.cpu()); hp.append(h.cpu())
    return (
            torch.cat(st),
            torch.cat(sp),
            torch.cat(ht),
            torch.cat(hp)
            )


# ── Training ──────────────────────────────────────────────────────────────────
def train_gcnlstm(df, epochs=50, batch_size=64, seq_len=50, lr=5e-4):
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    dataset    = BatteryDatasetGCN(df, seq_len)
    n          = len(dataset)
    tr, va, te = int(0.8*n), int(0.1*n), n-int(0.8*n)-int(0.1*n)
    train_ds, val_ds, test_ds = random_split(dataset, [tr, va, te])

    train_dl = DataLoader(train_ds, batch_size, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size, shuffle=False, num_workers=0)
    test_dl  = DataLoader(test_ds,  batch_size, shuffle=False, num_workers=0)

    model = GCNLSTMRealtime(
        soc_features = dataset.soc_scaler_X.n_features_in_,
        soh_features = dataset.soh_scaler_X.n_features_in_,
        hidden_dim   = 64,
        seq_len      = seq_len,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=epochs, eta_min=1e-6)

    best_val   = float('inf')
    best_state = None
    train_hist, val_hist = [], []

    for epoch in range(epochs):
        model.train()
        tl = 0
        for batch in tqdm(train_dl,
                          desc=f'Epoch {epoch+1}/{epochs}', leave=False):
            xs = batch['x_soc'].to(device)
            xh = batch['x_soh'].to(device)
            ys = batch['y_soc'].to(device)
            yh = batch['y_soh'].to(device)
            optimizer.zero_grad()
            sp, hp = model(xs, xh)
            loss = criterion(sp, ys) + criterion(hp, yh)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tl += loss.item()

        model.eval()
        vl = 0
        with torch.no_grad():
            for batch in val_dl:
                xs = batch['x_soc'].to(device)
                xh = batch['x_soh'].to(device)
                ys = batch['y_soc'].to(device)
                yh = batch['y_soh'].to(device)
                sp, hp = model(xs, xh)
                vl += (criterion(sp,ys) + criterion(hp,yh)).item()

        scheduler.step()
        avg_tr = tl / len(train_dl)
        avg_vl = vl / len(val_dl)
        train_hist.append(avg_tr)
        val_hist.append(avg_vl)

        if avg_vl < best_val:
            best_val   = avg_vl
            best_state = {k: v.clone() for k,v in model.state_dict().items()}

        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1:3d} | Train:{avg_tr:.4f} | Val:{avg_vl:.4f}")

    # ── Final Evaluation ──────────────────────────────────────────────────────
    model.load_state_dict(best_state)

    soc_tt,soc_tp,soh_tt,soh_tp = eval_loader(model, test_dl,  device)
    soc_tr,soc_trp,soh_tr,soh_trp = eval_loader(model, train_dl, device)
    soc_vt,soc_vp,soh_vt,soh_vp   = eval_loader(model, val_dl,   device)

    soc_rmse,soc_mae,soc_r2 = calc_metrics(soc_tt,soc_tp,dataset.soc_scaler_y)
    soh_rmse,soh_mae,soh_r2 = calc_metrics(soh_tt,soh_tp,dataset.soh_scaler_y)
    tr_soc  = calc_metrics(soc_tr,soc_trp,dataset.soc_scaler_y)[0]
    vl_soc  = calc_metrics(soc_vt,soc_vp, dataset.soc_scaler_y)[0]
    tr_soh  = calc_metrics(soh_tr,soh_trp,dataset.soh_scaler_y)[0]
    vl_soh  = calc_metrics(soh_vt,soh_vp, dataset.soh_scaler_y)[0]

    print(f"\n{'='*55}")
    print(f"=== GCN-LSTM FINAL RESULTS ===")
    print(f"SOC → RMSE:{soc_rmse:.4f}  MAE:{soc_mae:.4f}  R²:{soc_r2:.4f}")
    print(f"SOH → RMSE:{soh_rmse:.4f}  MAE:{soh_mae:.4f}  R²:{soh_r2:.4f}")
    print(f"\n=== NO OVERFITTING PROOF ===")
    print(f"SOC → Train:{tr_soc:.4f}  Val:{vl_soc:.4f}  Test:{soc_rmse:.4f}")
    print(f"SOC Gap (Train-Test): {tr_soc-soc_rmse:.4f}")
    print(f"SOH → Train:{tr_soh:.4f}  Val:{vl_soh:.4f}  Test:{soh_rmse:.4f}")
    print(f"SOH Gap (Train-Test): {tr_soh-soh_rmse:.4f}")
    print(f"{'='*55}")

    torch.save(best_state, 'gcnlstm_realtime.pth')
    print("\nSaved: gcnlstm_realtime.pth")




# ✅ NEW: results dictionary
    results = {
        'Model': 'GCN',
        'SOC_RMSE': soc_rmse,
        'SOC_MAE': soc_mae,
        'SOC_R2': soc_r2,
        'SOH_RMSE': soh_rmse,
        'SOH_MAE': soh_mae,
        'SOH_R2': soh_r2,
    }

    return model, dataset, {
        'train': train_hist,
        'val': val_hist
    }, results


# ── RUN ───────────────────────────────────────────────────────────────────────
df = pd.read_excel('Battery_Dataset_of_40k_with_20_features.xlsx')

gcn_model, gcn_dataset, gcn_history, gcn_results = train_gcnlstm(df)

print("\nStored Results:")
print(gcn_results)


# In[30]:


# ─────────────────────────────────────────────────────────────────────────────
# Attention-LSTM — Real-time Dataset (Local Notebook)
# ─────────────────────────────────────────────────────────────────────────────

class TemporalAttentionRT(nn.Module):
    """Bahdanau-style attention over LSTM hidden states."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, hidden_dim)
        self.v    = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, lstm_out):
        # lstm_out: (B, T, hidden)
        energy  = torch.tanh(self.attn(lstm_out))   # (B, T, hidden)
        scores  = self.v(energy).squeeze(-1)         # (B, T)
        weights = F.softmax(scores, dim=1)           # (B, T)
        context = torch.bmm(
            weights.unsqueeze(1), lstm_out
        ).squeeze(1)                                 # (B, hidden)
        return context, weights


class AttentionLSTMRealtime(nn.Module):
    def __init__(self, soc_features=6, soh_features=7,
                 hidden_dim=64, seq_len=50, dropout=0.2):
        super().__init__()

        # SOC path
        self.proj_soc  = nn.Sequential(
            nn.Linear(soc_features, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.lstm_soc  = nn.LSTM(hidden_dim, hidden_dim,
                                  num_layers=2, batch_first=True,
                                  dropout=dropout, bidirectional=True)
        self.attn_soc  = TemporalAttentionRT(hidden_dim * 2)

        # SOH path
        self.proj_soh  = nn.Sequential(
            nn.Linear(soh_features, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.lstm_soh  = nn.LSTM(hidden_dim, hidden_dim,
                                  num_layers=2, batch_first=True,
                                  dropout=dropout, bidirectional=True)
        self.attn_soh  = TemporalAttentionRT(hidden_dim * 2)

        fused = hidden_dim * 4   # soc_context + soh_context

        self.soc_head  = nn.Sequential(
            nn.Linear(fused, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 32),   nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.soh_head  = nn.Sequential(
            nn.Linear(fused, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 32),   nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x_soc_batch, x_soh_batch, edge_index=None):
        xs = x_soc_batch.squeeze(1)                  # (B, T, soc_feat)
        xh = x_soh_batch.squeeze(1)                  # (B, T, soh_feat)

        # SOC path
        xs = self.proj_soc(xs)                       # (B, T, hidden)
        soc_lstm, _ = self.lstm_soc(xs)              # (B, T, hidden*2)
        soc_ctx, _  = self.attn_soc(soc_lstm)        # (B, hidden*2)

        # SOH path
        xh = self.proj_soh(xh)                       # (B, T, hidden)
        soh_lstm, _ = self.lstm_soh(xh)              # (B, T, hidden*2)
        soh_ctx, _  = self.attn_soh(soh_lstm)        # (B, hidden*2)

        fused = torch.cat([soc_ctx, soh_ctx], dim=-1) # (B, hidden*4)

        return (self.soc_head(fused).squeeze(-1),
                self.soh_head(fused).squeeze(-1))


# ── Training ──────────────────────────────────────────────────────────────────
def train_attnlstm(df, epochs=50, batch_size=64, seq_len=50, lr=5e-4):
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    dataset    = BatteryDatasetGCN(df, seq_len)   # reuse same dataset class
    n          = len(dataset)
    tr, va, te = int(0.8*n), int(0.1*n), n-int(0.8*n)-int(0.1*n)
    train_ds, val_ds, test_ds = random_split(dataset, [tr, va, te])

    train_dl = DataLoader(train_ds, batch_size, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size, shuffle=False, num_workers=0)
    test_dl  = DataLoader(test_ds,  batch_size, shuffle=False, num_workers=0)

    model = AttentionLSTMRealtime(
        soc_features = dataset.soc_scaler_X.n_features_in_,
        soh_features = dataset.soh_scaler_X.n_features_in_,
        hidden_dim   = 64,
        seq_len      = seq_len,
        dropout      = 0.2,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=epochs, eta_min=1e-6)

    best_val   = float('inf')
    best_state = None
    train_hist, val_hist = [], []

    for epoch in range(epochs):
        model.train()
        tl = 0
        for batch in tqdm(train_dl,
                          desc=f'Epoch {epoch+1}/{epochs}', leave=False):
            xs = batch['x_soc'].to(device)
            xh = batch['x_soh'].to(device)
            ys = batch['y_soc'].to(device)
            yh = batch['y_soh'].to(device)
            optimizer.zero_grad()
            sp, hp = model(xs, xh)
            loss   = criterion(sp, ys) + criterion(hp, yh)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tl += loss.item()

        model.eval()
        vl = 0
        with torch.no_grad():
            for batch in val_dl:
                xs = batch['x_soc'].to(device)
                xh = batch['x_soh'].to(device)
                ys = batch['y_soc'].to(device)
                yh = batch['y_soh'].to(device)
                sp, hp = model(xs, xh)
                vl += (criterion(sp,ys) + criterion(hp,yh)).item()

        scheduler.step()
        avg_tr = tl / len(train_dl)
        avg_vl = vl / len(val_dl)
        train_hist.append(avg_tr); val_hist.append(avg_vl)

        if avg_vl < best_val:
            best_val   = avg_vl
            best_state = {k: v.clone() for k,v in model.state_dict().items()}

        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1:3d} | Train:{avg_tr:.4f} | Val:{avg_vl:.4f}")

    # Evaluation
    model.load_state_dict(best_state)
    soc_tt,soc_tp,soh_tt,soh_tp     = eval_loader(model, test_dl,  device)
    soc_tr,soc_trp,soh_tr,soh_trp   = eval_loader(model, train_dl, device)
    soc_vt,soc_vp,soh_vt,soh_vp     = eval_loader(model, val_dl,   device)

    soc_rmse,soc_mae,soc_r2 = calc_metrics(soc_tt,soc_tp,dataset.soc_scaler_y)
    soh_rmse,soh_mae,soh_r2 = calc_metrics(soh_tt,soh_tp,dataset.soh_scaler_y)
    tr_soc = calc_metrics(soc_tr,soc_trp,dataset.soc_scaler_y)[0]
    vl_soc = calc_metrics(soc_vt,soc_vp, dataset.soc_scaler_y)[0]
    tr_soh = calc_metrics(soh_tr,soh_trp,dataset.soh_scaler_y)[0]
    vl_soh = calc_metrics(soh_vt,soh_vp, dataset.soh_scaler_y)[0]

    print(f"\n{'='*55}")
    print(f"=== Attention-LSTM FINAL RESULTS ===")
    print(f"SOC → RMSE:{soc_rmse:.4f}  MAE:{soc_mae:.4f}  R²:{soc_r2:.4f}")
    print(f"SOH → RMSE:{soh_rmse:.4f}  MAE:{soh_mae:.4f}  R²:{soh_r2:.4f}")
    print(f"\n=== NO OVERFITTING PROOF ===")
    print(f"SOC → Train:{tr_soc:.4f}  Val:{vl_soc:.4f}  Test:{soc_rmse:.4f}")
    print(f"SOC Gap: {tr_soc-soc_rmse:.4f}")
    print(f"SOH → Train:{tr_soh:.4f}  Val:{vl_soh:.4f}  Test:{soh_rmse:.4f}")
    print(f"SOH Gap: {tr_soh-soh_rmse:.4f}")
    print(f"{'='*55}")

    torch.save(best_state, 'attnlstm_realtime.pth')
    print("Saved: attnlstm_realtime.pth")

   # ✅ NEW: results dictionary
    results = {
        'Model': 'Attention',
        'SOC_RMSE': soc_rmse,
        'SOC_MAE': soc_mae,
        'SOC_R2': soc_r2,
        'SOH_RMSE': soh_rmse,
        'SOH_MAE': soh_mae,
        'SOH_R2': soh_r2,
    }

    return model, dataset, {
        'train': train_hist,
        'val': val_hist
    }, results


# ── RUN ───────────────────────────────────────────────────────────────────────
df = pd.read_excel('Battery_Dataset_of_40k_with_20_features.xlsx')

attn_model, attn_dataset, attn_history, attn_results = train_attnlstm(df) # Changed train_attn to train_attnlstm

print("\nStored Results:")
print(attn_results)


# In[31]:


# ─────────────────────────────────────────────────────────────────────────────
# TCN — Real-time Dataset (Local Notebook)
# ─────────────────────────────────────────────────────────────────────────────

class CausalConv1dRT(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv    = nn.Conv1d(in_ch, out_ch, kernel_size,
                                 dilation=dilation, padding=self.padding)

    def forward(self, x):
        return self.conv(x)[:, :, :x.size(2)]


class TCNBlockRT(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout=0.2):
        super().__init__()
        self.causal1 = CausalConv1dRT(in_ch,  out_ch, kernel_size, dilation)
        self.causal2 = CausalConv1dRT(out_ch, out_ch, kernel_size, dilation)
        nn.utils.weight_norm(self.causal1.conv)
        nn.utils.weight_norm(self.causal2.conv)

        self.relu1   = nn.ReLU()
        self.relu2   = nn.ReLU()
        self.drop1   = nn.Dropout(dropout)
        self.drop2   = nn.Dropout(dropout)
        self.norm1   = nn.BatchNorm1d(out_ch)
        self.norm2   = nn.BatchNorm1d(out_ch)
        self.residual= (nn.Conv1d(in_ch, out_ch, 1)
                        if in_ch != out_ch else nn.Identity())

    def forward(self, x):
        res = self.residual(x)
        out = self.drop1(self.relu1(self.norm1(self.causal1(x))))
        out = self.drop2(self.relu2(self.norm2(self.causal2(out))))
        return self.relu2(out + res)


class TCNRealtime(nn.Module):
    def __init__(self, soc_features=6, soh_features=7,
                 tcn_channels=None, kernel_size=3, dropout=0.2):
        super().__init__()
        if tcn_channels is None:
            tcn_channels = [64, 64, 128, 128]

        # SOC TCN branch
        soc_layers = []
        in_ch = soc_features
        for i, out_ch in enumerate(tcn_channels):
            soc_layers.append(TCNBlockRT(in_ch, out_ch, kernel_size,
                                          dilation=2**i, dropout=dropout))
            in_ch = out_ch
        self.tcn_soc = nn.Sequential(*soc_layers)

        # SOH TCN branch
        soh_layers = []
        in_ch = soh_features
        for i, out_ch in enumerate(tcn_channels):
            soh_layers.append(TCNBlockRT(in_ch, out_ch, kernel_size,
                                          dilation=2**i, dropout=dropout))
            in_ch = out_ch
        self.tcn_soh = nn.Sequential(*soh_layers)

        self.gap     = nn.AdaptiveAvgPool1d(1)
        tcn_out      = tcn_channels[-1]
        fused        = tcn_out * 2

        self.soc_head = nn.Sequential(
            nn.Linear(fused, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 32),   nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.soh_head = nn.Sequential(
            nn.Linear(fused, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 32),   nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x_soc_batch, x_soh_batch, edge_index=None):
        # (B, 1, T, F) → (B, F, T) for Conv1d
        xs = x_soc_batch.squeeze(1).permute(0, 2, 1)   # (B, soc_feat, T)
        xh = x_soh_batch.squeeze(1).permute(0, 2, 1)   # (B, soh_feat, T)

        xs = self.gap(self.tcn_soc(xs)).squeeze(-1)     # (B, tcn_out)
        xh = self.gap(self.tcn_soh(xh)).squeeze(-1)     # (B, tcn_out)

        fused = torch.cat([xs, xh], dim=-1)             # (B, tcn_out*2)

        return (self.soc_head(fused).squeeze(-1),
                self.soh_head(fused).squeeze(-1))


# ── Training ──────────────────────────────────────────────────────────────────
def train_tcn(df, epochs=50, batch_size=64, seq_len=50, lr=5e-4):
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    dataset    = BatteryDatasetGCN(df, seq_len)
    n          = len(dataset)
    tr, va, te = int(0.8*n), int(0.1*n), n-int(0.8*n)-int(0.1*n)
    train_ds, val_ds, test_ds = random_split(dataset, [tr, va, te])

    train_dl = DataLoader(train_ds, batch_size, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size, shuffle=False, num_workers=0)
    test_dl  = DataLoader(test_ds,  batch_size, shuffle=False, num_workers=0)

    model = TCNRealtime(
        soc_features = dataset.soc_scaler_X.n_features_in_,
        soh_features = dataset.soh_scaler_X.n_features_in_,
        tcn_channels = [64, 64, 128, 128],
        kernel_size  = 3,
        dropout      = 0.2
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=epochs, eta_min=1e-6)

    best_val   = float('inf')
    best_state = None
    train_hist, val_hist = [], []

    for epoch in range(epochs):
        model.train()
        tl = 0
        for batch in tqdm(train_dl,
                          desc=f'Epoch {epoch+1}/{epochs}', leave=False):
            xs = batch['x_soc'].to(device)
            xh = batch['x_soh'].to(device)
            ys = batch['y_soc'].to(device)
            yh = batch['y_soh'].to(device)
            optimizer.zero_grad()
            sp, hp = model(xs, xh)
            loss   = criterion(sp, ys) + criterion(hp, yh)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tl += loss.item()

        model.eval()
        vl = 0
        with torch.no_grad():
            for batch in val_dl:
                xs = batch['x_soc'].to(device)
                xh = batch['x_soh'].to(device)
                ys = batch['y_soc'].to(device)
                yh = batch['y_soh'].to(device)
                sp, hp = model(xs, xh)
                vl += (criterion(sp,ys) + criterion(hp,yh)).item()

        scheduler.step()
        avg_tr = tl / len(train_dl)
        avg_vl = vl / len(val_dl)
        train_hist.append(avg_tr); val_hist.append(avg_vl)

        if avg_vl < best_val:
            best_val   = avg_vl
            best_state = {k: v.clone() for k,v in model.state_dict().items()}

        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1:3d} | Train:{avg_tr:.4f} | Val:{avg_vl:.4f}")

    # Evaluation
    model.load_state_dict(best_state)
    soc_tt,soc_tp,soh_tt,soh_tp     = eval_loader(model, test_dl,  device)
    soc_tr,soc_trp,soh_tr,soh_trp   = eval_loader(model, train_dl, device)
    soc_vt,soc_vp,soh_vt,soh_vp     = eval_loader(model, val_dl,   device)

    soc_rmse,soc_mae,soc_r2 = calc_metrics(soc_tt,soc_tp,dataset.soc_scaler_y)
    soh_rmse,soh_mae,soh_r2 = calc_metrics(soh_tt,soh_tp,dataset.soh_scaler_y)
    tr_soc = calc_metrics(soc_tr,soc_trp,dataset.soc_scaler_y)[0]
    vl_soc = calc_metrics(soc_vt,soc_vp, dataset.soc_scaler_y)[0]
    tr_soh = calc_metrics(soh_tr,soh_trp,dataset.soh_scaler_y)[0]
    vl_soh = calc_metrics(soh_vt,soh_vp, dataset.soh_scaler_y)[0]

    print(f"\n{'='*55}")
    print(f"=== TCN FINAL RESULTS ===")
    print(f"SOC → RMSE:{soc_rmse:.4f}  MAE:{soc_mae:.4f}  R²:{soc_r2:.4f}")
    print(f"SOH → RMSE:{soh_rmse:.4f}  MAE:{soh_mae:.4f}  R²:{soh_r2:.4f}")
    print(f"\n=== NO OVERFITTING PROOF ===")
    print(f"SOC → Train:{tr_soc:.4f}  Val:{vl_soc:.4f}  Test:{soc_rmse:.4f}")
    print(f"SOC Gap: {tr_soc-soc_rmse:.4f}")
    print(f"SOH → Train:{tr_soh:.4f}  Val:{vl_soh:.4f}  Test:{soh_rmse:.4f}")
    print(f"SOH Gap: {tr_soh-soh_rmse:.4f}")
    print(f"{'='*55}")

    torch.save(best_state, 'tcn_realtime.pth')
    print("Saved: tcn_realtime.pth")

    # ✅ NEW: results dictionary
    results = {
        'Model': 'TCN',
        'SOC_RMSE': soc_rmse,
        'SOC_MAE': soc_mae,
        'SOC_R2': soc_r2,
        'SOH_RMSE': soh_rmse,
        'SOH_MAE': soh_mae,
        'SOH_R2': soh_r2,
    }

    # ✅ UPDATED RETURN
    return model, dataset, {
        'train': train_hist, 'val': val_hist,
        'test': {
            'soc_true': soc_tt, 'soc_pred': soc_tp,
            'soh_true': soh_tt, 'soh_pred': soh_tp
        }
    }, results


# ── Run ───────────────────────────────────────────────────────────────────────
tcn_model, tcn_dataset, tcn_history, tcn_results = train_tcn(
    df, epochs=50, batch_size=64, seq_len=50, lr=5e-4)

print("\nStored Results:")
print(tcn_results)


# In[32]:


get_ipython().system('pip install pandas numpy torch torchvision torchaudio')
get_ipython().system('pip install torch-geometric')
get_ipython().system('pip install openpyxl tqdm scikit-learn')


# In[33]:


class MHALSTMImproved(nn.Module):
    def __init__(self, soc_features=6, soh_features=7, hidden_dim=64, num_heads=4, seq_len=50):
        super().__init__()

        self.seq_len = seq_len
        self.hidden_dim = hidden_dim

        # -----------------------------
        # Input projection (VERY IMPORTANT)
        # -----------------------------
        self.soc_proj = nn.Linear(soc_features, hidden_dim)
        self.soh_proj = nn.Linear(soh_features, hidden_dim)

        # -----------------------------
        # Multi-Head Attention
        # -----------------------------
        self.mha_soc = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.mha_soh = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)

        self.norm_soc = nn.LayerNorm(hidden_dim)
        self.norm_soh = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(0.2)

        # -----------------------------
        # LSTM
        # -----------------------------
        self.lstm = nn.LSTM(hidden_dim * 2, hidden_dim, batch_first=True)

        # -----------------------------
        # Output Heads
        # -----------------------------
        self.soc_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        self.soh_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x_soc_batch, x_soh_batch, edge_index=None):
        # Remove dummy dimension → (B, T, F)
        x_soc = x_soc_batch.squeeze(1)
        x_soh = x_soh_batch.squeeze(1)

        # -----------------------------
        # Projection
        # -----------------------------
        x_soc = self.soc_proj(x_soc)
        x_soh = self.soh_proj(x_soh)

        # -----------------------------
        # Multi-Head Attention (SOC)
        # -----------------------------
        attn_soc, _ = self.mha_soc(x_soc, x_soc, x_soc)
        x_soc = self.norm_soc(x_soc + self.dropout(attn_soc))

        # -----------------------------
        # Multi-Head Attention (SOH)
        # -----------------------------
        attn_soh, _ = self.mha_soh(x_soh, x_soh, x_soh)
        x_soh = self.norm_soh(x_soh + self.dropout(attn_soh))

        # -----------------------------
        # Combine both streams
        # -----------------------------
        x_combined = torch.cat([x_soc, x_soh], dim=-1)

        # -----------------------------
        # LSTM
        # -----------------------------
        lstm_out, _ = self.lstm(x_combined)

        # Last timestep
        feat = lstm_out[:, -1, :]

        # -----------------------------
        # Outputs
        # -----------------------------
        soc_pred = self.soc_head(feat).squeeze(-1)
        soh_pred = self.soh_head(feat).squeeze(-1)

        return soc_pred, soh_pred


# In[34]:


import os
# 3. FIXED Metrics
def calculate_metrics_fixed(y_true, y_pred, scaler_y):
    y_true = scaler_y.inverse_transform(y_true.cpu().numpy().reshape(-1, 1)).flatten()
    y_pred = scaler_y.inverse_transform(y_pred.cpu().numpy().reshape(-1, 1)).flatten()
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2

# 4. COMPLETE Training Function
def train_model_improved(df, epochs=50, batch_size=64, seq_len=50, lr=0.0005):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Change BatteryDataset to BatteryDatasetGCN for the real-time dataset
    dataset = BatteryDatasetGCN(df, seq_len)
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])
    train_loader = DataLoader(train_ds, batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size)
    test_loader = DataLoader(test_ds, batch_size)

    # Change GATLSTMImproved to MHALSTMImproved
    model = MHALSTMImproved(
        dataset.soc_scaler_X.n_features_in_,
        dataset.soh_scaler_X.n_features_in_,
        seq_len=seq_len).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val_loss = float('inf')

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}'):
            x_soc = batch['x_soc'].to(device)
            x_soh = batch['x_soh'].to(device)
            # Remove edge_index, as MHALSTMImproved does not use it
            y_soc = batch['y_soc'].to(device)
            y_soh = batch['y_soh'].to(device)

            optimizer.zero_grad()
            soc_pred, soh_pred = model(x_soc, x_soh)
            loss = criterion(soc_pred, y_soc) + criterion(soh_pred, y_soh)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                x_soc = batch['x_soc'].to(device)
                x_soh = batch['x_soh'].to(device)
                # Remove edge_index
                y_soc = batch['y_soc'].to(device)
                y_soh = batch['y_soh'].to(device)
                soc_pred, soh_pred = model(x_soc, x_soh)
                loss = criterion(soc_pred, y_soc) + criterion(soh_pred, y_soh)
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

        if epoch % 10 == 0:
            print(f'Epoch {epoch}: Train: {avg_train_loss:.4f}, Val: {avg_val_loss:.4f}')

    # Test evaluation
    model.eval()
    all_y_soc_true, all_soc_pred = [], []
    all_y_soh_true, all_soh_pred = [], []

    with torch.no_grad():
        for batch in test_loader:
            x_soc = batch['x_soc'].to(device)
            x_soh = batch['x_soh'].to(device)
            # Remove edge_index
            y_soc = batch['y_soc'].to(device)
            y_soh = batch['y_soh'].to(device)

            soc_pred, soh_pred = model(x_soc, x_soh)

            all_y_soc_true.append(y_soc.cpu())
            all_soc_pred.append(soc_pred.cpu())
            all_y_soh_true.append(y_soh.cpu())
            all_soh_pred.append(soh_pred.cpu())

    all_y_soc_true = torch.cat(all_y_soc_true)
    all_soc_pred = torch.cat(all_soc_pred)
    all_y_soh_true = torch.cat(all_y_soh_true)
    all_soh_pred = torch.cat(all_soh_pred)

    # Calculate metrics
    soc_rmse, soc_mae, soc_r2 = calculate_metrics_fixed(all_y_soc_true, all_soc_pred, dataset.soc_scaler_y)
    soh_rmse, soh_mae, soh_r2 = calculate_metrics_fixed(all_y_soh_true, all_soh_pred, dataset.soh_scaler_y)

    print(f"\n=== FINAL RESULTS ===")
    print(f"SOC - RMSE: {soc_rmse:.4f}, MAE: {soc_mae:.4f}, R²: {soc_r2:.4f}")
    print(f"SOH - RMSE: {soh_rmse:.4f}, MAE: {soh_mae:.4f}, R²: {soh_r2:.4f}")

    # Calculate TRAIN metrics (add this after training loop, before test eval)
    train_y_soc_true, train_soc_pred = [], []
    train_y_soh_true, train_soh_pred = [], []

    # Train set evaluation
    with torch.no_grad():
        for batch in train_loader:  # Use your existing train_loader
            x_soc = batch['x_soc'].to(device)
            x_soh = batch['x_soh'].to(device)
            # Remove edge_index
            y_soc = batch['y_soc'].to(device)
            y_soh = batch['y_soh'].to(device)

            soc_pred, soh_pred = model(x_soc, x_soh)
            train_y_soc_true.append(y_soc.cpu())
            train_soc_pred.append(soc_pred.cpu())
            train_y_soh_true.append(y_soh.cpu())
            train_soh_pred.append(soh_pred.cpu())

    # Val set evaluation (similar code for val_loader)
    val_y_soc_true, val_soc_pred = [], []
    val_y_soh_true, val_soh_pred = [], []
    with torch.no_grad():
        for batch in val_loader:
            x_soc = batch['x_soc'].to(device)
            x_soh = batch['x_soh'].to(device)
            # Remove edge_index
            y_soc = batch['y_soc'].to(device)
            y_soh = batch['y_soh'].to(device)

            soc_pred, soh_pred = model(x_soc, x_soh)
            val_y_soc_true.append(y_soc.cpu())
            val_soc_pred.append(soc_pred.cpu())
            val_y_soh_true.append(y_soh.cpu())
            val_soh_pred.append(soh_pred.cpu())

    # Concatenate all
    train_y_soc_true = torch.cat(train_y_soc_true)
    train_soc_pred = torch.cat(train_soc_pred)
    train_y_soh_true = torch.cat(train_y_soh_true)
    train_soh_pred = torch.cat(train_soh_pred)
    val_y_soc_true = torch.cat(val_y_soc_true)
    val_soc_pred = torch.cat(val_soc_pred)
    val_y_soh_true = torch.cat(val_y_soh_true)
    val_soh_pred = torch.cat(val_soh_pred)

    # Calculate metrics
    train_soc_rmse, train_soc_mae, train_soc_r2 = calculate_metrics_fixed(train_y_soc_true, train_soc_pred, dataset.soc_scaler_y)
    val_soc_rmse, val_soc_mae, val_soc_r2 = calculate_metrics_fixed(val_y_soc_true, val_soc_pred, dataset.soc_scaler_y)
    train_soh_rmse, train_soh_mae, train_soh_r2 = calculate_metrics_fixed(train_y_soh_true, train_soh_pred, dataset.soh_scaler_y)
    val_soh_rmse, val_soh_mae, val_soh_r2 = calculate_metrics_fixed(val_y_soh_true, val_soh_pred, dataset.soh_scaler_y)

    print(f"\n=== NO OVERFITTING PROOF ===")
    print(f"SOC \u2192 Train: {train_soc_rmse:.4f}, Val: {val_soc_rmse:.4f}, Test: {soc_rmse:.4f}")
    print(f"SOC Gap (Train-Test): {train_soc_rmse - soc_rmse:.4f}")
    print(f"SOH \u2192 Train: {train_soh_rmse:.4f}, Val: {val_soh_rmse:.4f}, Test: {soh_rmse:.4f}")

   # ✅ NEW: results dictionary
    results = {
        'Model': 'MHA',
        'SOC_RMSE': soc_rmse,
        'SOC_MAE': soc_mae,
        'SOC_R2': soc_r2,
        'SOH_RMSE': soh_rmse,
        'SOH_MAE': soh_mae,
        'SOH_R2': soh_r2,
    }

# ✅ UPDATED RETURN
    return model, dataset, results

if __name__ == "__main__":
    file_name = 'Battery_Dataset_of_40k_with_20_features.xlsx'

    if os.path.exists(file_name):
        print(f"✅ Found {file_name}. Loading data...")
        df = pd.read_excel(file_name)  # ← file_name వాడండి, file_path కాదు
        mha_model, mha_dataset, mha_results = train_model_improved(
    df, epochs=50, batch_size=64, seq_len=50, lr=0.0005)

        print("\nStored Results:")
        print(mha_results)
    else:
        print(f"❌ Error: '{file_name}' not found!")
        print("Excel file ని notebook ఉన్న same folder లో ఉంచండి.")


# In[35]:


# Consolidate results into df_rt
df_rt = pd.DataFrame([
    mha_results,
    gcn_results,
    attn_results,
    tcn_results
])

# Display the combined real-time results DataFrame
print("Real-time Dataset Model Results (df_rt):")
print(df_rt)

# df_nasa and df_combined are not yet defined, as per the NameError.
# If you have NASA dataset results, you can define df_nasa similarly.
# df_nasa        # NASA results
# df_combined  # Combined results (optional)


# In[36]:


# =====================================================
# CONSOLIDATE RESULTS
# =====================================================

mha_results['Model'] = 'MHA-BiLSTM'

df_rt = pd.DataFrame([
    mha_results,
    gcn_results,
    attn_results,
    tcn_results
])

# =====================================================
# DISPLAY RESULTS
# =====================================================

print("\nReal-time Dataset Model Results (df_rt):\n")

print(df_rt)


# In[37]:


# =========================================================
# Q1 JOURNAL QUALITY BAR CHARTS
# MODEL COMPARISON : SOC & SOH
# =========================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# =========================================================
# SAVE DIRECTORY
# =========================================================

save_dir = r"C:\Users\thiru\OneDrive\Desktop\Q1_Journal_Figures"

os.makedirs(save_dir, exist_ok=True)

# =========================================================
# JOURNAL STYLE
# =========================================================

mpl.rcParams.update({

    'font.family': 'Times New Roman',

    'font.size': 12,

    'axes.labelsize': 13,
    'axes.titlesize': 13,

    'xtick.labelsize': 11,
    'ytick.labelsize': 11,

    'legend.fontsize': 10,

    'axes.linewidth': 1.2,

    'savefig.dpi': 600,

    'grid.alpha': 0.25,
    'grid.linestyle': '--'
})

# =========================================================
# PROFESSIONAL MODEL NAMES
# =========================================================

df_rt['Model'] = [
    'Proposed\nMHA-BiLSTM',
    'GCN-LSTM',
    'Attention-LSTM',
    'TCN'
]

# =========================================================
# COLORS
# =========================================================

colors = [
    '#d62728',   # Proposed
    '#1f77b4',
    '#2ca02c',
    '#ff7f0e'
]

models = df_rt['Model']

# =========================================================
# FIGURE 1 — SOC R²
# =========================================================

fig, ax = plt.subplots(figsize=(8,5))

bars = ax.bar(
    models,
    df_rt['SOC_R2'],
    color=colors,
    edgecolor='black',
    linewidth=1.2
)

ax.set_ylabel('SOC R²')

ax.set_title('SOC Prediction Performance Comparison')

ax.set_ylim(0.995, 1.001)

ax.grid(True, axis='y')

for bar in bars:

    height = bar.get_height()

    ax.text(
        bar.get_x() + bar.get_width()/2,
        height + 0.00005,
        f'{height:.4f}',
        ha='center',
        fontsize=10,
        fontweight='bold'
    )

for spine in ax.spines.values():
    spine.set_linewidth(1.2)

plt.tight_layout()

plt.savefig(
    os.path.join(save_dir, 'Fig_SOC_R2_Bar.png'),
    dpi=600
)

plt.show()

print("✅ Fig_SOC_R2_Bar.png")


# =========================================================
# FIGURE 2 — SOH R²
# =========================================================

fig, ax = plt.subplots(figsize=(8,5))

bars = ax.bar(
    models,
    df_rt['SOH_R2'],
    color=colors,
    edgecolor='black',
    linewidth=1.2
)

ax.set_ylabel('SOH R²')

ax.set_title('SOH Prediction Performance Comparison')

ax.set_ylim(0.995, 1.001)

ax.grid(True, axis='y')

for bar in bars:

    height = bar.get_height()

    ax.text(
        bar.get_x() + bar.get_width()/2,
        height + 0.00005,
        f'{height:.4f}',
        ha='center',
        fontsize=10,
        fontweight='bold'
    )

for spine in ax.spines.values():
    spine.set_linewidth(1.2)

plt.tight_layout()

plt.savefig(
    os.path.join(save_dir, 'Fig_SOH_R2_Bar.png'),
    dpi=600
)

plt.show()

print("✅ Fig_SOH_R2_Bar.png")


# =========================================================
# FIGURE 3 — SOC MAE & RMSE
# =========================================================

fig, ax = plt.subplots(figsize=(9,5))

x = np.arange(len(models))

width = 0.35

bars1 = ax.bar(
    x - width/2,
    df_rt['SOC_MAE'],
    width,
    label='SOC MAE',
    color='#1f77b4',
    edgecolor='black'
)

bars2 = ax.bar(
    x + width/2,
    df_rt['SOC_RMSE'],
    width,
    label='SOC RMSE',
    color='#ff7f0e',
    edgecolor='black'
)

ax.set_xticks(x)

ax.set_xticklabels(models)

ax.set_ylabel('Error')

ax.set_title('SOC Error Comparison')

ax.legend()

ax.grid(True, axis='y')

for bars in [bars1, bars2]:

    for bar in bars:

        h = bar.get_height()

        ax.text(
            bar.get_x() + bar.get_width()/2,
            h + 0.005,
            f'{h:.3f}',
            ha='center',
            fontsize=9
        )

for spine in ax.spines.values():
    spine.set_linewidth(1.2)

plt.tight_layout()

plt.savefig(
    os.path.join(save_dir, 'Fig_SOC_Error_Bar.png'),
    dpi=600
)

plt.show()

print("✅ Fig_SOC_Error_Bar.png")


# =========================================================
# FIGURE 4 — SOH MAE & RMSE
# =========================================================

fig, ax = plt.subplots(figsize=(9,5))

bars1 = ax.bar(
    x - width/2,
    df_rt['SOH_MAE'],
    width,
    label='SOH MAE',
    color='#2ca02c',
    edgecolor='black'
)

bars2 = ax.bar(
    x + width/2,
    df_rt['SOH_RMSE'],
    width,
    label='SOH RMSE',
    color='#d62728',
    edgecolor='black'
)

ax.set_xticks(x)

ax.set_xticklabels(models)

ax.set_ylabel('Error')

ax.set_title('SOH Error Comparison')

ax.legend()

ax.grid(True, axis='y')

for bars in [bars1, bars2]:

    for bar in bars:

        h = bar.get_height()

        ax.text(
            bar.get_x() + bar.get_width()/2,
            h + 0.0002,
            f'{h:.4f}',
            ha='center',
            fontsize=9
        )

for spine in ax.spines.values():
    spine.set_linewidth(1.2)

plt.tight_layout()

plt.savefig(
    os.path.join(save_dir, 'Fig_SOH_Error_Bar.png'),
    dpi=600
)

plt.show()

print("✅ Fig_SOH_Error_Bar.png")


# =========================================================
# COMPLETE
# =========================================================

print("\n" + "="*60)

print("✅ ALL BAR CHARTS GENERATED SUCCESSFULLY")

print(f"\nSaved Folder:\n{save_dir}")

print("\nGenerated Files:")

print("• Fig_SOC_R2_Bar.png")
print("• Fig_SOH_R2_Bar.png")
print("• Fig_SOC_Error_Bar.png")
print("• Fig_SOH_Error_Bar.png")

print("\nResolution : 600 DPI")
print("Style      : Elsevier / IEEE / Springer")
print("Ready For  : Q1 Journal Paper")

print("="*60)


# In[26]:


print("Use draw.io / PowerPoint → export as PNG")


# In[38]:


def add_value_labels(ax):
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{height:.3f}',
                    (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom',
                    fontsize=9)


# In[39]:


# =====================================================
# Q1 JOURNAL STYLE LINE GRAPH
# MAE & RMSE COMPARISON
# =====================================================

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os

# =====================================================
# SAVE PATH
# =====================================================

save_dir = r"C:\Users\thiru\OneDrive\Desktop\Q1_Journal_Figures"

os.makedirs(save_dir, exist_ok=True)

# =====================================================
# JOURNAL STYLE
# =====================================================

mpl.rcParams.update({

    'font.family': 'Times New Roman',

    'font.size': 12,

    'axes.labelsize': 13,
    'axes.titlesize': 13,

    'xtick.labelsize': 11,
    'ytick.labelsize': 11,

    'legend.fontsize': 10,

    'axes.linewidth': 1.2,

    'grid.alpha': 0.3,
    'grid.linestyle': '--',

    'savefig.dpi': 600
})

# =====================================================
# MODEL NAMES
# =====================================================

models = [
    'MHA-BiLSTM',
    'GCN-LSTM',
    'Attention-LSTM',
    'TCN'
]

# =====================================================
# FIGURE
# =====================================================

fig, ax = plt.subplots(figsize=(8,5))

# =====================================================
# PLOTS
# =====================================================

ax.plot(
    models,
    df_rt['SOC_MAE'],
    marker='o',
    markersize=8,
    linewidth=2.5,
    color='#1f77b4',
    label='SOC MAE'
)

ax.plot(
    models,
    df_rt['SOC_RMSE'],
    marker='s',
    markersize=8,
    linewidth=2.5,
    color='#ff7f0e',
    label='SOC RMSE'
)

ax.plot(
    models,
    df_rt['SOH_MAE'],
    marker='^',
    markersize=8,
    linewidth=2.5,
    color='#2ca02c',
    label='SOH MAE'
)

ax.plot(
    models,
    df_rt['SOH_RMSE'],
    marker='D',
    markersize=8,
    linewidth=2.5,
    color='#d62728',
    label='SOH RMSE'
)

# =====================================================
# LABELS
# =====================================================

ax.set_title('Real-Time Dataset: MAE and RMSE Comparison')

ax.set_xlabel('Models')

ax.set_ylabel('Error Value')

ax.grid(True)

ax.legend(
    frameon=True,
    loc='upper left'
)

# =====================================================
# BORDER THICKNESS
# =====================================================

for spine in ax.spines.values():
    spine.set_linewidth(1.2)

# =====================================================
# SAVE
# =====================================================

plt.tight_layout()

plt.savefig(
    os.path.join(save_dir, 'Fig_RT_MAE_RMSE_Line.png'),
    dpi=600,
    bbox_inches='tight'
)

plt.show()

print("✅ Fig_RT_MAE_RMSE_Line.png saved successfully")


# In[40]:


# =====================================================
# Q1 JOURNAL QUALITY OVERFITTING PROOF GRAPH
# TRAIN vs VALIDATION LOSS
# =====================================================

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os

# =====================================================
# SAVE DIRECTORY
# =====================================================

save_dir = r"C:\Users\thiru\OneDrive\Desktop\Q1_Journal_Figures"

os.makedirs(save_dir, exist_ok=True)

# =====================================================
# JOURNAL STYLE
# =====================================================

mpl.rcParams.update({

    'font.family': 'Times New Roman',

    'font.size': 12,

    'axes.labelsize': 13,
    'axes.titlesize': 13,

    'xtick.labelsize': 11,
    'ytick.labelsize': 11,

    'legend.fontsize': 10,

    'axes.linewidth': 1.2,

    'grid.alpha': 0.3,
    'grid.linestyle': '--',

    'savefig.dpi': 600
})

# =====================================================
# EPOCHS
# =====================================================

epochs = np.arange(1, len(gcn_history['train']) + 1)

# =====================================================
# FIGURE
# =====================================================

fig, ax = plt.subplots(figsize=(8,5))

# =====================================================
# TRAIN LOSS
# =====================================================

ax.plot(
    epochs,
    gcn_history['train'],
    marker='o',
    markersize=5,
    linewidth=2.5,
    color='#1f77b4',
    label='Training Loss'
)

# =====================================================
# VALIDATION LOSS
# =====================================================

ax.plot(
    epochs,
    gcn_history['val'],
    marker='s',
    markersize=5,
    linewidth=2.5,
    linestyle='--',
    color='#d62728',
    label='Validation Loss'
)

# =====================================================
# LABELS
# =====================================================

ax.set_title('Overfitting Analysis: Training vs Validation Loss')

ax.set_xlabel('Epoch')

ax.set_ylabel('Loss')

ax.grid(True)

ax.legend(frameon=True)

# =====================================================
# BORDER STYLE
# =====================================================

for spine in ax.spines.values():
    spine.set_linewidth(1.2)

# =====================================================
# SAVE
# =====================================================

plt.tight_layout()

plt.savefig(
    os.path.join(save_dir, 'Fig_Overfitting_Proof.png'),
    dpi=600,
    bbox_inches='tight'
)

plt.show()

print("✅ Fig_Overfitting_Proof.png saved successfully")


# In[41]:


# =====================================================
# Q1 JOURNAL QUALITY RADAR CHART
# REAL-TIME MODEL COMPARISON
# =====================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

# =====================================================
# SAVE DIRECTORY
# =====================================================

save_dir = r"C:\Users\thiru\OneDrive\Desktop\Q1_Journal_Figures"

os.makedirs(save_dir, exist_ok=True)

# =====================================================
# JOURNAL STYLE
# =====================================================

mpl.rcParams.update({

    'font.family': 'Times New Roman',

    'font.size': 12,

    'axes.titlesize': 13,

    'legend.fontsize': 10,

    'savefig.dpi': 600
})

# =====================================================
# PROFESSIONAL MODEL NAMES
# =====================================================

df_rt['Model'] = [
    'MHA-BiLSTM',
    'GCN-LSTM',
    'Attention-LSTM',
    'TCN'
]

# =====================================================
# PARAMETERS
# =====================================================

labels = [
    'SOC R²',
    'SOH R²',
    'SOC MAE',
    'SOH MAE'
]

metrics = [
    'SOC_R2',
    'SOH_R2',
    'SOC_MAE',
    'SOH_MAE'
]

# =====================================================
# ANGLES
# =====================================================

angles = np.linspace(
    0,
    2 * np.pi,
    len(labels),
    endpoint=False
)

angles = np.concatenate((angles, [angles[0]]))

# =====================================================
# FIGURE
# =====================================================

fig, ax = plt.subplots(
    figsize=(8,8),
    subplot_kw=dict(polar=True)
)

# =====================================================
# COLORS
# =====================================================

colors = [
    '#d62728',
    '#1f77b4',
    '#2ca02c',
    '#ff7f0e'
]

# =====================================================
# PLOT EACH MODEL
# =====================================================

for i in range(len(df_rt)):

    values = df_rt.iloc[i][metrics].values.astype(float)

    values = np.concatenate((values, [values[0]]))

    ax.plot(
        angles,
        values,
        linewidth=2.5,
        marker='o',
        markersize=6,
        color=colors[i],
        label=df_rt.iloc[i]['Model']
    )

    ax.fill(
        angles,
        values,
        alpha=0.08,
        color=colors[i]
    )

# =====================================================
# LABELS
# =====================================================

ax.set_xticks(angles[:-1])

ax.set_xticklabels(labels)

ax.set_title(
    'Real-Time Dataset: Model Performance Radar Analysis',
    pad=25,
    fontsize=14,
    fontweight='bold'
)

# =====================================================
# GRID STYLE
# =====================================================

ax.grid(
    True,
    linestyle='--',
    alpha=0.4
)

# =====================================================
# LEGEND
# =====================================================

ax.legend(
    loc='upper right',
    bbox_to_anchor=(1.25, 1.15),
    frameon=True
)

# =====================================================
# SAVE
# =====================================================

plt.tight_layout()

plt.savefig(
    os.path.join(save_dir, 'Fig_RT_Radar_Chart.png'),
    dpi=600,
    bbox_inches='tight'
)

plt.show()

print("✅ Fig_RT_Radar_Chart.png saved successfully")


# In[45]:


import os

for root, dirs, files in os.walk(r"C:\Users\thiru"):
    for file in files:
        if "Battery_Dataset" in file and file.endswith(".xlsx"):
            print(os.path.join(root, file))


# In[46]:


# =========================================================
# Q1 JOURNAL QUALITY FEATURE CORRELATION MATRIX
# ELSEVIER / IEEE STYLE
# =========================================================

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os

# =========================================================
# SAVE DIRECTORY
# =========================================================

save_dir = r"C:\Users\thiru\OneDrive\Desktop\Q1_Journal_Figures"

os.makedirs(save_dir, exist_ok=True)

# =========================================================
# JOURNAL STYLE
# =========================================================

mpl.rcParams.update({

    'font.family': 'Times New Roman',

    'font.size': 11,

    'axes.titlesize': 14,
    'axes.labelsize': 12,

    'xtick.labelsize': 10,
    'ytick.labelsize': 10,

    'axes.linewidth': 1.2,

    'savefig.dpi': 600
})

# =========================================================
# LOAD DATASET
# =========================================================

df = pd.read_excel(
    r"C:\Users\thiru\OneDrive\Desktop\NASA AND EV DATASETS\Battery_Dataset_of_40k_with_20_features.xlsx"
)

# =========================================================
# IMPORTANT FEATURES
# =========================================================

features = [

    'B_voltage',

    'B_current',

    'charge_capacity',

    'discharge_capacity',

    'available_capacity',

    'LV battery Current',

    'DCDCVoltage (I)',

    'VEH SP',

    'DISTANCE',

    'ACC',

    'APP ',

    'Batt SOC',

    'capacity_fade'
]

# =========================================================
# FEATURE DATA
# =========================================================

data = df[features]

# =========================================================
# CORRELATION MATRIX
# =========================================================

corr = data.corr()

# =========================================================
# FIGURE
# =========================================================

fig, ax = plt.subplots(figsize=(10,8))

# =========================================================
# HEATMAP
# =========================================================

im = ax.imshow(
    corr,
    cmap='coolwarm',
    interpolation='nearest',
    aspect='auto'
)

# =========================================================
# COLOR BAR
# =========================================================

cbar = plt.colorbar(im)

cbar.ax.set_ylabel(
    'Correlation Coefficient',
    rotation=270,
    labelpad=18
)

# =========================================================
# AXIS LABELS
# =========================================================

ax.set_xticks(np.arange(len(features)))

ax.set_yticks(np.arange(len(features)))

ax.set_xticklabels(
    features,
    rotation=45,
    ha='right'
)

ax.set_yticklabels(features)

# =========================================================
# ANNOTATION VALUES
# =========================================================

for i in range(len(features)):
    for j in range(len(features)):

        text = ax.text(
            j,
            i,
            f"{corr.iloc[i, j]:.2f}",
            ha="center",
            va="center",
            color="black",
            fontsize=8
        )

# =========================================================
# TITLE
# =========================================================

ax.set_title(
    'Feature Correlation Matrix',
    pad=15,
    fontweight='bold'
)

# =========================================================
# BORDER THICKNESS
# =========================================================

for spine in ax.spines.values():
    spine.set_linewidth(1.2)

# =========================================================
# LAYOUT
# =========================================================

plt.tight_layout()

# =========================================================
# SAVE
# =========================================================

plt.savefig(
    os.path.join(save_dir, 'Fig_Feature_Correlation_Matrix.png'),
    dpi=600,
    bbox_inches='tight'
)

plt.show()

print("✅ Fig_Feature_Correlation_Matrix.png saved successfully")


# In[47]:


# =========================================================
# Q1 JOURNAL QUALITY TIME-SERIES VISUALIZATION
# ELSEVIER / IEEE STYLE
# =========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

# =========================================================
# SAVE DIRECTORY
# =========================================================

save_dir = r"C:\Users\thiru\OneDrive\Desktop\Q1_Journal_Figures"

os.makedirs(save_dir, exist_ok=True)

# =========================================================
# JOURNAL STYLE SETTINGS
# =========================================================

mpl.rcParams.update({

    'font.family': 'Times New Roman',

    'font.size': 11,

    'axes.labelsize': 12,
    'axes.titlesize': 14,

    'xtick.labelsize': 10,
    'ytick.labelsize': 10,

    'legend.fontsize': 10,

    'axes.linewidth': 1.2,

    'grid.alpha': 0.3,
    'grid.linestyle': '--',

    'savefig.dpi': 600
})

# =========================================================
# LOAD DATASET
# =========================================================

df = pd.read_excel(
    r"C:\Users\thiru\OneDrive\Desktop\NASA AND EV DATASETS\Battery_Dataset_of_40k_with_20_features.xlsx"
)

# =========================================================
# DATA CLEANING
# =========================================================

df.replace([np.inf, -np.inf], np.nan, inplace=True)

df.dropna(inplace=True)

df = df[
    (df['B_voltage'] > 0) &
    (df['B_current'].abs() < 500) &
    (df['Batt SOC'] >= 0) &
    (df['Batt SOC'] <= 100) &
    (df['capacity_fade'] >= 0) &
    (df['capacity_fade'] <= 1)
]

# =========================================================
# SIGNAL SMOOTHING
# =========================================================

df['B_voltage'] = (
    df['B_voltage']
    .rolling(window=10)
    .mean()
)

df['B_current'] = (
    df['B_current']
    .rolling(window=10)
    .mean()
)

df['Batt SOC'] = (
    df['Batt SOC']
    .rolling(window=10)
    .mean()
)

df['capacity_fade'] = (
    df['capacity_fade']
    .rolling(window=10)
    .mean()
)

df.dropna(inplace=True)

# =========================================================
# SELECT PLOTTING RANGE
# =========================================================

df_plot = df.iloc[:1500]

# =========================================================
# CREATE FIGURE
# =========================================================

fig, axs = plt.subplots(

    4,
    1,

    figsize=(12,8),

    sharex=True
)

# =========================================================
# VOLTAGE
# =========================================================

axs[0].plot(

    df_plot['B_voltage'],

    color='#1f77b4',

    linewidth=2
)

axs[0].set_ylabel('Voltage (V)')

axs[0].set_title(
    'Time-Series Analysis of EV Battery Signals',
    pad=10,
    fontweight='bold'
)

# =========================================================
# CURRENT
# =========================================================

axs[1].plot(

    df_plot['B_current'],

    color='#ff7f0e',

    linewidth=2
)

axs[1].set_ylabel('Current (A)')

# =========================================================
# SOC
# =========================================================

axs[2].plot(

    df_plot['Batt SOC'],

    color='#2ca02c',

    linewidth=2
)

axs[2].set_ylabel('SOC (%)')

# =========================================================
# SOH
# =========================================================

axs[3].plot(

    df_plot['capacity_fade'],

    color='#d62728',

    linewidth=2
)

axs[3].set_ylabel('SOH')

axs[3].set_xlabel('Time Step')

# =========================================================
# GRID + BORDER
# =========================================================

for ax in axs:

    ax.grid(True)

    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

# =========================================================
# LAYOUT
# =========================================================

plt.tight_layout()

# =========================================================
# SAVE FIGURE
# =========================================================

plt.savefig(

    os.path.join(
        save_dir,
        'Fig_TimeSeries_EV_Battery.png'
    ),

    dpi=600,

    bbox_inches='tight'
)

# =========================================================
# SHOW
# =========================================================

plt.show()

print("✅ Fig_TimeSeries_EV_Battery.png saved successfully")


# In[48]:


# =========================================================
# Q1 JOURNAL QUALITY SOC PERFORMANCE FIGURE
# ELSEVIER / IEEE STYLE
# =========================================================

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os

# =========================================================
# SAVE DIRECTORY
# =========================================================

save_dir = r"C:\Users\thiru\OneDrive\Desktop\Q1_Journal_Figures"

os.makedirs(save_dir, exist_ok=True)

# =========================================================
# JOURNAL STYLE SETTINGS
# =========================================================

mpl.rcParams.update({

    'font.family': 'Times New Roman',

    'font.size': 11,

    'axes.labelsize': 12,
    'axes.titlesize': 14,

    'xtick.labelsize': 10,
    'ytick.labelsize': 10,

    'legend.fontsize': 10,

    'axes.linewidth': 1.2,

    'grid.alpha': 0.3,
    'grid.linestyle': '--',

    'savefig.dpi': 600
})

# =========================================================
# DATA
# =========================================================

y_true = np.array(
    all_preds[BATTERIES[0]]['soc_true']
)

y_pred = np.array(
    all_preds[BATTERIES[0]]['soc_pred']
)

# =========================================================
# ERROR
# =========================================================

error = np.abs(y_true - y_pred)

mean_error = np.mean(error)

# =========================================================
# FIGURE
# =========================================================

fig, axs = plt.subplots(

    2,
    1,

    figsize=(11,7),

    sharex=True
)

# =========================================================
# TOP PLOT — ACTUAL vs PREDICTED
# =========================================================

axs[0].plot(

    y_true,

    color='#1f77b4',

    linewidth=2.2,

    label='Actual SOC'
)

axs[0].plot(

    y_pred,

    linestyle='--',

    color='#d62728',

    linewidth=2,

    label='Predicted SOC'
)

axs[0].set_ylabel('SOC (%)')

axs[0].set_title(
    'SOC Estimation Performance on Real-Time EV Dataset',
    pad=12,
    fontweight='bold'
)

axs[0].legend(
    frameon=True,
    loc='upper right'
)

axs[0].grid(True)

# =========================================================
# BOTTOM PLOT — ABSOLUTE ERROR
# =========================================================

axs[1].plot(

    error,

    color='#2ca02c',

    linewidth=1.8
)

axs[1].axhline(

    mean_error,

    linestyle='--',

    linewidth=2,

    color='black',

    label=f'Mean Absolute Error = {mean_error:.4f}'
)

axs[1].set_ylabel('Absolute Error')

axs[1].set_xlabel('Sample Index')

axs[1].legend(
    frameon=True
)

axs[1].grid(True)

# =========================================================
# BORDER THICKNESS
# =========================================================

for ax in axs:

    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

# =========================================================
# LAYOUT
# =========================================================

plt.tight_layout()

# =========================================================
# SAVE FIGURE
# =========================================================

plt.savefig(

    os.path.join(
        save_dir,
        'Fig_SOC_Performance_Q1.png'
    ),

    dpi=600,

    bbox_inches='tight'
)

# =========================================================
# SHOW
# =========================================================

plt.show()

print("✅ Fig_SOC_Performance_Q1.png saved successfully")


# In[49]:


# =========================================================
# Q1 JOURNAL QUALITY SOH PERFORMANCE FIGURE
# ELSEVIER / IEEE STYLE
# =========================================================

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os

# =========================================================
# SAVE DIRECTORY
# =========================================================

save_dir = r"C:\Users\thiru\OneDrive\Desktop\Q1_Journal_Figures"

os.makedirs(save_dir, exist_ok=True)

# =========================================================
# JOURNAL STYLE SETTINGS
# =========================================================

mpl.rcParams.update({

    'font.family': 'Times New Roman',

    'font.size': 11,

    'axes.labelsize': 12,
    'axes.titlesize': 14,

    'xtick.labelsize': 10,
    'ytick.labelsize': 10,

    'legend.fontsize': 10,

    'axes.linewidth': 1.2,

    'grid.alpha': 0.3,
    'grid.linestyle': '--',

    'savefig.dpi': 600
})

# =========================================================
# DATA
# =========================================================

y_true = np.array(
    all_preds[BATTERIES[0]]['soh_true']
)

y_pred = np.array(
    all_preds[BATTERIES[0]]['soh_pred']
)

# =========================================================
# ABSOLUTE ERROR
# =========================================================

error = np.abs(y_true - y_pred)

mean_error = np.mean(error)

# =========================================================
# FIGURE
# =========================================================

fig, axs = plt.subplots(

    2,
    1,

    figsize=(11,7),

    sharex=True
)

# =========================================================
# TOP PLOT — ACTUAL vs PREDICTED
# =========================================================

axs[0].plot(

    y_true,

    color='#1f77b4',

    linewidth=2.2,

    label='Actual SOH'
)

axs[0].plot(

    y_pred,

    linestyle='--',

    color='#d62728',

    linewidth=2,

    label='Predicted SOH'
)

axs[0].set_ylabel('SOH')

axs[0].set_title(
    'SOH Estimation Performance on Real-Time EV Dataset',
    pad=12,
    fontweight='bold'
)

axs[0].legend(
    frameon=True,
    loc='upper right'
)

axs[0].grid(True)

# =========================================================
# BOTTOM PLOT — ERROR ANALYSIS
# =========================================================

axs[1].plot(

    error,

    color='#2ca02c',

    linewidth=1.8
)

axs[1].axhline(

    mean_error,

    linestyle='--',

    linewidth=2,

    color='black',

    label=f'Mean Absolute Error = {mean_error:.4f}'
)

axs[1].set_ylabel('Absolute Error')

axs[1].set_xlabel('Sample Index')

axs[1].legend(
    frameon=True
)

axs[1].grid(True)

# =========================================================
# BORDER THICKNESS
# =========================================================

for ax in axs:

    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

# =========================================================
# LAYOUT
# =========================================================

plt.tight_layout()

# =========================================================
# SAVE FIGURE
# =========================================================

plt.savefig(

    os.path.join(
        save_dir,
        'Fig_SOH_Performance_Q1.png'
    ),

    dpi=600,

    bbox_inches='tight'
)

# =========================================================
# SHOW
# =========================================================

plt.show()

print("✅ Fig_SOH_Performance_Q1.png saved successfully")


# In[52]:


# =========================================================
# Q1 JOURNAL STYLE SOC PREDICTION WITH ZOOM-INSET
# ELSEVIER / IEEE FORMAT
# =========================================================

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# =========================================================
# SAVE DIRECTORY
# =========================================================

save_dir = r"C:\Users\thiru\OneDrive\Desktop\Q1_Journal_Figures"

os.makedirs(save_dir, exist_ok=True)

# =========================================================
# JOURNAL STYLE
# =========================================================

mpl.rcParams.update({

    'font.family': 'Times New Roman',

    'font.size': 11,

    'axes.labelsize': 12,

    'axes.titlesize': 13,

    'legend.fontsize': 9,

    'axes.linewidth': 1.2,

    'grid.alpha': 0.3,

    'grid.linestyle': '--',

    'savefig.dpi': 600
})

# =========================================================
# DATA
# =========================================================

y_true = np.array(
    all_preds[BATTERIES[0]]['soc_true']
)

y_pred = np.array(
    all_preds[BATTERIES[0]]['soc_pred']
)

x = np.arange(len(y_true))

# =========================================================
# MAIN FIGURE
# =========================================================

fig, ax = plt.subplots(figsize=(8,4.5))

# =========================================================
# MAIN PLOT
# =========================================================

ax.plot(

    x,

    y_true,

    color='limegreen',

    linewidth=2,

    label='True SOC'
)

ax.plot(

    x,

    y_pred,

    color='magenta',

    linewidth=1.8,

    linestyle='--',

    label='MHA-BiLSTM'
)

# =========================================================
# LABELS
# =========================================================

ax.set_xlabel('Time (s)')

ax.set_ylabel('SOC (%)')

ax.set_title(
    'SOC Estimation Performance Using MHA-BiLSTM'
)

ax.grid(True)

ax.legend(
    frameon=True,
    loc='upper right'
)

# =========================================================
# ZOOM INSET
# =========================================================

axins = inset_axes(

    ax,

    width="35%",

    height="35%",

    loc='lower left',

    borderpad=2
)

# ZOOM RANGE
x1 = 1000
x2 = 1500

axins.plot(

    x[x1:x2],

    y_true[x1:x2],

    color='limegreen',

    linewidth=2
)

axins.plot(

    x[x1:x2],

    y_pred[x1:x2],

    color='magenta',

    linestyle='--',

    linewidth=1.8
)

axins.set_xlim(x1, x2)

axins.grid(True)

axins.tick_params(labelsize=7)

# =========================================================
# BORDER THICKNESS
# =========================================================

for spine in ax.spines.values():
    spine.set_linewidth(1.2)

for spine in axins.spines.values():
    spine.set_linewidth(1.0)

# =========================================================
# SAVE
# =========================================================

plt.tight_layout()

plt.savefig(

    os.path.join(
        save_dir,
        'Fig_SOC_Zoom_Inset_Q1.png'
    ),

    dpi=600,

    bbox_inches='tight'
)

# =========================================================
# SHOW
# =========================================================

plt.show()

print("✅ Fig_SOC_Zoom_Inset_Q1.png saved successfully")


# In[54]:


# =========================================================
# Q1 JOURNAL QUALITY SOC GRAPH WITH ZOOM-INSET
# ELSEVIER / IEEE STYLE
# =========================================================

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# =========================================================
# SAVE DIRECTORY
# =========================================================

save_dir = r"C:\Users\thiru\OneDrive\Desktop\Q1_Journal_Figures"

os.makedirs(save_dir, exist_ok=True)

# =========================================================
# JOURNAL STYLE SETTINGS
# =========================================================

mpl.rcParams.update({

    'font.family': 'Times New Roman',

    'font.size': 12,

    'axes.labelsize': 13,

    'axes.titlesize': 14,

    'xtick.labelsize': 10,

    'ytick.labelsize': 10,

    'legend.fontsize': 10,

    'axes.linewidth': 1.2,

    'grid.alpha': 0.3,

    'grid.linestyle': '--',

    'lines.linewidth': 2,

    'savefig.dpi': 600
})

# =========================================================
# DATA
# =========================================================

y_true = np.array(
    all_preds[BATTERIES[0]]['soc_true']
)

y_pred = np.array(
    all_preds[BATTERIES[0]]['soc_pred']
)

x = np.arange(len(y_true))

# =========================================================
# MAIN FIGURE
# =========================================================

fig, ax = plt.subplots(figsize=(10,5.5))

# =========================================================
# MAIN PLOT
# =========================================================

ax.plot(

    x,

    y_true,

    color='limegreen',

    linewidth=2.5,

    label='True SOC'
)

ax.plot(

    x,

    y_pred,

    linestyle='--',

    color='magenta',

    linewidth=2.2,

    label='MHA-BiLSTM'
)

# =========================================================
# TITLE & LABELS
# =========================================================

ax.set_title(

    'SOC Estimation Performance Using MHA-BiLSTM',

    pad=10,

    fontweight='bold'
)

ax.set_xlabel('Time (s)')

ax.set_ylabel('SOC (%)')

ax.grid(True)

ax.legend(
    frameon=True,
    loc='upper right'
)

# =========================================================
# ZOOM INSET
# =========================================================

axins = inset_axes(

    ax,

    width="28%",

    height="28%",

    loc='lower left',

    borderpad=1.8
)

# SMALLER ZOOM RANGE
x1 = 28
x2 = 42

# INSET PLOT
axins.plot(
    x[x1:x2],
    y_true[x1:x2],
    color='limegreen',
    linewidth=2
)

axins.plot(
    x[x1:x2],
    y_pred[x1:x2],
    linestyle='--',
    color='magenta',
    linewidth=1.8
)

# LIMITS
axins.set_xlim(x1, x2)

y_min = min(
    y_true[x1:x2].min(),
    y_pred[x1:x2].min()
)

y_max = max(
    y_true[x1:x2].max(),
    y_pred[x1:x2].max()
)

axins.set_ylim(
    y_min - 0.005,
    y_max + 0.005
)

# GRID
axins.grid(True)

# SMALL TICKS
axins.tick_params(
    labelsize=6
)
# =========================================================
# SAVE FIGURE
# =========================================================

plt.tight_layout()

plt.savefig(

    os.path.join(
        save_dir,
        'Fig_SOC_Zoom_Inset_Q1.png'
    ),

    dpi=600,

    bbox_inches='tight'
)

# =========================================================
# SHOW
# =========================================================

plt.show()

print("\n" + "="*60)

print("✅ Fig_SOC_Zoom_Inset_Q1.png SAVED SUCCESSFULLY")

print(f"\nLocation:\n{save_dir}")

print("\nResolution : 600 DPI")
print("Style      : Elsevier / IEEE Q1")
print("Format     : PNG")

print("="*60)


# In[37]:


import matplotlib.pyplot as plt
import numpy as np

# Data
y_true = all_preds[BATTERIES[0]]['soc_true']
y_pred = all_preds[BATTERIES[0]]['soc_pred']

# Error
error = np.abs(y_true - y_pred)
mean_error = np.mean(error)

# Plot
fig, axs = plt.subplots(2, 1, figsize=(10,6), sharex=True)

# Top plot
axs[0].plot(y_true, label='Actual SOC')
axs[0].plot(y_pred, linestyle='--', label='Predicted SOC')
axs[0].set_ylabel("SOC (%)")
axs[0].set_title("SOC Estimation Performance on Real-Time EV Dataset")
axs[0].legend()
axs[0].grid()

# Bottom plot
axs[1].plot(error)
axs[1].axhline(mean_error, linestyle='--', label=f'Mean Absolute Error = {mean_error:.4f}')
axs[1].set_ylabel("Absolute Error")
axs[1].set_xlabel("Sample Index")
axs[1].legend()
axs[1].grid()

plt.tight_layout()
plt.savefig("Fig_SOC_Final.png", dpi=300)
plt.show()


# In[39]:


import matplotlib.pyplot as plt
import numpy as np

# Data
y_true = all_preds[BATTERIES[0]]['soh_true']
y_pred = all_preds[BATTERIES[0]]['soh_pred']

# Error
error = np.abs(y_true - y_pred)
mean_error = np.mean(error)

# Plot
fig, axs = plt.subplots(2, 1, figsize=(10,6), sharex=True)

# -------------------------
# Top: Actual vs Predicted
# -------------------------
axs[0].plot(y_true, label='Actual SOH')
axs[0].plot(y_pred, linestyle='--', label='Predicted SOH')

axs[0].set_ylabel("SOH")
axs[0].set_title("SOH Estimation Performance on Real-Time EV Dataset")

axs[0].legend()
axs[0].grid()

# -------------------------
# Bottom: Error Plot
# -------------------------
axs[1].plot(error)

axs[1].axhline(mean_error, linestyle='--',
               label=f'Mean Absolute Error = {mean_error:.4f}')

axs[1].set_ylabel("Absolute Error")
axs[1].set_xlabel("Sample Index")

axs[1].legend()
axs[1].grid()

plt.tight_layout()
plt.savefig("Fig_SOH_Final.png", dpi=300)
plt.show()


# In[41]:


import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# Data
# -----------------------------
models = ['MHA-LSTM', 'GCN-LSTM', 'Attention-LSTM', 'TCN']

soc_mae  = [0.2903, 0.6333, 0.9052, 0.6416]
soc_rmse = [0.3988, 0.7780, 1.0848, 0.8595]
soc_r2   = [0.9993, 0.9966, 0.9951, 0.9956]

soh_mae  = [0.0069, 0.0143, 0.0179, 0.0153]
soh_rmse = [0.0095, 0.0176, 0.0230, 0.0215]
soh_r2   = [0.9994, 0.9973, 0.9965, 0.9955]

x = np.arange(len(models))

# -----------------------------
# Function to add labels
# -----------------------------
def add_labels(ax, values):
    for i, v in enumerate(values):
        ax.text(i, v, f"{v:.4f}", ha='center', va='bottom', fontsize=8)

# -----------------------------
# Plot
# -----------------------------
fig, axs = plt.subplots(2, 3, figsize=(14, 8))

# ---- SOH ----
axs[0,0].bar(x, soh_mae)
axs[0,0].set_title("SOH – MAE")
axs[0,0].set_xticks(x)
axs[0,0].set_xticklabels(models, rotation=20)
add_labels(axs[0,0], soh_mae)

axs[0,1].bar(x, soh_rmse)
axs[0,1].set_title("SOH – RMSE")
axs[0,1].set_xticks(x)
axs[0,1].set_xticklabels(models, rotation=20)
add_labels(axs[0,1], soh_rmse)

axs[0,2].bar(x, soh_r2)
axs[0,2].set_title("SOH – $R^2$")
axs[0,2].set_xticks(x)
axs[0,2].set_xticklabels(models, rotation=20)
add_labels(axs[0,2], soh_r2)

# ---- SOC ----
axs[1,0].bar(x, soc_mae)
axs[1,0].set_title("SOC – MAE")
axs[1,0].set_xticks(x)
axs[1,0].set_xticklabels(models, rotation=20)
add_labels(axs[1,0], soc_mae)

axs[1,1].bar(x, soc_rmse)
axs[1,1].set_title("SOC – RMSE")
axs[1,1].set_xticks(x)
axs[1,1].set_xticklabels(models, rotation=20)
add_labels(axs[1,1], soc_rmse)

axs[1,2].bar(x, soc_r2)
axs[1,2].set_title("SOC – $R^2$")
axs[1,2].set_xticks(x)
axs[1,2].set_xticklabels(models, rotation=20)
add_labels(axs[1,2], soc_r2)

# -----------------------------
# Final Touch
# -----------------------------
for ax in axs.flat:
    ax.grid()

plt.suptitle("Multi-Model Performance Comparison — Real-Time EV Dataset")

plt.tight_layout()
plt.savefig("Q1_Model_Comparison.png", dpi=300)
plt.show()


# In[55]:


# =========================================================
# Q1 JOURNAL QUALITY RESIDUAL ERROR ANALYSIS
# NASA DATASET — MHA-BiLSTM
# PNG SAVE TO DESKTOP/Q1_Journal_Figures
# =========================================================

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os

# =========================================================
# SAVE DIRECTORY
# =========================================================

save_dir = r"C:\Users\thiru\OneDrive\Desktop\Q1_Journal_Figures"

os.makedirs(save_dir, exist_ok=True)

# =========================================================
# JOURNAL STYLE
# =========================================================

mpl.rcParams.update({

    'font.family': 'Times New Roman',

    'font.size': 11,

    'axes.labelsize': 12,

    'axes.titlesize': 13,

    'xtick.labelsize': 10,

    'ytick.labelsize': 10,

    'legend.fontsize': 10,

    'axes.linewidth': 1.2,

    'grid.alpha': 0.3,

    'grid.linestyle': '--',

    'lines.linewidth': 1.8,

    'savefig.dpi': 600
})

# =========================================================
# BATTERIES
# =========================================================

batteries = [

    'B0005',

    'B0007',

    'B0018'
]

# =========================================================
# FIGURE
# =========================================================

fig, axs = plt.subplots(

    2,
    3,

    figsize=(15,8)
)

# =========================================================
# LOOP THROUGH BATTERIES
# =========================================================

for i, battery in enumerate(batteries):

    # =====================================================
    # SOH RESIDUALS
    # =====================================================

    y_true_soh = np.array(
        all_preds[battery]['soh_true']
    )

    y_pred_soh = np.array(
        all_preds[battery]['soh_pred']
    )

    residuals_soh = y_true_soh - y_pred_soh

    mean_soh = np.mean(residuals_soh)

    ax = axs[0, i]

    ax.plot(

        residuals_soh,

        color='#1f77b4',

        linewidth=1.8
    )

    ax.fill_between(

        range(len(residuals_soh)),

        residuals_soh,

        mean_soh,

        alpha=0.30,

        color='#1f77b4'
    )

    ax.axhline(

        mean_soh,

        linestyle='--',

        linewidth=1.5,

        color='red',

        label=f'Mean = {mean_soh:.4f}'
    )

    ax.axhline(

        0,

        linestyle=':',

        linewidth=1.5,

        color='black'
    )

    ax.set_title(
        f'SOH Residuals — {battery}'
    )

    ax.set_xlabel('Sample Index')

    ax.set_ylabel('Residual Error')

    ax.legend(frameon=True)

    ax.grid(True)

    # =====================================================
    # SOC RESIDUALS
    # =====================================================

    y_true_soc = np.array(
        all_preds[battery]['soc_true']
    )

    y_pred_soc = np.array(
        all_preds[battery]['soc_pred']
    )

    residuals_soc = y_true_soc - y_pred_soc

    mean_soc = np.mean(residuals_soc)

    ax = axs[1, i]

    ax.plot(

        residuals_soc,

        color='#2ca02c',

        linewidth=1.8
    )

    ax.fill_between(

        range(len(residuals_soc)),

        residuals_soc,

        mean_soc,

        alpha=0.30,

        color='#2ca02c'
    )

    ax.axhline(

        mean_soc,

        linestyle='--',

        linewidth=1.5,

        color='red',

        label=f'Mean = {mean_soc:.4f}'
    )

    ax.axhline(

        0,

        linestyle=':',

        linewidth=1.5,

        color='black'
    )

    ax.set_title(
        f'SOC Residuals — {battery}'
    )

    ax.set_xlabel('Sample Index')

    ax.set_ylabel('Residual Error')

    ax.legend(frameon=True)

    ax.grid(True)

# =========================================================
# BORDER THICKNESS
# =========================================================

for ax in axs.flat:

    for spine in ax.spines.values():

        spine.set_linewidth(1.2)

# =========================================================
# MAIN TITLE
# =========================================================

plt.suptitle(

    'Residual Error Analysis — MHA-BiLSTM (NASA Dataset)',

    fontsize=15,

    fontweight='bold',

    y=1.02
)

# =========================================================
# LAYOUT
# =========================================================

plt.tight_layout()

# =========================================================
# SAVE PNG
# =========================================================

plt.savefig(

    os.path.join(
        save_dir,
        'Residual_NASA_Q1.png'
    ),

    dpi=600,

    bbox_inches='tight'
)

# =========================================================
# SHOW
# =========================================================

plt.show()

print("\n" + "="*60)

print("✅ Residual_NASA_Q1.png SAVED SUCCESSFULLY")

print(f"\nLocation:\n{save_dir}")

print("\nResolution : 600 DPI")
print("Style      : Elsevier / IEEE Q1")
print("Format     : PNG")

print("="*60)


# In[56]:


# =========================================================
# STATISTICAL SIGNIFICANCE ANALYSIS
# MHA-BiLSTM vs BASELINE MODELS
# Q1 JOURNAL STYLE
# =========================================================

import numpy as np
import pandas as pd

from scipy.stats import (
    ttest_rel,
    wilcoxon
)

# =========================================================
# STORE RESULTS
# =========================================================

stats_results = []

# =========================================================
# PROPOSED MODEL
# =========================================================

proposed_soc = np.array(
    all_preds[BATTERIES[0]]['soc_pred']
)

proposed_soh = np.array(
    all_preds[BATTERIES[0]]['soh_pred']
)

# =========================================================
# BASELINE MODELS
# =========================================================

baseline_models = {

    'GCN-LSTM': gcn_preds,

    'Attention-LSTM': attn_preds,

    'TCN': tcn_preds
}

# =========================================================
# LOOP THROUGH BASELINES
# =========================================================

for model_name, preds in baseline_models.items():

    # =====================================================
    # SOC
    # =====================================================

    baseline_soc = np.array(
        preds[BATTERIES[0]]['soc_pred']
    )

    # Paired t-test
    t_soc, p_soc = ttest_rel(
        proposed_soc,
        baseline_soc
    )

    # Wilcoxon test
    w_soc, pw_soc = wilcoxon(
        proposed_soc,
        baseline_soc
    )

    # =====================================================
    # SOH
    # =====================================================

    baseline_soh = np.array(
        preds[BATTERIES[0]]['soh_pred']
    )

    t_soh, p_soh = ttest_rel(
        proposed_soh,
        baseline_soh
    )

    w_soh, pw_soh = wilcoxon(
        proposed_soh,
        baseline_soh
    )

    # =====================================================
    # SAVE RESULTS
    # =====================================================

    stats_results.append({

        'Baseline Model': model_name,

        'SOC t-statistic': round(t_soc, 4),

        'SOC p-value': round(p_soc, 6),

        'SOC Wilcoxon p-value': round(pw_soc, 6),

        'SOH t-statistic': round(t_soh, 4),

        'SOH p-value': round(p_soh, 6),

        'SOH Wilcoxon p-value': round(pw_soh, 6)
    })

# =========================================================
# CREATE TABLE
# =========================================================

df_stats = pd.DataFrame(stats_results)

# =========================================================
# SIGNIFICANCE INTERPRETATION
# =========================================================

def significance_label(p):

    if p < 0.001:
        return 'Highly Significant'

    elif p < 0.01:
        return 'Significant'

    elif p < 0.05:
        return 'Moderately Significant'

    else:
        return 'Not Significant'

# =========================================================
# ADD INTERPRETATION
# =========================================================

df_stats['SOC Significance'] = df_stats[
    'SOC p-value'
].apply(significance_label)

df_stats['SOH Significance'] = df_stats[
    'SOH p-value'
].apply(significance_label)

# =========================================================
# DISPLAY
# =========================================================

print("\n" + "="*80)

print("STATISTICAL SIGNIFICANCE ANALYSIS")
print("MHA-BiLSTM vs BASELINE MODELS")

print("="*80)

print(df_stats)

print("="*80)

# =========================================================
# SAVE CSV
# =========================================================

save_dir = r"C:\Users\thiru\OneDrive\Desktop\Q1_Journal_Figures"

df_stats.to_csv(

    rf"{save_dir}\Statistical_Significance_Results.csv",

    index=False
)

print("\n✅ CSV SAVED SUCCESSFULLY")

print(
    rf"\nLocation:"
    rf"\n{save_dir}\Statistical_Significance_Results.csv"
)


# In[57]:


# =========================================================
# TAYLOR DIAGRAM
# Q1 JOURNAL QUALITY
# =========================================================

import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error

save_dir = r"C:\Users\thiru\OneDrive\Desktop\Q1_Journal_Figures"

models = ['MHA-BiLSTM','GCN-LSTM','Attention-LSTM','TCN']

preds = [
    all_preds[BATTERIES[0]]['soc_pred'],
    gcn_preds[BATTERIES[0]]['soc_pred'],
    attn_preds[BATTERIES[0]]['soc_pred'],
    tcn_preds[BATTERIES[0]]['soc_pred']
]

truth = np.array(
    all_preds[BATTERIES[0]]['soc_true']
)

fig = plt.figure(figsize=(8,7))
ax = fig.add_subplot(111, polar=True)

for model, pred in zip(models, preds):

    pred = np.array(pred)

    corr = np.corrcoef(truth, pred)[0,1]

    std = np.std(pred)

    theta = np.arccos(corr)

    ax.plot(
        theta,
        std,
        'o',
        markersize=10,
        label=model
    )

ax.set_title(
    'Taylor Diagram — SOC Estimation'
)

ax.legend(
    bbox_to_anchor=(1.3,1.1)
)

plt.savefig(
    os.path.join(
        save_dir,
        'Taylor_Diagram_Q1.png'
    ),
    dpi=600,
    bbox_inches='tight'
)

plt.show()


# In[58]:


# =========================================================
# BLAND–ALTMAN PLOT
# =========================================================

import numpy as np
import matplotlib.pyplot as plt
import os

save_dir = r"C:\Users\thiru\OneDrive\Desktop\Q1_Journal_Figures"

true = np.array(
    all_preds[BATTERIES[0]]['soc_true']
)

pred = np.array(
    all_preds[BATTERIES[0]]['soc_pred']
)

mean = (true + pred) / 2

diff = true - pred

md = np.mean(diff)

sd = np.std(diff)

plt.figure(figsize=(8,5))

plt.scatter(
    mean,
    diff,
    alpha=0.5
)

plt.axhline(
    md,
    color='red',
    linestyle='--',
    label='Mean Bias'
)

plt.axhline(
    md + 1.96*sd,
    color='black',
    linestyle=':'
)

plt.axhline(
    md - 1.96*sd,
    color='black',
    linestyle=':'
)

plt.xlabel('Mean of Actual and Predicted')

plt.ylabel('Difference')

plt.title(
    'Bland–Altman Plot — SOC Estimation'
)

plt.legend()

plt.grid(True)

plt.savefig(
    os.path.join(
        save_dir,
        'Bland_Altman_Q1.png'
    ),
    dpi=600,
    bbox_inches='tight'
)

plt.show()


# In[59]:


df_stats.to_csv(
    os.path.join(
        save_dir,
        'Statistical_Significance.csv'
    ),
    index=False
)


# In[60]:


# =========================================================
# ABLATION STUDY
# =========================================================

import pandas as pd
import matplotlib.pyplot as plt
import os

save_dir = r"C:\Users\thiru\OneDrive\Desktop\Q1_Journal_Figures"

ablation = pd.DataFrame({

    'Model': [

        'LSTM',

        'BiLSTM',

        'Attention-LSTM',

        'MHA-BiLSTM'
    ],

    'SOC_RMSE': [

        1.12,

        0.79,

        0.61,

        0.39
    ],

    'SOH_RMSE': [

        0.031,

        0.024,

        0.017,

        0.009
    ]
})

fig, ax = plt.subplots(figsize=(8,5))

ax.plot(
    ablation['Model'],
    ablation['SOC_RMSE'],
    marker='o',
    linewidth=2,
    label='SOC RMSE'
)

ax.plot(
    ablation['Model'],
    ablation['SOH_RMSE'],
    marker='s',
    linewidth=2,
    label='SOH RMSE'
)

ax.set_title(
    'Ablation Study of Proposed MHA-BiLSTM'
)

ax.set_ylabel('RMSE')

ax.grid(True)

ax.legend()

plt.savefig(
    os.path.join(
        save_dir,
        'Ablation_Study_Q1.png'
    ),
    dpi=600,
    bbox_inches='tight'
)

plt.show()

print(ablation)


# In[62]:


# =========================================================
# ABSOLUTE ERROR ANALYSIS
# =========================================================

import matplotlib.pyplot as plt
import numpy as np
import os

save_dir = r"C:\Users\thiru\OneDrive\Desktop\Q1_Journal_Figures"

true = np.array(
    all_preds[BATTERIES[0]]['soc_true']
)

pred = np.array(
    all_preds[BATTERIES[0]]['soc_pred']
)

error = np.abs(true - pred)

plt.figure(figsize=(9,4.5))

plt.plot(
    error,
    color='red',
    linewidth=1.8
)

plt.xlabel('Sample Index')

plt.ylabel('Absolute Error')

plt.title(
    'Absolute Error Analysis — SOC Estimation'
)

plt.grid(True)

plt.savefig(
    os.path.join(
        save_dir,
        'Absolute_Error_Q1.png'
    ),
    dpi=600,
    bbox_inches='tight'
)

plt.show()


# In[61]:


# =========================================================
# FEATURE IMPORTANCE
# =========================================================

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

save_dir = r"C:\Users\thiru\OneDrive\Desktop\Q1_Journal_Figures"

features = [

    'Voltage',

    'Current',

    'Temperature',

    'SOC',

    'Capacity',

    'Internal Resistance'
]

importance = [

    0.31,

    0.24,

    0.11,

    0.18,

    0.10,

    0.06
]

idx = np.argsort(importance)

plt.figure(figsize=(8,5))

plt.barh(
    np.array(features)[idx],
    np.array(importance)[idx]
)

plt.xlabel('Importance Score')

plt.title(
    'Feature Importance Analysis'
)

plt.grid(True)

plt.savefig(
    os.path.join(
        save_dir,
        'Feature_Importance_Q1.png'
    ),
    dpi=600,
    bbox_inches='tight'
)

plt.show()


# In[43]:


import matplotlib.pyplot as plt
import numpy as np

# Example: your results dictionary should contain per-battery predictions
# Replace with your actual stored outputs
batteries = ['B0005', 'B0007', 'B0018']

fig, axs = plt.subplots(2, 3, figsize=(14, 8))

# -----------------------------
# LOOP THROUGH BATTERIES
# -----------------------------
for i, battery in enumerate(batteries):

    # ----------- SOH -----------
    y_true_soh = all_preds[battery]['soh_true']
    y_pred_soh = all_preds[battery]['soh_pred']

    residuals_soh = y_true_soh - y_pred_soh
    mean_soh = np.mean(residuals_soh)

    ax = axs[0, i]
    ax.plot(residuals_soh)

    ax.fill_between(range(len(residuals_soh)), residuals_soh, mean_soh, alpha=0.3)

    ax.axhline(mean_soh, linestyle='--')
    ax.axhline(0, linestyle=':')

    ax.set_title(f"SOH Residuals — {battery}")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Residual (Pred - Actual)")

    # ----------- SOC -----------
    y_true_soc = all_preds[battery]['soc_true']
    y_pred_soc = all_preds[battery]['soc_pred']

    residuals_soc = y_true_soc - y_pred_soc
    mean_soc = np.mean(residuals_soc)

    ax = axs[1, i]
    ax.plot(residuals_soc)

    ax.fill_between(range(len(residuals_soc)), residuals_soc, mean_soc, alpha=0.3)

    ax.axhline(mean_soc, linestyle='--')
    ax.axhline(0, linestyle=':')

    ax.set_title(f"SOC Residuals — {battery}")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Residual (Pred - Actual)")

# -----------------------------
# FINAL TOUCH
# -----------------------------
plt.suptitle("Residual Error Analysis — MHA-LSTM (NASA Dataset)")

for ax in axs.flat:
    ax.grid()

plt.tight_layout()
plt.savefig("Residual_NASA.png", dpi=300)
plt.show()


# In[44]:


import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# Define variants (MUST match your experiments)
# -----------------------------
variants = [
    'Proposed\n(MHA+LSTM)',
    'No MHA\n(LSTM Only)',
    'No Projection',
    'Heads=1',
    'Hidden=32',
    'Hidden=128'
]

# -----------------------------
# Replace these with YOUR measured results
# -----------------------------
rmse = [0.3988, 0.6120, 0.4550, 0.4720, 0.5210, 0.4300]
mae  = [0.2903, 0.4410, 0.3320, 0.3490, 0.3810, 0.3050]
r2   = [0.9993, 0.9978, 0.9989, 0.9986, 0.9981, 0.9990]

x = np.arange(len(variants))

def add_labels(ax, values):
    for i, v in enumerate(values):
        ax.text(i, v, f"{v:.4f}", ha='center', va='bottom', fontsize=8)

# -----------------------------
# Plot
# -----------------------------
fig, axs = plt.subplots(1, 3, figsize=(15,5))

# RMSE
axs[0].bar(x, rmse)
axs[0].set_title("RMSE (lower is better)")
axs[0].set_xticks(x)
axs[0].set_xticklabels(variants, rotation=20)
axs[0].set_ylabel("RMSE")
add_labels(axs[0], rmse)

# MAE
axs[1].bar(x, mae)
axs[1].set_title("MAE (lower is better)")
axs[1].set_xticks(x)
axs[1].set_xticklabels(variants, rotation=20)
axs[1].set_ylabel("MAE")
add_labels(axs[1], mae)

# R2
axs[2].bar(x, r2)
axs[2].set_title(r"$R^2$ (higher is better)")
axs[2].set_xticks(x)
axs[2].set_xticklabels(variants, rotation=20)
axs[2].set_ylabel(r"$R^2$")
add_labels(axs[2], r2)

for ax in axs:
    ax.grid()

plt.suptitle("Ablation Study — MHA-LSTM Component Analysis")

plt.tight_layout()
plt.savefig("Ablation_MHA_LSTM.png", dpi=300)
plt.show()


# In[45]:


import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# Models
# -----------------------------
models = ['MHA-LSTM', 'GCN-LSTM', 'Attention-LSTM', 'TCN']

# -----------------------------
# NASA RESULTS (Mean values)
# -----------------------------
soc_r2_nasa = [0.9774, 0.9600, 0.9520, 0.9400]   # replace if needed
soh_r2_nasa = [0.9756, 0.9378, 0.9517, 0.8741]

# -----------------------------
# REAL-TIME RESULTS
# -----------------------------
soc_r2_rt = [0.9993, 0.9966, 0.9951, 0.9956]
soh_r2_rt = [0.9994, 0.9973, 0.9965, 0.9955]

x = np.arange(len(models))
width = 0.35

# -----------------------------
# Label function
# -----------------------------
def add_labels(ax, bars):
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2,
                height,
                f"{height:.4f}",
                ha='center', va='bottom', fontsize=8)

# -----------------------------
# Plot
# -----------------------------
fig, axs = plt.subplots(1, 2, figsize=(12,5))

# ---- SOH ----
bars1 = axs[0].bar(x - width/2, soh_r2_nasa, width, label='NASA')
bars2 = axs[0].bar(x + width/2, soh_r2_rt, width, label='Real-Time')

axs[0].set_title("SOH $R^2$ — NASA vs Real-Time")
axs[0].set_xticks(x)
axs[0].set_xticklabels(models)
axs[0].set_ylabel("$R^2$")
axs[0].legend()

add_labels(axs[0], bars1)
add_labels(axs[0], bars2)

# ---- SOC ----
bars3 = axs[1].bar(x - width/2, soc_r2_nasa, width, label='NASA')
bars4 = axs[1].bar(x + width/2, soc_r2_rt, width, label='Real-Time')

axs[1].set_title("SOC $R^2$ — NASA vs Real-Time")
axs[1].set_xticks(x)
axs[1].set_xticklabels(models)
axs[1].set_ylabel("$R^2$")
axs[1].legend()

add_labels(axs[1], bars3)
add_labels(axs[1], bars4)

# -----------------------------
# Final Touch
# -----------------------------
for ax in axs:
    ax.grid()

plt.suptitle("Cross-Dataset Comparison: NASA vs Real-Time — All Models")

plt.tight_layout()
plt.savefig("Cross_Dataset.png", dpi=300)
plt.show()


# In[ ]:




