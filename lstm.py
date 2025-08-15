import pandas as pd, numpy as np, math, time
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

CSV_PATH = "combined.csv"
DATE_COL = "Date"
CLOSE_COL = "Close"
LOOKBACK = 30
EPOCHS = 50
BATCH = 64
TC_BPS = 10
RANDOM_SEED = 42
SPLIT_DATE = "2025-07-01"

def compute_indicators(df):
    df["ret"] = df[CLOSE_COL].pct_change()
    df["SMA10"] = df[CLOSE_COL].rolling(10).mean()
    df["SMA20"] = df[CLOSE_COL].rolling(20).mean()
    delta = df[CLOSE_COL].diff()
    up = delta.clip(lower=0).rolling(14).mean()
    down = -delta.clip(upper=0).rolling(14).mean()
    rs = up / (down + 1e-9)
    df["RSI14"] = 100 - (100 / (1 + rs))
    ema12 = df[CLOSE_COL].ewm(span=12, adjust=False).mean()
    ema26 = df[CLOSE_COL].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_sig"] = df["MACD"].ewm(span=9, adjust=False).mean()
    return df

def make_sequences(X, y, lookback):
    xs, ys, idxs = [], [], []
    for i in range(lookback, len(X)):
        xs.append(X[i-lookback:i])
        ys.append(y[i])
        idxs.append(i)
    return np.array(xs), np.array(ys), np.array(idxs)

df = pd.read_csv(CSV_PATH)

df.columns = df.columns.str.strip()

df[DATE_COL] = pd.to_datetime(df[DATE_COL], dayfirst=True)
df.sort_values(DATE_COL, inplace=True)
df.set_index(DATE_COL, inplace=True)

df = compute_indicators(df)
df.dropna(inplace=True)

features = [CLOSE_COL, "ret", "SMA10", "SMA20", "RSI14", "MACD", "MACD_sig"]
for f in features:
    if f not in df.columns:
        raise ValueError(f"Missing feature {f}")

train_df = df[df.index <= pd.Timestamp(SPLIT_DATE)]
overlap_df = train_df.tail(LOOKBACK)
test_df = pd.concat([overlap_df, df[df.index > pd.Timestamp(SPLIT_DATE)]])

X_train_raw = train_df[features].values
y_train = train_df["ret"].shift(-1).dropna().values
X_train_raw = X_train_raw[:-1]
dates_train = train_df.index[:-1]

X_test_raw = test_df[features].values
y_test = test_df["ret"].shift(-1).dropna().values
X_test_raw = X_test_raw[:-1]
dates_test = test_df.index[:-1]

scaler = MinMaxScaler()
ns = X_train_raw.shape[1]
X_train_scaled = scaler.fit_transform(X_train_raw.reshape(-1, ns)).reshape(X_train_raw.shape)
X_test_scaled = scaler.transform(X_test_raw.reshape(-1, ns)).reshape(X_test_raw.shape)

X_tr, y_tr, idx_tr = make_sequences(X_train_scaled, y_train, LOOKBACK)
X_te, y_te, idx_te = make_sequences(X_test_scaled, y_test, LOOKBACK)

idx_te = idx_te.astype(int)
test_dates_seq = dates_test[idx_te]

valid_mask = test_dates_seq >= pd.Timestamp("2025-07-02")
X_te, y_te, test_dates_seq = X_te[valid_mask], y_te[valid_mask], test_dates_seq[valid_mask]

print("Train samples:", X_tr.shape[0], "Test samples:", X_te.shape[0], "features:", X_tr.shape[2])

tf.random.set_seed(RANDOM_SEED)
model = models.Sequential([
    layers.Input(shape=(LOOKBACK, X_tr.shape[2])),
    layers.LSTM(32, return_sequences=True),
    layers.Dropout(0.2),
    layers.LSTM(16),
    layers.Dense(1)
])
model.compile(optimizer="adam", loss="mse")
es = callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

start = time.time()
hist = model.fit(X_tr, y_tr, validation_split=0.12, epochs=EPOCHS, batch_size=BATCH, callbacks=[es], verbose=1)
print("Training time (s):", time.time()-start)

pred_te = model.predict(X_te).ravel()
signals = (pred_te > 0).astype(float)
positions = signals
trades = np.abs(np.diff(np.concatenate([[0.0], positions])))
tc = trades * (TC_BPS/10000.0)
strategy_ret = positions * y_te - tc
cum_curve = np.cumprod(1 + strategy_ret) - 1
bh_ret = np.cumprod(1 + y_te) - 1

def ann_stats(returns):
    if len(returns) == 0:
        return {"cum":np.nan,"ann":np.nan,"vol":np.nan,"sharpe":np.nan,"mdd":np.nan}
    cum = np.prod(1+returns) - 1
    ann = (1 + cum) ** (252/len(returns)) - 1
    vol = np.std(returns) * math.sqrt(252)
    sharpe = ann / (vol + 1e-9)
    nav = np.cumprod(1+returns)
    peak = np.maximum.accumulate(nav)
    dd = (peak - nav) / peak
    mdd = dd.max()
    return {"cum":cum,"ann":ann,"vol":vol,"sharpe":sharpe,"mdd":mdd}

strategy_stats = ann_stats(strategy_ret)
bh_stats = ann_stats(y_te)

print("Strategy stats:", strategy_stats)
print("Buy&Hold stats:", bh_stats)

outdir = Path("results_lstm"); outdir.mkdir(exist_ok=True)
plt.figure(figsize=(10,5))
plt.plot(test_dates_seq, bh_ret, label="Buy & Hold cumulative")
plt.plot(test_dates_seq, cum_curve, label="LSTM-strategy cumulative")
plt.legend(); plt.title("Strategy vs Buy&Hold (test period)")
plt.xticks(rotation=30)
plt.tight_layout(); plt.savefig(outdir/"equity_curve.png"); plt.close()

plt.figure(figsize=(6,4))
plt.scatter(y_te, pred_te, alpha=0.6)
plt.xlabel("Real next-day return"); plt.ylabel("Predicted next-day return")
plt.title("Pred vs Real (test)")
plt.tight_layout(); plt.savefig(outdir/"pred_vs_real.png"); plt.close()

res = pd.DataFrame({
    "date": test_dates_seq,
    "real_ret": y_te,
    "pred_ret": pred_te,
    "signal": positions,
    "strategy_ret": strategy_ret
})
res.to_csv(outdir/"test_predictions.csv", index=False)
print("Saved outputs to", outdir.resolve())
