"""
evaluate.py — 模型评估与可视化
================================
- 加载测试集和最佳模型，输出 MAE / RMSE / R2
- 随机选取一个城市，画真实 vs 预测温度时间序列对比图
- 画预测误差分布直方图
- 多模型性能对比表格

用法:
    python evaluate.py                         (评估 config 中的模型)
    python evaluate.py --model lstm1           (指定模型)
    python evaluate.py --models lstm1,lstm2,transformer  (多模型对比)
    python evaluate.py --all                   (自动发现 checkpoints/ 中所有模型)
"""

import argparse
import os
import pickle
import sys
from glob import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from models import build_model


def load_config():
    cfg_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def inverse_tavg(y_scaled, city_arr, scalers, n_features):
    out = np.zeros_like(y_scaled)
    for i, c in enumerate(city_arr):
        dummy = np.zeros((1, n_features))
        dummy[0, 0] = y_scaled[i]
        out[i] = scalers[c].inverse_transform(dummy)[0, 0]
    return out


def plot_time_series(y_true, y_pred, city_name, model_name, plot_dir):
    n = len(y_true)
    x = np.arange(n)
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(x, y_true, "k-", lw=0.7, alpha=0.7, label="True")
    ax.plot(x, y_pred, "r-", lw=0.7, alpha=0.7, label="Pred")
    ax.fill_between(x, y_true, y_pred, alpha=0.15, color="gray")
    ax.set_title(f"{city_name} | {model_name}: True vs Predicted Temperature (Test Period)")
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Temperature (degC)")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(plot_dir, f"{model_name}_{city_name}_timeseries.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  [plot] {path}")


def plot_error_histogram(errors, model_name, plot_dir):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(errors, bins=50, color="steelblue", edgecolor="white", alpha=0.85)
    ax.axvline(0, color="red", linestyle="--", lw=1)
    m, s = np.mean(errors), np.std(errors)
    ax.set_title(f"Prediction Error Distribution | {model_name}\nmean={m:.3f} degC, std={s:.3f} degC")
    ax.set_xlabel("Error (degC, true - pred)")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    path = os.path.join(plot_dir, f"{model_name}_error_hist.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  [plot] {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--models", type=str, default=None)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    cfg = load_config()
    proj_root = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(proj_root, "data")
    ckpt_dir = os.path.join(proj_root, "checkpoints")
    plot_dir = os.path.join(proj_root, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    if args.all:
        model_names = sorted(set(
            os.path.splitext(os.path.basename(p))[0].replace("_best", "")
            for p in glob(os.path.join(ckpt_dir, "*_best.pt"))
        ))
    elif args.models:
        model_names = [m.strip() for m in args.models.split(",")]
    elif args.model:
        model_names = [args.model]
    else:
        model_names = [cfg["model"]["name"]]

    print(f"Models to evaluate: {model_names}")

    test = np.load(os.path.join(data_dir, "test.npz"))
    X_test, y_test, city_test = test["X"], test["y"], test["city"]
    n_features = X_test.shape[2]
    scalers = pickle.load(open(os.path.join(data_dir, "scalers.pkl"), "rb"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    y_true_orig = inverse_tavg(y_test, city_test, scalers, n_features)
    all_metrics = {}

    for mname in model_names:
        if mname == "ensemble":
            from models import build_ensemble
            model = build_ensemble(cfg, ckpt_dir)
            model.eval()
        else:
            print(f"\n{'='*60}\n  Evaluating: {mname}\n{'='*60}")
            cfg_copy = cfg.copy()
            cfg_copy["model"] = {**cfg["model"], "name": mname}
            model = build_model(cfg_copy).to(device)
            ckpt_path = os.path.join(ckpt_dir, f"{mname}_best.pt")
            if not os.path.exists(ckpt_path):
                print(f"  [SKIP] Checkpoint not found: {ckpt_path}")
                continue
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            model.eval()

        with torch.no_grad():
            preds = model(torch.from_numpy(X_test).to(device)).cpu().numpy()
        y_pred_orig = inverse_tavg(preds, city_test, scalers, n_features)

        mae = mean_absolute_error(y_true_orig, y_pred_orig)
        rmse = np.sqrt(mean_squared_error(y_true_orig, y_pred_orig))
        r2 = r2_score(y_true_orig, y_pred_orig)
        print(f"  MAE:  {mae:.4f} degC")
        print(f"  RMSE: {rmse:.4f} degC")
        print(f"  R2:   {r2:.4f}")
        all_metrics[mname] = {"MAE": mae, "RMSE": rmse, "R2": r2}

        # 随机选城市画图
        unique_cities = sorted(set(city_test))
        city = unique_cities[np.random.default_rng(42).integers(len(unique_cities))]
        mask = city_test == city
        plot_time_series(y_true_orig[mask], y_pred_orig[mask], city, mname, plot_dir)

        # 误差直方图
        errors = y_true_orig - y_pred_orig
        plot_error_histogram(errors, mname, plot_dir)

    if len(all_metrics) > 1:
        print(f"\n{'='*60}\n  Model Comparison\n{'='*60}")
        hdr = f"{'Model':<14} {'MAE(degC)':>10} {'RMSE(degC)':>10} {'R2':>10}"
        print(hdr)
        print("-" * len(hdr))
        best_r2 = max(m["R2"] for m in all_metrics.values())
        for name, m in all_metrics.items():
            star = " *" if abs(m["R2"] - best_r2) < 1e-8 else ""
            print(f"{name:<14} {m['MAE']:>10.4f} {m['RMSE']:>10.4f} {m['R2']:>10.4f}{star}")
        print(f"\n  * best R2")

    print(f"\n  Plots saved to: {plot_dir}/")


if __name__ == "__main__":
    main()
