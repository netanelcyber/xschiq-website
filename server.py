"""
Server-side EEG analysis service with a deep-learning inference path.

Endpoints
---------
- GET /health
- POST /analyze

`POST /analyze` accepts either:
- raw CSV in the request body (samples x channels)
- JSON body: {"samples": [[...], [...], ...]}

This service provides two layers of analysis:
- classical signal statistics + relative bandpower
- self-supervised deep representation learning via a 1D convolutional autoencoder

The deep-learning path is intended for research/demo use and is not a medical device.
"""

from __future__ import annotations

import csv
import io
import json
import math
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

app = FastAPI(title="EEG Deep Analysis Service", version="0.2.0")


BANDS: Dict[str, Tuple[float, float]] = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "beta": (12.0, 30.0),
    "gamma": (30.0, 80.0),
}


class EEGAutoencoder(nn.Module):
    def __init__(self, channels: int, latent_channels: int = 16) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(channels, 32, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv1d(64, latent_channels, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(latent_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv1d(32, channels, kernel_size=7, padding=3),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction, latent


def _parse_csv(file_bytes: bytes) -> np.ndarray:
    text = file_bytes.decode("utf-8", errors="ignore")
    reader = csv.reader(io.StringIO(text))
    rows: List[List[float]] = []
    for row in reader:
        if not row:
            continue
        try:
            rows.append([float(cell) for cell in row])
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"Non-numeric value in row: {row}") from exc
    if not rows:
        raise HTTPException(status_code=400, detail="No data rows found in CSV.")
    return np.asarray(rows, dtype=np.float64)


def _parse_json(file_bytes: bytes) -> np.ndarray:
    try:
        payload = json.loads(file_bytes.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON payload.") from exc

    samples = payload.get("samples", payload) if isinstance(payload, dict) else payload
    matrix = np.asarray(samples, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise HTTPException(status_code=400, detail="JSON body must contain a 2D samples matrix.")
    return matrix


async def _load_matrix_from_request(request: Request) -> np.ndarray:
    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="Request body is empty.")

    content_type = request.headers.get("content-type", "").lower()
    if "application/json" in content_type:
        return _parse_json(body)
    return _parse_csv(body)


def _bandpower(signal: np.ndarray, fs: float) -> Dict[str, float]:
    freqs = np.fft.rfftfreq(signal.size, d=1.0 / fs)
    psd = np.abs(np.fft.rfft(signal)) ** 2

    band_power: Dict[str, float] = {}
    for name, (low, high) in BANDS.items():
        idx = np.logical_and(freqs >= low, freqs <= high)
        power = np.trapz(psd[idx], freqs[idx]) if np.any(idx) else 0.0
        band_power[name] = float(power)

    total_power = float(np.trapz(psd, freqs))
    if total_power > 0:
        for key in band_power:
            band_power[key] = band_power[key] / total_power
    return band_power


def analyze_signal_statistics(matrix: np.ndarray, fs: float) -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {}
    for ch in range(matrix.shape[1]):
        signal = matrix[:, ch]
        results[f"ch_{ch + 1}"] = {
            "mean": float(np.mean(signal)),
            "std": float(np.std(signal)),
            "min": float(np.min(signal)),
            "max": float(np.max(signal)),
            "band_power_fraction": _bandpower(signal, fs),
        }
    return results


def _aligned_window_size(window_size: int) -> int:
    base_size = max(128, int(window_size))
    return ((base_size + 3) // 4) * 4


def _prepare_windows(
    matrix: np.ndarray,
    window_size: int,
    stride: int,
    max_windows: int,
) -> Tuple[np.ndarray, np.ndarray]:
    aligned_window = _aligned_window_size(window_size)
    stride = max(1, int(stride))
    samples, channels = matrix.shape

    if samples < aligned_window:
        pad = aligned_window - samples
        matrix = np.pad(matrix, ((0, pad), (0, 0)), mode="edge")
        samples = matrix.shape[0]

    channel_mean = np.mean(matrix, axis=0, keepdims=True)
    channel_std = np.std(matrix, axis=0, keepdims=True)
    channel_std[channel_std < 1e-6] = 1.0
    normalized = (matrix - channel_mean) / channel_std

    starts = np.arange(0, samples - aligned_window + 1, stride, dtype=np.int32)
    if starts.size == 0:
        starts = np.asarray([0], dtype=np.int32)

    if starts.size > max_windows:
        select_idx = np.linspace(0, starts.size - 1, num=max_windows, dtype=np.int32)
        starts = starts[select_idx]

    windows = np.stack(
        [normalized[start : start + aligned_window].T for start in starts],
        axis=0,
    )
    if windows.shape != (starts.size, channels, aligned_window):
        raise HTTPException(status_code=500, detail="Window preparation produced an unexpected shape.")
    return windows.astype(np.float32), starts


def run_deep_learning_analysis(
    matrix: np.ndarray,
    fs: float,
    *,
    epochs: int = 6,
    window_size: int = 512,
    stride: int = 256,
    max_windows: int = 128,
    batch_size: int = 16,
) -> Dict[str, Any]:
    samples, channels = matrix.shape
    if samples < 32 or channels < 1:
        raise HTTPException(status_code=400, detail="EEG scan is too small for deep analysis.")

    epochs = max(1, min(int(epochs), 20))
    max_windows = max(8, min(int(max_windows), 256))
    batch_size = max(1, min(int(batch_size), 64))

    windows, starts = _prepare_windows(matrix, window_size, stride, max_windows)
    window_tensor = torch.tensor(windows, dtype=torch.float32)
    dataset = TensorDataset(window_tensor)
    loader = DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = EEGAutoencoder(channels=channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    training_curve: List[float] = []
    model.train()
    for _ in range(epochs):
        epoch_loss = 0.0
        sample_count = 0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            reconstruction, _ = model(batch)
            loss = loss_fn(reconstruction, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item()) * batch.shape[0]
            sample_count += int(batch.shape[0])
        training_curve.append(epoch_loss / max(1, sample_count))

    model.eval()
    with torch.no_grad():
        input_tensor = window_tensor.to(device)
        reconstruction, latent = model(input_tensor)
        error_tensor = (reconstruction - input_tensor) ** 2

    reconstruction_errors = error_tensor.mean(dim=(1, 2)).cpu().numpy()
    per_channel_errors = error_tensor.mean(dim=(0, 2)).cpu().numpy()
    latent_embedding = latent.mean(dim=(0, 2)).cpu().numpy()

    threshold = float(np.mean(reconstruction_errors) + 2.0 * np.std(reconstruction_errors))
    anomaly_mask = reconstruction_errors >= threshold
    top_indices = np.argsort(reconstruction_errors)[::-1][: min(5, reconstruction_errors.size)]
    aligned_window = _aligned_window_size(window_size)

    anomalies = [
        {
            "window_index": int(idx),
            "start_sample": int(starts[idx]),
            "end_sample": int(starts[idx] + aligned_window),
            "start_seconds": round(float(starts[idx] / fs), 4),
            "end_seconds": round(float((starts[idx] + aligned_window) / fs), 4),
            "reconstruction_error": round(float(reconstruction_errors[idx]), 8),
        }
        for idx in top_indices
    ]

    total_channel_error = float(np.sum(per_channel_errors))
    if total_channel_error > 0:
        channel_importance = {
            f"ch_{idx + 1}": round(float(err / total_channel_error), 6)
            for idx, err in enumerate(per_channel_errors)
        }
    else:
        channel_importance = {f"ch_{idx + 1}": 0.0 for idx in range(channels)}

    return {
        "analysis_mode": "self_supervised_convolutional_autoencoder",
        "device": device,
        "samples": int(samples),
        "channels": int(channels),
        "sampling_rate_hz": fs,
        "window_size_samples": int(aligned_window),
        "stride_samples": int(max(1, stride)),
        "windows_analyzed": int(reconstruction_errors.size),
        "training_epochs": int(epochs),
        "training_loss_curve": [round(loss, 8) for loss in training_curve],
        "reconstruction_error_mean": round(float(np.mean(reconstruction_errors)), 8),
        "reconstruction_error_std": round(float(np.std(reconstruction_errors)), 8),
        "anomaly_threshold": round(threshold, 8),
        "anomalous_window_fraction": round(float(np.mean(anomaly_mask.astype(np.float32))), 6),
        "top_anomalous_windows": anomalies,
        "channel_importance": channel_importance,
        "scan_embedding_preview": [round(float(value), 6) for value in latent_embedding[:8]],
        "note": "Research-only deep representation analysis. Output is not a clinical diagnosis.",
    }


def analyze_scan(
    matrix: np.ndarray,
    fs: float,
    *,
    epochs: int,
    window_size: int,
    stride: int,
    max_windows: int,
) -> Dict[str, Any]:
    if matrix.ndim != 2:
        raise HTTPException(status_code=400, detail="Expected a 2D matrix of samples x channels.")
    if matrix.shape[0] < 8 or matrix.shape[1] < 1:
        raise HTTPException(status_code=400, detail="Input matrix is too small for EEG analysis.")

    return {
        "sampling_rate_hz": fs,
        "samples": int(matrix.shape[0]),
        "channels": int(matrix.shape[1]),
        "classical_analysis": analyze_signal_statistics(matrix, fs),
        "deep_learning_analysis": run_deep_learning_analysis(
            matrix,
            fs,
            epochs=epochs,
            window_size=window_size,
            stride=stride,
            max_windows=max_windows,
        ),
    }


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok", "service": "eeg-deep-analysis"})


@app.post("/analyze")
async def analyze(
    request: Request,
    fs: float = 256.0,
    epochs: int = 6,
    window_size: int = 512,
    stride: int = 256,
    max_windows: int = 128,
) -> JSONResponse:
    if fs <= 0 or not math.isfinite(fs):
        raise HTTPException(status_code=400, detail="Sampling rate `fs` must be positive.")

    matrix = await _load_matrix_from_request(request)
    analysis = analyze_scan(
        matrix,
        fs,
        epochs=epochs,
        window_size=window_size,
        stride=stride,
        max_windows=max_windows,
    )
    return JSONResponse(analysis)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
