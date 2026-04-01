"""
Server-side EEG analysis and article generation service.

Endpoints
---------
- GET /health
- POST /analyze
- POST /generate-article

Accepted request formats
------------------------
- raw CSV in the request body (samples x channels)
- JSON body with ``samples`` plus optional text fields
- multipart/form-data with:
  - ``eeg``: CSV or JSON file
  - ``report``: TXT / JSON / CSV / PDF diagnostic report
  - ``previous_articles``: one or more prior article files
  - ``title_hint`` / ``study_focus``: optional text fields
"""

from __future__ import annotations

import csv
import io
import json
import math
import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

app = FastAPI(title="EEG Deep Analysis Service", version="0.3.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


BANDS: Dict[str, Tuple[float, float]] = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "beta": (12.0, 30.0),
    "gamma": (30.0, 80.0),
}

STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "into", "were", "was", "are",
    "our", "their", "using", "used", "over", "than", "after", "before", "within",
    "through", "during", "between", "suggest", "suggests", "study", "analysis",
    "patient", "patients", "signal", "signals", "results", "result", "section",
    "clinical", "research", "data", "model", "deep", "learning", "eeg", "scan",
    "theory", "background", "report", "article",
    "של", "עם", "על", "זה", "זאת", "הוא", "היא", "הם", "הן", "גם", "עוד", "לא",
    "כן", "דרך", "כדי", "אחד", "אחת", "שיש", "שהוא", "שהיא", "אלה", "אלו", "מתוך",
    "לאחר", "לפני", "במהלך", "ניתוח", "מחקר", "קליני", "מחקרים", "מטופל", "מטופלים",
    "תוצאה", "תוצאות", "מערכת", "המערכת", "מאמר", "מאמרים", "סריקה", "סריקות",
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

def serialize_model(model: EEGAutoencoder) -> Dict[str, Any]:
    return {k: v.detach().cpu().numpy().tolist() for k, v in model.state_dict().items()}

def train_model(matrix: np.ndarray, fs: float, epochs: int = 6, window_size: int = 512, stride: int = 256, max_windows: int = 128, batch_size: int = 16) -> Tuple[Dict[str, Any], str, List[float]]:
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

    return serialize_model(model), device, training_curve


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


def _matrix_from_payload(payload: Any) -> np.ndarray:
    samples = payload.get("samples", payload) if isinstance(payload, dict) else payload
    matrix = np.asarray(samples, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise HTTPException(status_code=400, detail="JSON body must contain a 2D samples matrix.")
    return matrix


def _parse_json_matrix(file_bytes: bytes) -> np.ndarray:
    try:
        payload = json.loads(file_bytes.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON payload.") from exc
    return _matrix_from_payload(payload)


def _extract_boundary(content_type: str) -> bytes:
    match = re.search(r'boundary="?([^";]+)"?', content_type, flags=re.IGNORECASE)
    if not match:
        raise HTTPException(status_code=400, detail="Multipart request is missing a boundary.")
    return match.group(1).encode("utf-8")


def _parse_header_parameters(header_value: str) -> Dict[str, str]:
    params: Dict[str, str] = {}
    for part in header_value.split(";"):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        params[key.strip().lower()] = value.strip().strip('"')
    return params


def _parse_multipart_parts(body: bytes, content_type: str) -> Dict[str, List[Dict[str, Any]]]:
    boundary = b"--" + _extract_boundary(content_type)
    parts: Dict[str, List[Dict[str, Any]]] = {}

    for chunk in body.split(boundary):
        chunk = chunk.strip(b"\r\n")
        if not chunk or chunk == b"--":
            continue
        if b"\r\n\r\n" not in chunk:
            continue

        header_blob, content = chunk.split(b"\r\n\r\n", 1)
        content = content.rstrip(b"\r\n")
        header_lines = header_blob.decode("latin-1", errors="ignore").split("\r\n")
        headers: Dict[str, str] = {}
        for line in header_lines:
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            headers[key.strip().lower()] = value.strip()

        disposition = headers.get("content-disposition", "")
        disposition_params = _parse_header_parameters(disposition)
        field_name = disposition_params.get("name")
        if not field_name:
            continue

        part = {
            "name": field_name,
            "filename": disposition_params.get("filename", ""),
            "content_type": headers.get("content-type", "application/octet-stream"),
            "content": content,
        }
        parts.setdefault(field_name, []).append(part)

    return parts


def _decode_text_part(part: Dict[str, Any]) -> str:
    content = part.get("content", b"")
    if not isinstance(content, bytes):
        return ""
    return content.decode("utf-8", errors="ignore").strip()


def _clean_text(text: str, *, max_chars: int = 18000) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    text = text.strip()
    return text[:max_chars]


def _best_effort_pdf_text(file_bytes: bytes) -> str:
    raw = file_bytes.decode("latin-1", errors="ignore")
    candidates = re.findall(r"\(([^()]{4,240})\)", raw)
    fragments: List[str] = []

    for candidate in candidates:
        candidate = candidate.replace("\\n", " ").replace("\\r", " ")
        candidate = candidate.replace("\\(", "(").replace("\\)", ")")
        cleaned = _clean_text(candidate, max_chars=240)
        if len(cleaned) >= 4:
            fragments.append(cleaned)

    if not fragments:
        ascii_runs = re.findall(r"[A-Za-z0-9\u0590-\u05FF ,.;:()/%-]{12,}", raw)
        fragments = [_clean_text(item, max_chars=240) for item in ascii_runs if item.strip()]

    return _clean_text(" ".join(fragments[:80]))


def _extract_text_from_part(part: Dict[str, Any]) -> str:
    filename = str(part.get("filename", "")).lower()
    content_type = str(part.get("content_type", "")).lower()
    raw = part.get("content", b"")
    if not isinstance(raw, bytes):
        return ""

    if filename.endswith(".pdf") or "pdf" in content_type:
        extracted = _best_effort_pdf_text(raw)
        return extracted or "PDF uploaded. Text extraction was limited for this document."

    if filename.endswith(".json") or "application/json" in content_type:
        try:
            payload = json.loads(raw.decode("utf-8"))
            if isinstance(payload, (dict, list)):
                return _clean_text(json.dumps(payload, ensure_ascii=False))
        except json.JSONDecodeError:
            pass

    try:
        return _clean_text(raw.decode("utf-8"))
    except UnicodeDecodeError:
        return _clean_text(raw.decode("latin-1", errors="ignore"))


def _normalize_previous_articles(payload: Any) -> List[Dict[str, str]]:
    normalized: List[Dict[str, str]] = []
    if not payload:
        return normalized

    items = payload if isinstance(payload, list) else [payload]
    for index, item in enumerate(items, start=1):
        if isinstance(item, dict):
            text = _clean_text(str(item.get("text", "")))
            name = str(item.get("name", f"prior_article_{index}"))
        else:
            text = _clean_text(str(item))
            name = f"prior_article_{index}"
        if text:
            normalized.append({"name": name, "text": text})

    return normalized


async def _load_request_inputs(request: Request) -> Dict[str, Any]:
    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="Request body is empty.")

    content_type = request.headers.get("content-type", "").lower()
    matrix: np.ndarray | None = None
    report_text = ""
    previous_articles: List[Dict[str, str]] = []
    title_hint = ""
    study_focus = ""

    if "multipart/form-data" in content_type:
        parts = _parse_multipart_parts(body, content_type)

        eeg_part = (parts.get("eeg") or [None])[0]
        if eeg_part:
            filename = str(eeg_part.get("filename", "")).lower()
            part_content_type = str(eeg_part.get("content_type", "")).lower()
            if filename.endswith(".json") or "application/json" in part_content_type:
                matrix = _parse_json_matrix(eeg_part["content"])
            else:
                matrix = _parse_csv(eeg_part["content"])

        report_part = (parts.get("report") or [None])[0]
        if report_part:
            report_text = _extract_text_from_part(report_part)

        article_parts = (
            parts.get("previous_articles", [])
            + parts.get("previous_article", [])
            + parts.get("article", [])
        )
        for index, part in enumerate(article_parts, start=1):
            text = _extract_text_from_part(part)
            if text:
                previous_articles.append(
                    {
                        "name": str(part.get("filename") or f"prior_article_{index}"),
                        "text": text,
                    }
                )

        title_part = (parts.get("title_hint") or [None])[0]
        if title_part:
            title_hint = _decode_text_part(title_part)

        focus_part = (parts.get("study_focus") or [None])[0]
        if focus_part:
            study_focus = _decode_text_part(focus_part)

    elif "application/json" in content_type:
        try:
            payload = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="Invalid JSON payload.") from exc

        if isinstance(payload, (dict, list)) and (
            (isinstance(payload, dict) and "samples" in payload) or isinstance(payload, list)
        ):
            matrix = _matrix_from_payload(payload)

        if isinstance(payload, dict):
            report_text = _clean_text(str(payload.get("report_text", payload.get("report", ""))))
            previous_articles = _normalize_previous_articles(
                payload.get("previous_articles", payload.get("articles", []))
            )
            title_hint = _clean_text(str(payload.get("title_hint", "")), max_chars=160)
            study_focus = _clean_text(str(payload.get("study_focus", "")), max_chars=220)
    else:
        matrix = _parse_csv(body)

    return {
        "matrix": matrix,
        "report_text": report_text,
        "previous_articles": previous_articles,
        "title_hint": title_hint,
        "study_focus": study_focus,
    }


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


def _tokenize(text: str) -> List[str]:
    return [
        token.lower()
        for token in re.findall(r"[A-Za-z\u0590-\u05FF][A-Za-z0-9_\-\u0590-\u05FF]{2,}", text.lower())
        if token.lower() not in STOPWORDS
    ]


def _top_keywords(texts: Iterable[str], limit: int = 10) -> List[str]:
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(_tokenize(text))
    return [token for token, _ in counts.most_common(limit)]


def _aggregate_band_profile(classical_analysis: Dict[str, Dict[str, Any]]) -> List[Tuple[str, float]]:
    totals = {band: 0.0 for band in BANDS}
    channel_count = 0

    for channel in classical_analysis.values():
        channel_count += 1
        band_fractions = channel.get("band_power_fraction", {})
        for band, value in band_fractions.items():
            totals[band] += float(value)

    if channel_count == 0:
        return [(band, 0.0) for band in BANDS]

    ranked = [(band, totals[band] / channel_count) for band in totals]
    return sorted(ranked, key=lambda item: item[1], reverse=True)


def _sentenceify(text: str) -> List[str]:
    raw_sentences = re.split(r"(?<=[\.\!\?\n])\s+", text)
    return [_clean_text(sentence, max_chars=320) for sentence in raw_sentences if _clean_text(sentence, max_chars=320)]


def _rank_context_snippets(
    report_text: str,
    previous_articles: List[Dict[str, str]],
    query_terms: List[str],
    *,
    limit: int = 5,
) -> List[Dict[str, Any]]:
    snippets: List[Tuple[float, str, str]] = []
    query_set = set(query_terms)

    sources: List[Tuple[str, str]] = []
    if report_text:
        sources.append(("diagnostic_report", report_text))
    for article in previous_articles:
        sources.append((article["name"], article["text"]))

    for source_name, source_text in sources:
        for sentence in _sentenceify(source_text):
            tokens = set(_tokenize(sentence))
            if not tokens:
                continue
            overlap = len(tokens & query_set)
            density = min(len(tokens), 12) / 12.0
            score = overlap * 1.6 + density
            if source_name == "diagnostic_report":
                score += 0.4
            snippets.append((score, source_name, sentence))

    snippets.sort(key=lambda item: item[0], reverse=True)
    results: List[Dict[str, Any]] = []
    seen = set()
    for score, source_name, sentence in snippets:
        key = (source_name, sentence)
        if key in seen:
            continue
        seen.add(key)
        results.append({"source": source_name, "score": round(float(score), 3), "excerpt": sentence})
        if len(results) >= limit:
            break
    return results


def _deep_profile(analysis: Dict[str, Any] | None) -> Dict[str, Any]:
    if not analysis:
        return {
            "top_bands": [],
            "top_channels": [],
            "anomaly_fraction": None,
            "severity_label": "text_context_only",
            "analysis_mode": "text_context_only",
            "windows_analyzed": 0,
            "top_anomalous_windows": [],
            "embedding_preview": [],
        }

    band_profile = _aggregate_band_profile(analysis["classical_analysis"])
    deep = analysis["deep_learning_analysis"]
    anomaly_fraction = float(deep["anomalous_window_fraction"])

    if anomaly_fraction >= 0.35:
        severity = "high_anomaly_load"
    elif anomaly_fraction >= 0.15:
        severity = "moderate_anomaly_load"
    else:
        severity = "low_anomaly_load"

    top_channels = sorted(
        deep["channel_importance"].items(),
        key=lambda item: item[1],
        reverse=True,
    )[:3]

    return {
        "top_bands": [band for band, _ in band_profile[:2]],
        "top_channels": [channel for channel, _ in top_channels],
        "anomaly_fraction": anomaly_fraction,
        "severity_label": severity,
        "analysis_mode": deep["analysis_mode"],
        "windows_analyzed": deep["windows_analyzed"],
        "top_anomalous_windows": deep["top_anomalous_windows"][:3],
        "embedding_preview": deep["scan_embedding_preview"][:6],
    }


def _build_article_markdown(
    title: str,
    sections: Dict[str, str],
    snippets: List[Dict[str, Any]],
    keywords: List[str],
) -> str:
    snippet_block = "\n".join(
        f"- {item['source']}: {item['excerpt']}"
        for item in snippets
    ) or "- No previous article snippets were retrieved."

    return "\n\n".join(
        [
            f"# {title}",
            f"**Keywords:** {', '.join(keywords) if keywords else 'EEG, deep learning, diagnostic synthesis'}",
            f"## Abstract\n{sections['abstract']}",
            f"## Introduction\n{sections['introduction']}",
            f"## Materials and Methods\n{sections['methods']}",
            f"## Results\n{sections['results']}",
            f"## Discussion\n{sections['discussion']}",
            f"## Conclusion\n{sections['conclusion']}",
            f"## Retrieved Context\n{snippet_block}",
        ]
    )


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "mode": "server-train-only"}


@app.post("/train")
async def train_endpoint(request: Request) -> Dict[str, Any]:
    inputs = await _load_request_inputs(request)
    matrix = inputs.get("matrix")
    if matrix is None:
        raise HTTPException(status_code=400, detail="No EEG data provided for training.")

    fs = 250.0
    if request.query_params.get("sampling_rate_hz"):
        try:
            fs = float(request.query_params.get("sampling_rate_hz", "250"))
        except ValueError:
            raise HTTPException(status_code=400, detail="sampling_rate_hz must be numeric")

    model_state, device, training_curve = train_model(matrix, fs=fs)

    return {
        "message": "training_complete",
        "device": device,
        "samples": int(matrix.shape[0]),
        "channels": int(matrix.shape[1]),
        "sampling_rate_hz": fs,
        "training_curve": training_curve,
        "model_state": model_state,
    }

