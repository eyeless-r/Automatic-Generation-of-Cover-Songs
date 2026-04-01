import gin
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

from metrics.metrics import compute_metrics
from diffusion.inference import inference


def compute_dataset_metrics(checkpoint_path, config_path, autoencoder_path, dataset_path, source_separation=False, sr=24000):
    dataset_dir = Path(dataset_path)
    audios = sorted([str(p) for p in dataset_dir.glob("Track*/mix.flac")])
    all_metrics = []
    for i in tqdm(range(len(audios))):
        content_path = audios[i]
        style_path = audios[(i + 1) % len(audios)]
        result = inference(checkpoint_path, config_path, autoencoder_path, content_path, style_path, 
                           source_separation=source_separation)
        metrics = compute_metrics(result['content'], result['style'], result['combined'], sr)
        all_metrics.append(metrics)
    
    average_metrics = {
        'content_preservation': np.mean([metrics['content_preservation'] for metrics in all_metrics]),
        'style_transfer': np.mean([metrics['style_transfer'] for metrics in all_metrics]),
    }
    return average_metrics