import numpy as np
import librosa
from scipy.spatial.distance import cdist


def extract_mfcc(audio, sr, n_mfcc=13):
    return librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)


def compute_gram_matrix(features):
    features = features.reshape(features.shape[0], -1)
    gram = features @ features.T
    return gram / features.size


def compute_metrics(content_audio, style_audio, result_audio, sr=24000):
    min_len = min(len(content_audio), len(style_audio), len(result_audio))
    content = content_audio[:min_len]
    style = style_audio[:min_len]
    result = result_audio[:min_len]
    
    mfcc_content = extract_mfcc(content, sr)
    mfcc_result = extract_mfcc(result, sr)
    min_frames = min(mfcc_content.shape[1], mfcc_result.shape[1])
    mfcc_content = mfcc_content[:, :min_frames]
    mfcc_result = mfcc_result[:, :min_frames]
    content_score = np.mean(cdist(mfcc_content.T, mfcc_result.T, metric='euclidean'))
    
    stft_style = np.abs(librosa.stft(style))
    stft_result = np.abs(librosa.stft(result))
    min_frames = min(stft_style.shape[1], stft_result.shape[1])
    stft_style = stft_style[:, :min_frames]
    stft_result = stft_result[:, :min_frames]
    gram_style = compute_gram_matrix(stft_style)
    gram_result = compute_gram_matrix(stft_result)
    style_score = np.mean((gram_style - gram_result) ** 2)

    return {
        'content_preservation': float(content_score),
        'style_transfer': float(style_score)
    }