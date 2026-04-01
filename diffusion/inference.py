import gin
import torch
import numpy as np
import librosa
import sys
import os

from demucs.pretrained import get_model
from demucs.apply import apply_model

from diffusion.model import EDM_ADV, EDM_ADV_SS

def load_audio(path, sr, audio_length):
    audio, sr_orig = librosa.load(path, sr=sr, mono=True)
    if sr_orig != sr:
        audio = librosa.resample(audio, orig_sr=sr_orig, target_sr=SR)
    audio = audio[:audio_length]
    audio = torch.from_numpy(audio).float()
    audio = audio / audio.abs().max()
    return audio

def process_audio(model, emb_model, audio, device):
    wav = audio.to(device)
    wav = wav.reshape(1, 1, -1)
    z = emb_model.encode(wav)
    cqt = model.time_transform(wav)
    cqt = torch.nn.functional.interpolate(cqt,
                                          size=(z.shape[-1]),
                                          mode="nearest")
    cqt = (cqt - torch.min(cqt)) / (torch.max(cqt) - torch.min(cqt) + 1e-4)
    return z, cqt

def separate_process_audio(model, emb_model, demucs, audio, device):
    wav = audio.to(device)
    wav = torch.stack([wav, wav], axis=0)
    wav = wav.unsqueeze(0)
    with torch.no_grad():
        stems = apply_model(demucs, wav, device=device, progress=False)
    zs = []
    cqts = []
    for i in range(6):
        stem = stems[0, i, 0, :]
        z, cqt = process_audio(model, emb_model, stem, device)
        zs.append(z)
        cqts.append(cqt)
    return zs, cqts

def style_transfer(model, emb_model, shape, cqt_content, z_style, device, nb_steps=40, guidance=2.0, source_separation=False):
    if source_separation:
        time_cond = [model.encoders_time[i](cqt) for i, cqt in enumerate(cqt_content)]
        zsem = [model.encoders[i](z) for i, z in enumerate(z_style)]
    else:
        time_cond = model.encoder_time(cqt_content)
        zsem = model.encoder(z_style)
    x0 = torch.randn(shape).to(device)
    with torch.no_grad():
        xS = model.sample(x0,
                          time_cond=time_cond,
                          zsem=zsem,
                          nb_step=nb_steps,
                          guidance=guidance,
                          guidance_type="time_cond",
                          verbose=False)
    audio = emb_model.decode(xS).cpu().numpy().squeeze()
    audio = audio / np.abs(audio).max()
    return audio

def init_models(checkpoint_path, config_path, autoencoder_path, device, source_separation=False):
    torch.set_grad_enabled(False)
    gin.parse_config_file(config_path)
    if source_separation:
        blender = EDM_ADV_SS()
    else:
        blender = EDM_ADV()
    state_dict = torch.load(checkpoint_path, map_location=device)["model_state"]
    blender.load_state_dict(state_dict, strict=False)
    emb_model = torch.jit.load(autoencoder_path).eval().to(device)
    blender = blender.eval().to(device)
    return blender, emb_model

def inference(checkpoint_path, config_path, autoencoder_path, content_path, style_path,
               source_separation=False, nb_steps=40, guidance=2.0):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    blender, emb_model = init_models(checkpoint_path, config_path, autoencoder_path, 
                                    device, source_separation=source_separation)
    SR = gin.query_parameter("%SR")
    audio_length = gin.query_parameter("%X_LENGTH") * 3
    if source_separation:
        demucs = get_model('htdemucs_6s')
        demucs = demucs.to(device)
        demucs.eval()
    content_audio = load_audio(content_path, sr=SR, audio_length=audio_length)
    style_audio = load_audio(style_path, sr=SR, audio_length=audio_length)
    z1, cqt1 = process_audio(blender, emb_model, content_audio, device)
    z2, cqt2 = process_audio(blender, emb_model, style_audio, device)
    if source_separation:
        zs1, cqts1 = separate_process_audio(blender, emb_model, demucs, content_audio, device)
        zs2, cqts2 = separate_process_audio(blender, emb_model, demucs, style_audio, device)
        combined = style_transfer(blender, emb_model, z1.shape, [cqt1] + cqts1, [z2] + zs2, 
                                  device, nb_steps, guidance, source_separation=True)
    else:
        combined = style_transfer(blender, emb_model, z1.shape, cqt1, z2, device, nb_steps, guidance)
    return {
        'content': content_audio.numpy(),
        'style': style_audio.numpy(),
        'combined': combined
    }
