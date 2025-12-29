# ==========================================
# Stable High-Quality RVC Gradio Inference
# ==========================================

import torch
import gradio as gr
import librosa
import soundfile as sf
import numpy as np
from scipy.signal import butter, filtfilt
import os

# --------------------
# Device
# --------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------
# High-pass filter
# --------------------
def highpass_filter(audio, sr, cutoff=48):
    b, a = butter(5, cutoff, btype="high", fs=sr)
    return filtfilt(b, a, audio)

# --------------------
# RVC imports
# --------------------
from rvc.infer.infer_pipeline import VC
from rvc.lib.predictors.RMVPE import RMVPE0Predictor
from rvc.lib.predictors.FCPE import FCPEF0Predictor

vc = VC()

rmvpe = RMVPE0Predictor(
    model_path="rmvpe.pt",
    device=DEVICE,
    is_half=(DEVICE == "cuda")
)

# --------------------
# Convert function
# --------------------
def convert(
    audio_path,
    model_file,
    index_file,
    pitch_guidance,
    f0_method,
    transpose,
    index_rate,
    rms_mix
):
    # ---- Safety checks ----
    if audio_path is None:
        raise gr.Error("Please upload input audio")

    if model_file is None:
        raise gr.Error("Please upload a .pth model")

    model_path = model_file.name
    index_path = index_file.name if index_file is not None else None

    # ---- Load audio ----
    audio, sr = librosa.load(audio_path, sr=None, mono=True)
    audio = highpass_filter(audio, sr)

    # ---- F0 predictor ----
    f0_predictor = None
    if pitch_guidance:
        if f0_method == "RMVPE":
            f0_predictor = rmvpe
        else:
            f0_predictor = FCPEF0Predictor(device=DEVICE)

    # ---- Inference ----
    wav = vc.pipeline(
        model_path=model_path,
        index_path=index_path,
        audio=audio,
        sr=sr,
        f0_predictor=f0_predictor,
        f0_up_key=int(transpose),
        index_rate=float(index_rate),
        rms_mix_rate=float(rms_mix),
        use_f0=1 if pitch_guidance else 0
    )

    # ---- Normalize ----
    wav = wav / (np.max(np.abs(wav)) + 1e-6)

    out_path = "converted.wav"
    sf.write(out_path, wav, vc.tgt_sr)

    return out_path

# --------------------
# Gradio UI
# --------------------
with gr.Blocks() as app:
    gr.Markdown("## High-Quality RVC Inference (Stable)")

    audio_in = gr.Audio(type="filepath", label="Input Audio")
    audio_out = gr.Audio(label="Converted Audio")

    model = gr.File(label="Model (.pth)")
    index = gr.File(label="Index (.index)", visible=True)

    pitch_guidance = gr.Checkbox(label="Enable Pitch Guidance", value=True)
    f0_method = gr.Radio(
        ["RMVPE", "CREPE"],
        value="RMVPE",
        label="F0 Method"
    )

    transpose = gr.Slider(-24, 24, value=0, step=1, label="Pitch Transpose")
    index_rate = gr.Slider(0.0, 1.0, value=0.33, step=0.01, label="Index Rate")
    rms_mix = gr.Slider(0.0, 1.0, value=0.25, step=0.01, label="RMS Protection")

    btn = gr.Button("Convert")

    btn.click(
        fn=convert,
        inputs=[
            audio_in,
            model,
            index,
            pitch_guidance,
            f0_method,
            transpose,
            index_rate,
            rms_mix
        ],
        outputs=audio_out
    )

app.launch()
