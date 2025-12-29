import os
import torch
import gradio as gr
import librosa
import soundfile as sf
import numpy as np
from scipy.signal import butter, filtfilt

# =====================
# Device
# =====================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =====================
# High-pass filter
# =====================
def highpass_filter(audio, sr, cutoff=48):
    b, a = butter(5, cutoff, btype="high", fs=sr)
    return filtfilt(b, a, audio)

# =====================
# Load RVC pipeline
# =====================
from rvc.infer.infer_pipeline import VC
from rvc.lib.predictors.RMVPE import RMVPE0Predictor
from rvc.lib.predictors.FCPE import FCPEF0Predictor

vc = VC()

rmvpe = RMVPE0Predictor(
    model_path="rmvpe.pt",
    device=DEVICE,
    is_half=(DEVICE == "cuda")
)

# =====================
# Conversion function
# =====================
def convert(
    audio_path,
    model_path,
    index_path,
    pitch_guidance,
    f0_method,
    transpose,
    index_rate,
    rms_mix
):
    if audio_path is None or model_path is None:
        return None

    audio, sr = librosa.load(audio_path, sr=None, mono=True)
    audio = highpass_filter(audio, sr)

    f0_predictor = None
    if pitch_guidance:
        if f0_method == "RMVPE":
            f0_predictor = rmvpe
        else:
            f0_predictor = FCPEF0Predictor(device=DEVICE)

    wav = vc.pipeline(
        model_path=model_path,
        index_path=index_path,
        audio=audio,
        sr=sr,
        f0_predictor=f0_predictor,
        f0_up_key=transpose,
        index_rate=index_rate,
        rms_mix_rate=rms_mix,
        use_f0=1 if pitch_guidance else 0
    )

    wav = wav / (np.max(np.abs(wav)) + 1e-6)
    sf.write("converted.wav", wav, vc.tgt_sr)

    return "converted.wav"

# =====================
# Gradio UI
# =====================
with gr.Blocks() as app:
    gr.Markdown("## High Quality RVC Inference (Gradio)")

    audio_in = gr.Audio(label="Input Audio", type="filepath")
    audio_out = gr.Audio(label="Converted Audio")

    model = gr.File(label="Model (.pth)")
    index = gr.File(label="Index (.index)", optional=True)

    pitch_guidance = gr.Checkbox(label="Enable Pitch Guidance", value=True)
    f0_method = gr.Radio(["RMVPE", "CREPE"], value="RMVPE", label="F0 Method")

    transpose = gr.Slider(-24, 24, value=0, step=1, label="Pitch Transpose")
    index_rate = gr.Slider(0, 1, value=0.33, step=0.01, label="Index Rate")
    rms_mix = gr.Slider(0, 1, value=0.25, step=0.01, label="RMS Mix")

    btn = gr.Button("Convert")

    btn.click(
        convert,
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
