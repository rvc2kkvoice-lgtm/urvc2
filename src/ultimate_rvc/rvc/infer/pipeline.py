import pathlib
import sys
import logging

import numpy as np
from scipy import signal

import faiss
import torch
import torch.nn.functional as F
import librosa

from ultimate_rvc.rvc.lib.predictors.f0 import CREPE, FCPE, RMVPE

now_dir = pathlib.Path.cwd()
sys.path.append(str(now_dir))

logger = logging.getLogger(__name__)

# =========================
# Audio filter
# =========================
FILTER_ORDER = 5
CUTOFF_FREQUENCY = 48
SAMPLE_RATE = 16000

bh, ah = signal.butter(
    N=FILTER_ORDER,
    Wn=CUTOFF_FREQUENCY,
    btype="high",
    fs=SAMPLE_RATE,
)


# =========================
# RMS processor
# =========================
class AudioProcessor:
    def change_rms(
        source_audio: np.ndarray,
        source_rate: int,
        target_audio: np.ndarray,
        target_rate: int,
        rate: float,
    ) -> np.ndarray:
        rms1 = librosa.feature.rms(
            y=source_audio,
            frame_length=source_rate // 2 * 2,
            hop_length=source_rate // 2,
        )
        rms2 = librosa.feature.rms(
            y=target_audio,
            frame_length=target_rate // 2 * 2,
            hop_length=target_rate // 2,
        )

        rms1 = F.interpolate(
            torch.from_numpy(rms1).float().unsqueeze(0),
            size=target_audio.shape[0],
            mode="linear",
        ).squeeze()

        rms2 = F.interpolate(
            torch.from_numpy(rms2).float().unsqueeze(0),
            size=target_audio.shape[0],
            mode="linear",
        ).squeeze()

        rms2 = torch.maximum(rms2, torch.zeros_like(rms2) + 1e-6)

        return (
            target_audio
            * (torch.pow(rms1, 1 - rate) * torch.pow(rms2, rate - 1)).numpy()
        )


# =========================
# Autotune helper
# =========================
class Autotune:
    def __init__(self):
        self.note_dict = [
            49.00, 51.91, 55.00, 58.27, 61.74, 65.41, 69.30, 73.42,
            77.78, 82.41, 87.31, 92.50, 98.00, 103.83, 110.00,
            116.54, 123.47, 130.81, 138.59, 146.83, 155.56,
            164.81, 174.61, 185.00, 196.00, 207.65, 220.00,
            233.08, 246.94, 261.63, 277.18, 293.66, 311.13,
            329.63, 349.23, 369.99, 392.00, 415.30, 440.00,
            466.16, 493.88, 523.25, 554.37, 587.33, 622.25,
            659.25, 698.46, 739.99, 783.99, 830.61, 880.00,
            932.33, 987.77, 1046.50,
        ]

    def autotune_f0(self, f0, strength):
        out = np.zeros_like(f0)
        for i, freq in enumerate(f0):
            note = min(self.note_dict, key=lambda x: abs(x - freq))
            out[i] = freq + (note - freq) * strength
        return out


# =========================
# Pipeline
# =========================
class Pipeline:
    def __init__(self, tgt_sr, config):
        self.x_pad = config.x_pad
        self.x_query = config.x_query
        self.x_center = config.x_center
        self.x_max = config.x_max

        self.sample_rate = 16000
        self.tgt_sr = tgt_sr
        self.window = 160

        self.t_pad = self.sample_rate * self.x_pad
        self.t_pad_tgt = tgt_sr * self.x_pad
        self.t_pad2 = self.t_pad * 2
        self.t_query = self.sample_rate * self.x_query
        self.t_center = self.sample_rate * self.x_center
        self.t_max = self.sample_rate * self.x_max

        self.f0_min = 50
        self.f0_max = 1100
        self.f0_mel_min = 1127 * np.log(1 + self.f0_min / 700)
        self.f0_mel_max = 1127 * np.log(1 + self.f0_max / 700)

        self.device = config.device
        self.autotune = Autotune()

    # =========================
    # F0 extraction
    # =========================
    def get_f0(
        self,
        x,
        p_len,
        f0_method,
        pitch,
        f0_autotune,
        f0_autotune_strength,
        proposed_pitch,
        proposed_pitch_threshold,
    ):
        if f0_method == "crepe":
            model = CREPE(
                device=self.device,
                sample_rate=self.sample_rate,
                hop_size=self.window,
            )
            f0 = model.get_f0(x, self.f0_min, self.f0_max, p_len, "full")

        elif f0_method == "crepe-tiny":
            model = CREPE(
                device=self.device,
                sample_rate=self.sample_rate,
                hop_size=self.window,
            )
            f0 = model.get_f0(x, self.f0_min, self.f0_max, p_len, "tiny")

        elif f0_method == "rmvpe":
            model = RMVPE(
                device=self.device,
                sample_rate=self.sample_rate,
                hop_size=self.window,
            )
            f0 = model.get_f0(x, filter_radius=0.03)

        else:
            model = FCPE(
                device=self.device,
                sample_rate=self.sample_rate,
                hop_size=self.window,
            )
            f0 = model.get_f0(x, p_len, filter_radius=0.006)

        if f0_autotune:
            f0 = self.autotune.autotune_f0(f0, f0_autotune_strength)

        f0 *= pow(2, pitch / 12)

        f0bak = f0.copy()
        mel = 1127 * np.log(1 + f0 / 700)
        mel[mel > 0] = (mel[mel > 0] - self.f0_mel_min) * 254 / (
            self.f0_mel_max - self.f0_mel_min
        ) + 1
        mel[mel <= 1] = 1
        mel[mel > 255] = 255

        return np.rint(mel).astype(int), f0bak

    # =========================
    # Voice conversion
    # =========================
    def voice_conversion(
        self,
        model,
        net_g,
        sid,
        audio0,
        pitch,
        pitchf,
        index,
        big_npy,
        index_rate,
        version,
        protect,
    ):
        with torch.no_grad():
            # 🔒 FORCE pitch guidance
            protect = min(protect, 0.49)

            feats = torch.from_numpy(audio0).float()
            feats = feats.mean(-1) if feats.dim() == 2 else feats
            feats = feats.view(1, -1).to(self.device)

            feats = model(feats)["last_hidden_state"]
            feats = model.final_proj(feats[0]).unsqueeze(0) if version == "v1" else feats
            feats0 = feats.clone()

            if index is not None:
                npy = feats[0].cpu().numpy()
                score, ix = index.search(npy, k=8)
                score = np.maximum(score, 1e-6)
                weight = np.square(1 / score)
                weight /= weight.sum(axis=1, keepdims=True)
                feats = (
                    torch.from_numpy(
                        np.sum(big_npy[ix] * weight[..., None], axis=1)
                    )
                    .unsqueeze(0)
                    .to(self.device)
                    * index_rate
                    + feats * (1 - index_rate)
                )

            feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
            feats0 = F.interpolate(feats0.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)

            p_len = min(audio0.shape[0] // self.window, feats.shape[1])
            pitch, pitchf = pitch[:, :p_len], pitchf[:, :p_len]

            mask = pitchf.clone()
            mask[pitchf > 0] = 1
            mask[pitchf < 1] = protect

            feats = feats * mask.unsqueeze(-1) + feats0 * (1 - mask.unsqueeze(-1))

            p_len = torch.tensor([p_len], device=self.device)
            audio = net_g.infer(feats.float(), p_len, pitch, pitchf.float(), sid)[0][0, 0]

            return audio.cpu().numpy()

    # =========================
    # Main pipeline
    # =========================
    def pipeline(
        self,
        model,
        net_g,
        sid,
        audio,
        pitch,
        f0_method,
        file_index,
        index_rate,
        pitch_guidance,
        volume_envelope,
        version,
        protect,
        f0_autotune,
        f0_autotune_strength,
        proposed_pitch,
        proposed_pitch_threshold,
    ):
        # 🔒 FORCE pitch guidance permanently
        pitch_guidance = True

        if file_index and pathlib.Path(file_index).exists() and index_rate > 0:
            index = faiss.read_index(file_index)
            big_npy = index.reconstruct_n(0, index.ntotal)
        else:
            index = big_npy = None

        audio = signal.filtfilt(bh, ah, audio)

        audio_pad = np.pad(audio, (self.window // 2, self.window // 2), mode="reflect")

        opt_ts = []
        if audio_pad.shape[0] > self.t_max:
            audio_sum = np.zeros_like(audio)
            for i in range(self.window):
                audio_sum += audio_pad[i : i - self.window]
            for t in range(self.t_center, audio.shape[0], self.t_center):
                opt_ts.append(
                    t
                    - self.t_query
                    + np.where(
                        np.abs(audio_sum[t - self.t_query : t + self.t_query])
                        == np.abs(
                            audio_sum[t - self.t_query : t + self.t_query]
                        ).min()
                    )[0][0]
                )

        s = 0
        audio_out = []
        t = None

        audio_pad = np.pad(audio, (self.t_pad, self.t_pad), mode="reflect")
        p_len = audio_pad.shape[0] // self.window
        sid = torch.tensor(sid, device=self.device).unsqueeze(0)

        pitch, pitchf = self.get_f0(
            audio_pad,
            p_len,
            f0_method,
            pitch,
            f0_autotune,
            f0_autotune_strength,
            proposed_pitch,
            proposed_pitch_threshold,
        )

        pitch = torch.tensor(pitch, device=self.device).unsqueeze(0)
        pitchf = torch.tensor(pitchf, device=self.device).unsqueeze(0).float()

        for t in opt_ts:
            t = t // self.window * self.window
            audio_out.append(
                self.voice_conversion(
                    model,
                    net_g,
                    sid,
                    audio_pad[s : t + self.t_pad2 + self.window],
                    pitch[:, s // self.window : (t + self.t_pad2) // self.window],
                    pitchf[:, s // self.window : (t + self.t_pad2) // self.window],
                    index,
                    big_npy,
                    index_rate,
                    version,
                    protect,
                )[self.t_pad_tgt : -self.t_pad_tgt]
            )
            s = t

        audio_out.append(
            self.voice_conversion(
                model,
                net_g,
                sid,
                audio_pad[t:] if t is not None else audio_pad,
                pitch[:, t // self.window :] if t is not None else pitch,
                pitchf[:, t // self.window :] if t is not None else pitchf,
                index,
                big_npy,
                index_rate,
                version,
                protect,
            )[self.t_pad_tgt : -self.t_pad_tgt]
        )

        audio_out = np.concatenate(audio_out)

        if volume_envelope != 1:
            audio_out = AudioProcessor.change_rms(
                audio,
                self.sample_rate,
                audio_out,
                self.tgt_sr,
                volume_envelope,
            )

        mx = np.abs(audio_out).max() / 0.99
        if mx > 1:
            audio_out /= mx

        return audio_out
