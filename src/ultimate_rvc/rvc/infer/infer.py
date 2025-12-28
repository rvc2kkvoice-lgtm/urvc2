from typing import TYPE_CHECKING, Unpack

import logging
import os
import pathlib
import sys
import time
import traceback

import numpy as np
import torch
import librosa
import soundfile as sf

from pedalboard import (
    Bitcrush,
    Chorus,
    Clipping,
    Compressor,
    Delay,
    Distortion,
    Gain,
    Limiter,
    Pedalboard,
    PitchShift,
    Reverb,
)

now_dir = pathlib.Path.cwd()
sys.path.append(str(now_dir))

import lazy_loader as lazy

from ultimate_rvc.rvc.configs.config import Config
from ultimate_rvc.rvc.infer.pipeline import Pipeline as VC
from ultimate_rvc.rvc.infer.typing_extra import ConvertAudioKwArgs
from ultimate_rvc.rvc.lib.algorithm.synthesizers import Synthesizer
from ultimate_rvc.rvc.lib.tools.split_audio import merge_audio, process_audio
from ultimate_rvc.rvc.lib.utils import load_audio_infer, load_embedding
from ultimate_rvc.typing_extra import F0Method

if TYPE_CHECKING:
    import noisereduce as nr
else:
    nr = lazy.load("noisereduce")

logger = logging.getLogger(__name__)


class VoiceConverter:
    def __init__(self):
        self.config = Config()
        self.hubert_model = None
        self.last_embedder_model = None
        self.tgt_sr = None
        self.net_g = None
        self.vc = None
        self.cpt = None
        self.version = None
        self.n_spk = None
        self.use_f0 = 1  # 🔒 FORCE pitch guidance permanently
        self.loaded_model = None

    # ---------------- EMBEDDER ---------------- #

    def load_hubert(self, embedder_model: str, embedder_model_custom: str | None = None):
        self.hubert_model = load_embedding(embedder_model, embedder_model_custom)
        self.hubert_model = self.hubert_model.to(self.config.device).float()
        self.hubert_model.eval()

    # ---------------- AUDIO UTILS ---------------- #

    @staticmethod
    def remove_audio_noise(data, sr, reduction_strength=0.7):
        try:
            return nr.reduce_noise(y=data, sr=sr, prop_decrease=reduction_strength)
        except Exception as e:
            logger.warning("Noise reduction failed: %s", e)
            return data

    @staticmethod
    def post_process_audio(audio_input, sample_rate, **kwargs):
        board = Pedalboard()

        if kwargs.get("reverb"):
            board.append(Reverb(
                room_size=kwargs.get("reverb_room_size", 0.5),
                damping=kwargs.get("reverb_damping", 0.5),
                wet_level=kwargs.get("reverb_wet_level", 0.33),
                dry_level=kwargs.get("reverb_dry_level", 0.4),
                width=kwargs.get("reverb_width", 1.0),
                freeze_mode=kwargs.get("reverb_freeze_mode", 0),
            ))

        if kwargs.get("pitch_shift"):
            board.append(PitchShift(semitones=kwargs.get("pitch_shift_semitones", 0)))

        if kwargs.get("limiter"):
            board.append(Limiter(
                threshold_db=kwargs.get("limiter_threshold", -6),
                release_ms=kwargs.get("limiter_release", 0.05),
            ))

        if kwargs.get("gain"):
            board.append(Gain(gain_db=kwargs.get("gain_db", 0)))

        if kwargs.get("distortion"):
            board.append(Distortion(drive_db=kwargs.get("distortion_gain", 25)))

        if kwargs.get("chorus"):
            board.append(Chorus(
                rate_hz=kwargs.get("chorus_rate", 1.0),
                depth=kwargs.get("chorus_depth", 0.25),
            ))

        if kwargs.get("bitcrush"):
            board.append(Bitcrush(bit_depth=kwargs.get("bitcrush_bit_depth", 8)))

        if kwargs.get("clipping"):
            board.append(Clipping(threshold_db=kwargs.get("clipping_threshold", 0)))

        if kwargs.get("compressor"):
            board.append(Compressor(
                threshold_db=kwargs.get("compressor_threshold", 0),
                ratio=kwargs.get("compressor_ratio", 1),
                attack_ms=kwargs.get("compressor_attack", 1.0),
                release_ms=kwargs.get("compressor_release", 100),
            ))

        if kwargs.get("delay"):
            board.append(Delay(
                delay_seconds=kwargs.get("delay_seconds", 0.5),
                feedback=kwargs.get("delay_feedback", 0.0),
                mix=kwargs.get("delay_mix", 0.5),
            ))

        return board(audio_input, sample_rate)

    # ---------------- MAIN CONVERSION ---------------- #

    def convert_audio(
        self,
        audio_input_path: str,
        audio_output_path: str,
        model_path: str,
        index_path: str,
        pitch: int = 0,
        f0_method: F0Method = "rmvpe",
        index_rate: float = 0.75,
        volume_envelope: float = 1,
        protect: float = 0.5,
        split_audio: bool = False,
        f0_autotune: bool = False,
        f0_autotune_strength: float = 1.0,
        embedder_model: str = "contentvec",
        embedder_model_custom: str | None = None,
        clean_audio: bool = False,
        clean_strength: float = 0.5,
        export_format: str = "WAV",
        post_process: bool = False,
        resample_sr: int = 0,
        sid: int = 0,
        proposed_pitch: bool = False,
        proposed_pitch_threshold: float = 155.0,
        **kwargs: Unpack[ConvertAudioKwArgs],
    ):
        self.get_vc(model_path, sid)

        audio = load_audio_infer(audio_input_path, 16000, **kwargs)
        audio /= max(1.0, np.abs(audio).max() / 0.95)

        if not self.hubert_model or embedder_model != self.last_embedder_model:
            self.load_hubert(embedder_model, embedder_model_custom)
            self.last_embedder_model = embedder_model

        file_index = index_path.replace("trained", "added").strip()

        chunks = [audio]
        intervals = None
        if split_audio:
            chunks, intervals = process_audio(audio, 16000)

        converted_chunks = []
        for c in chunks:
            converted_chunks.append(
                self.vc.pipeline(
                    model=self.hubert_model,
                    net_g=self.net_g,
                    sid=sid,
                    audio=c,
                    pitch=pitch,
                    f0_method=f0_method,
                    file_index=file_index,
                    index_rate=index_rate,
                    pitch_guidance=True,  # 🔒 FORCE ENABLED
                    volume_envelope=volume_envelope,
                    version=self.version,
                    protect=protect,
                    f0_autotune=f0_autotune,
                    f0_autotune_strength=f0_autotune_strength,
                    proposed_pitch=proposed_pitch,
                    proposed_pitch_threshold=proposed_pitch_threshold,
                )
            )

        audio_opt = (
            merge_audio(chunks, converted_chunks, intervals, 16000, self.tgt_sr)
            if split_audio
            else converted_chunks[0]
        )

        if clean_audio:
            audio_opt = self.remove_audio_noise(audio_opt, self.tgt_sr, clean_strength)

        if post_process:
            audio_opt = self.post_process_audio(audio_opt, self.tgt_sr, **kwargs)

        sf.write(audio_output_path, audio_opt, self.tgt_sr, format="WAV")

    # ---------------- MODEL LOADING ---------------- #

    def get_vc(self, weight_root, sid):
        if self.loaded_model != weight_root:
            self.cleanup_model()
            self.load_model(weight_root)
            self.setup_network()
            self.setup_vc_instance()
            self.loaded_model = weight_root

    def cleanup_model(self):
        for attr in ("net_g", "vc", "hubert_model"):
            if getattr(self, attr, None) is not None:
                delattr(self, attr)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def load_model(self, weight_root):
        self.cpt = torch.load(weight_root, map_location="cpu", weights_only=False)

    def setup_network(self):
        self.tgt_sr = self.cpt["config"][-1]
        self.cpt["config"][-3] = self.cpt["weight"]["emb_g.weight"].shape[0]

        self.version = self.cpt.get("version", "v1")
        text_dim = 768 if self.version == "v2" else 256
        vocoder = self.cpt.get("vocoder", "HiFi-GAN")

        self.net_g = Synthesizer(
            *self.cpt["config"],
            use_f0=1,  # 🔒 PERMANENT
            text_enc_hidden_dim=text_dim,
            vocoder=vocoder,
        )

        del self.net_g.enc_q
        self.net_g.load_state_dict(self.cpt["weight"], strict=False)
        self.net_g = self.net_g.to(self.config.device).float()
        self.net_g.eval()

    def setup_vc_instance(self):
        self.vc = VC(self.tgt_sr, self.config)
        self.n_spk = self.cpt["config"][-3]
