# save as make_speech_plots.py
import os
import numpy as np
import soundfile as sf
import librosa
import webrtcvad
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
import tgt  # Praat TextGrid parser
import shutil

# try Parselmouth for Praat
try:
    import parselmouth
    HAVE_PRAAT = True
except Exception:
    HAVE_PRAAT = False

# ------------- constants -------------
VOWELS = {"AA","AE","AH","AO","AW","AY","EH","ER","EY","IH","IY","OW","OY","UH","UW"}
SIL_TOKENS = {"sil","sp","spn","pau","nsn"}

COLORS = {
    "speech":  "#4e79a7",
    "LS":      "#4e79a7",
    "MS":      "#f28e2b",
    "HS":      "#59a14f",
    "V0":      "#4e79a7",
    "V1":      "#f28e2b",
    "V2":      "#59a14f",
    "C":       "#e15759",
}

# ------------- audio utils -------------
def load_audio_16k(path, target_sr=16000):
    try:
        y, sr = sf.read(path, always_2d=False)
    except Exception:
        y, sr = librosa.load(path, sr=None, mono=True)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != target_sr:
        y = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    y = np.clip(y, -1.0, 1.0).astype(np.float32)
    return y, sr

def float_to_int16_bytes(y):
    return (np.int16(np.clip(y, -1.0, 1.0) * 32767)).tobytes()

def webrtc_vad_mask(y, sr, frame_ms=20, aggressiveness=2):
    assert sr in [8000, 16000, 32000]
    vad = webrtcvad.Vad(aggressiveness)
    frame_len = int(sr * frame_ms / 1000)
    pcm = float_to_int16_bytes(y)
    n_frames = len(y) // frame_len
    speech = np.zeros(n_frames, dtype=bool)
    for i in range(n_frames):
        b0 = i * frame_len * 2
        b1 = b0 + frame_len * 2
        fb = pcm[b0:b1]
        if len(fb) < frame_len * 2:
            break
        speech[i] = vad.is_speech(fb, sr)
    return speech, frame_len

def stft_energy_L2(y, sr, frame_ms=20):
    hop = int(sr * frame_ms / 1000)
    win = hop
    n_fft = max(512, 1 << (win - 1).bit_length())
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop, win_length=win, center=False, window="hann")
    mag = np.abs(S)
    energy = np.linalg.norm(mag, ord=2, axis=0)
    return energy, hop, S.shape[1]

def normalize_log_energy(energy, speech_mask, eps=1e-8):
    speech_energy = energy[speech_mask]
    ref = (np.max(speech_energy) if speech_energy.size else np.max(energy)) + eps
    energy_db = 20.0 * np.log10(np.maximum(energy, eps) / ref)
    if speech_energy.size:
        min_db = 20.0 * np.log10(np.maximum(np.percentile(speech_energy, 5.0), eps) / ref)
    else:
        min_db = np.min(energy_db)
    en = (energy_db - min_db) / (0.0 - min_db + eps)
    return np.clip(en, 0.0, 1.0)

def mask_to_intervals(mask, frame_ms):
    out = []
    i, n = 0, len(mask)
    while i < n:
        if mask[i]:
            j = i + 1
            while j < n and mask[j]:
                j += 1
            out.append((i * frame_ms / 1000.0, j * frame_ms / 1000.0))
            i = j
        else:
            i += 1
    return out

# ------------- textgrid utils -------------
def load_textgrid(path):
    return tgt.io.read_textgrid(path)

def get_intervals(tg, tier_name):
    tier = tg.get_tier_by_name(tier_name)
    return [(it.start_time, it.end_time, it.text.strip()) for it in tier]

def classify_phone(label):
    lab = label.upper()
    if lab in SIL_TOKENS or lab == "":
        return None
    base = "".join([c for c in lab if c.isalpha()])
    stress = "".join([c for c in lab if c.isdigit()])
    if base in VOWELS:
        if stress == "1":
            return "V1"
        elif stress == "2":
            return "V2"
        else:
            return "V0"
    return "C"

def phones_to_class_intervals(tg, phones_tier="phones"):
    classes = {"V0": [], "V1": [], "V2": [], "C": []}
    for s, e, lab in get_intervals(tg, phones_tier):
        cls = classify_phone(lab)
        if cls:
            classes[cls].append((s, e))
    return classes

# ------------- F0 extraction with Praat -------------
def f0_parselmouth(
    y, sr,
    fmin=75.0, fmax=500.0, time_step=None,
    silence_threshold=0.03, voicing_threshold=0.45,
    octave_cost=0.01, octave_jump_cost=0.35, voiced_unvoiced_cost=0.14,
    alg="cc"  # "cc" (cross-correlation) or "ac" (autocorrelation)
):
    if not HAVE_PRAAT:
        raise RuntimeError("Parselmouth is not installed, install with: pip install praat-parselmouth")

    snd = parselmouth.Sound(y, sampling_frequency=sr)
    tstep = None if time_step is None else float(time_step)

    if alg.lower() == "ac":
        pitch = snd.to_pitch_ac(time_step=tstep, pitch_floor=fmin, pitch_ceiling=fmax)
    else:
        # CC with custom thresholds
        try:
            pitch = snd.to_pitch_cc(
                time_step=tstep,
                pitch_floor=fmin,
                max_number_of_candidates=15,
                very_accurate=False,
                silence_threshold=silence_threshold,
                voicing_threshold=voicing_threshold,
                octave_cost=octave_cost,
                octave_jump_cost=octave_jump_cost,
                voiced_unvoiced_cost=voiced_unvoiced_cost,
                pitch_ceiling=fmax,
            )
        except TypeError:
            pitch = snd.to_pitch_cc(time_step=tstep, pitch_floor=fmin, pitch_ceiling=fmax)

    f0 = pitch.selected_array["frequency"].astype(float)
    start = pitch.xmin
    step = pitch.dx
    t = start + np.arange(f0.shape[0]) * step
    return t, f0

def f0_librosa(y, sr, fmin=75.0, fmax=500.0, hop=None):
    hop = 160 if hop is None else hop
    f0, vflag, _ = librosa.pyin(y, fmin=fmin, fmax=fmax, sr=sr,
                                frame_length=2048, hop_length=hop, center=False)
    t = np.arange(len(f0)) * hop / sr
    return t, f0

# ------------- plotting helpers -------------
def mel_spectrogram_db(y, sr, n_mels=80, hop_length=None):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels,
                                       hop_length=hop_length, center=False)
    SdB = librosa.power_to_db(S, ref=np.max)
    return SdB

def draw_spec_bg(ax, y, sr, n_mels=80, hop_length=None, cmap="gray", alpha=0.35):
    SdB = mel_spectrogram_db(y, sr, n_mels=n_mels, hop_length=hop_length)
    img = ax.imshow(
        SdB, origin="lower", aspect="auto",
        extent=[0, len(y) / sr, 0, sr / 2],
        cmap=cmap, alpha=alpha, zorder=1
    )
    ax.set_ylabel("Frequency [Hz]")
    ax.set_xlabel("Time [s]")
    return img

def draw_spans(ax, intervals, color, label, alpha=0.6, edgecolor="black", lw=0.6, y0=0, y1=1, z=2):
    first = True
    for s, e in intervals:
        ax.axvspan(s, e, ymin=y0, ymax=y1,
                   facecolor=color, edgecolor=edgecolor,
                   alpha=alpha, linewidth=lw, label=(label if first else None),
                   zorder=z)
        first = False

def unique_legend(ax):
    h, l = ax.get_legend_handles_labels()
    d = {}
    for handle, lab in zip(h, l):
        if lab and lab not in d:
            d[lab] = handle
    if d:
        leg = ax.legend(d.values(), d.keys(), loc="upper right", frameon=True)
        leg.get_frame().set_alpha(0.85)

# ------------- main figure maker -------------
def make_bundle(
    wav_path,
    textgrid_path=None,
    out_root="viz_out",
    frame_ms=20,
    vad_aggr=2,
    phones_tier="phones",
    words_tier="words",
    bg_cmap="gray",
    bg_alpha=0.35,
    f0_method="praat",
    f0_floor=75.0,
    f0_ceiling=500.0,
    silence_threshold=0.03,
    voicing_threshold=0.45,
    octave_cost=0.01,
    octave_jump_cost=0.35,
    voiced_unvoiced_cost=0.14,
    ymax_hz=4000.0,
    clean_dir=False,
    praat_alg="cc",
    f0_axis_max=300.0
):
    """
    Writes four PNGs in out_root/<audio_id>
      speech_vs_nonspeech_spec.png
      ls_ms_hs_spec.png
      phones_V0_V1_V2_C_spec.png
      words_with_f0_spec.png
    """
    wav_path = Path(wav_path)
    audio_name = wav_path.stem
    out_dir = Path(out_root) / audio_name

    # optional: nuke the whole per-audio folder first
    if clean_dir and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # always overwrite the known files
    for old in ["speech_vs_nonspeech_spec.png",
                "ls_ms_hs_spec.png",
                "phones_V0_V1_V2_C_spec.png",
                "words_with_f0_spec.png"]:
        p = out_dir / old
        if p.exists():
            try:
                p.unlink()
            except Exception:
                pass

    y, sr = load_audio_16k(str(wav_path), 16000)
    hop_viz = int(sr * frame_ms / 1000)

    # 1) speech vs nonspeech
    speech_mask, _ = webrtc_vad_mask(y, sr, frame_ms=frame_ms, aggressiveness=vad_aggr)
    speech_intervals = mask_to_intervals(speech_mask, frame_ms)
    fig, ax = plt.subplots(figsize=(11, 3))
    draw_spec_bg(ax, y, sr, hop_length=hop_viz, cmap=bg_cmap, alpha=bg_alpha)
    draw_spans(ax, speech_intervals, color=COLORS["speech"], label="Speech", alpha=0.6)
    ax.set_title("Detected Speech on Mel Spectrogram")
    unique_legend(ax)
    if ymax_hz:
        ax.set_ylim(0, min(ymax_hz, sr/2))
    fig.tight_layout()
    fig.savefig(out_dir / "speech_vs_nonspeech_spec.png", dpi=150)
    plt.close(fig)

    # 2) LS MS HS
    energy, _, _ = stft_energy_L2(y, sr, frame_ms=frame_ms)
    n = min(len(energy), len(speech_mask))
    energy, speech_mask = energy[:n], speech_mask[:n]
    en = normalize_log_energy(energy, speech_mask)
    t1, t2 = 1/3, 2/3
    ls = mask_to_intervals((en < t1) & speech_mask, frame_ms)
    ms = mask_to_intervals((en >= t1) & (en < t2) & speech_mask, frame_ms)
    hs = mask_to_intervals((en >= t2) & speech_mask, frame_ms)
    fig, ax = plt.subplots(figsize=(11, 3))
    draw_spec_bg(ax, y, sr, hop_length=hop_viz, cmap=bg_cmap, alpha=bg_alpha)
    draw_spans(ax, ls, COLORS["LS"], "LS", 0.55)
    draw_spans(ax, ms, COLORS["MS"], "MS", 0.55)
    draw_spans(ax, hs, COLORS["HS"], "HS", 0.55)
    ax.set_title("Energy Regions on Mel Spectrogram, LS or MS or HS")
    unique_legend(ax)
    if ymax_hz:
        ax.set_ylim(0, min(ymax_hz, sr/2))
    fig.tight_layout()
    fig.savefig(out_dir / "ls_ms_hs_spec.png", dpi=150)
    plt.close(fig)

    # load TextGrid if present
    tg = None
    if textgrid_path is not None and Path(textgrid_path).exists():
        try:
            tg = load_textgrid(str(textgrid_path))
        except Exception:
            tg = None

    # 3) phones
    if tg is not None:
        classes = phones_to_class_intervals(tg, phones_tier=phones_tier)
        fig, ax = plt.subplots(figsize=(11, 3))
        draw_spec_bg(ax, y, sr, hop_length=hop_viz, cmap=bg_cmap, alpha=bg_alpha)
        draw_spans(ax, classes["V0"], COLORS["V0"], "V0", 0.55)
        draw_spans(ax, classes["V1"], COLORS["V1"], "V1", 0.55)
        draw_spans(ax, classes["V2"], COLORS["V2"], "V2", 0.55)
        draw_spans(ax, classes["C"],  COLORS["C"],  "C",  0.50)
        ax.set_title("Phoneme Classes on Mel Spectrogram, V0 or V1 or V2 or C")
        unique_legend(ax)
        if ymax_hz:
            ax.set_ylim(0, min(ymax_hz, sr/2))
        fig.tight_layout()
        fig.savefig(out_dir / "phones_V0_V1_V2_C_spec.png", dpi=150)
        plt.close(fig)

    # 4) words with F0 on a heatmap, yellow to orange palette
    cmap_heat = LinearSegmentedColormap.from_list(
        "yellow_orange",
        ["#ffffcc", "#ffd27f", "#ffb347", "#ff8c00", "#c84e00", "#7f2704"]
    )
    SdB = mel_spectrogram_db(y, sr, n_mels=80, hop_length=hop_viz)
    vmin, vmax = -45.0, 0.0

    # Praat F0 (or fallback)
    if f0_method.lower() == "praat":
        t_f0, f0 = f0_parselmouth(
            y, sr,
            fmin=f0_floor, fmax=f0_ceiling, time_step=None,
            silence_threshold=silence_threshold, voicing_threshold=voicing_threshold,
            octave_cost=octave_cost, octave_jump_cost=octave_jump_cost,
            voiced_unvoiced_cost=voiced_unvoiced_cost,
            alg=praat_alg,
        )
    else:
        t_f0, f0 = f0_librosa(y, sr, fmin=f0_floor, fmax=f0_ceiling, hop=hop_viz)

    # mask unvoiced/invalid F0 so we don't draw zeros
    if t_f0 is not None and f0 is not None:
        f0 = f0.astype(float)
        f0[(~np.isfinite(f0)) | (f0 <= 0.0)] = np.nan

    # words tier
    words = []
    if tg is not None:
        try:
            words = get_intervals(tg, words_tier)
        except Exception:
            words = []

    fig, ax = plt.subplots(figsize=(11, 3))
    im = ax.imshow(
        SdB, origin="lower", aspect="auto",
        extent=[0, len(y) / sr, 0, sr / 2],
        cmap=cmap_heat, vmin=vmin, vmax=vmax, zorder=1
    )
    if ymax_hz:
        ax.set_ylim(0, min(ymax_hz, sr/2))
    ax.set_ylabel("Frequency [Hz]")
    ax.set_xlabel("Time [s]")
    cbar = plt.colorbar(im, ax=ax, pad=0.08, shrink=0.9)
    cbar.set_label("Amplitude [dB]")

    # word boundaries, labels above axes
    if words:
        for s, e, w in words:
            ax.axvline(s, color="white", linewidth=1, alpha=0.85, zorder=3)
            ax.axvline(e, color="white", linewidth=1, alpha=0.85, zorder=3)
            xm = 0.5 * (s + e)
            ax.text(
                xm, 1.03, w,
                transform=ax.get_xaxis_transform(),
                ha="center", va="bottom",
                color="black", fontsize=10, fontweight="normal",
                clip_on=False, zorder=5
            )

    # RIGHT y axis for F0, independent scale
    ax2 = ax.twinx()
    ax2.set_ylim(0, float(f0_axis_max))
    ax2.set_ylabel("f0 [Hz]")

    # F0 curve and points drawn on RIGHT axis (shares x with left)
    if t_f0 is not None and f0 is not None:
        ax2.plot(t_f0, f0, linestyle="-", linewidth=1.2, color="black", zorder=4)
        voiced = np.isfinite(f0)
        ax2.scatter(t_f0[voiced], f0[voiced], s=7, color="black", zorder=5)

    # move the title up so it doesn't collide with labels
    ax.set_title("Words on Spectrogram, with F0 contour", pad=26)
    fig.tight_layout()
    fig.savefig(out_dir / "words_with_f0_spec.png", dpi=150)
    plt.close(fig)

    return {
        "audio_dir": str(out_dir),
        "speech_png": str(out_dir / "speech_vs_nonspeech_spec.png"),
        "energy_png": str(out_dir / "ls_ms_hs_spec.png"),
        "phones_png": str(out_dir / "phones_V0_V1_V2_C_spec.png")
                      if (out_dir / "phones_V0_V1_V2_C_spec.png").exists() else None,
        "words_png": str(out_dir / "words_with_f0_spec.png"),
    }

# ------------- CLI -------------
if __name__ == "__main__":
    import argparse, warnings, sys
    warnings.filterwarnings("ignore", category=UserWarning, module="webrtcvad")

    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", help="path to one audio file")
    ap.add_argument("--wav-dir", help="directory to scan for audio files")
    ap.add_argument("--textgrid", help="TextGrid path for the single wav")
    ap.add_argument("--textgrid-root", help="root dir to search for matching TextGrid when batching")
    ap.add_argument("--out-root", default="viz_out")
    ap.add_argument("--frame-ms", type=int, default=20)
    ap.add_argument("--vad", type=int, default=2)
    ap.add_argument("--phones-tier", default="phones")
    ap.add_argument("--words-tier", default="words")
    ap.add_argument("--bg-cmap", default="gray")
    ap.add_argument("--bg-alpha", type=float, default=0.35)
    ap.add_argument("--f0-method", default="praat", choices=["praat", "pyin"])
    ap.add_argument("--praat-alg", default="cc", choices=["cc","ac"],
                    help="Praat pitch algorithm: cc (cross-correlation) or ac (autocorrelation)")
    ap.add_argument("--f0-floor", type=float, default=75.0)
    ap.add_argument("--f0-ceiling", type=float, default=500.0)
    ap.add_argument("--f0-axis-max", type=float, default=300.0,
                    help="Right-axis maximum in Hz for plotting F0")
    ap.add_argument("--silence-threshold", type=float, default=0.03)
    ap.add_argument("--voicing-threshold", type=float, default=0.45)
    ap.add_argument("--octave-cost", type=float, default=0.01)
    ap.add_argument("--octave-jump-cost", type=float, default=0.35)
    ap.add_argument("--voiced-unvoiced-cost", type=float, default=0.14)
    ap.add_argument("--ymax-hz", type=float, default=4000.0)
    ap.add_argument("--clean-dir", action="store_true",
                    help="Delete the per-audio output folder before writing (strong overwrite).")
    args = ap.parse_args()

    if not args.wav and not args.wav_dir:
        print("Provide --wav or --wav-dir", file=sys.stderr)
        sys.exit(2)

    def find_matching_textgrid(stem, root):
        if not root:
            return None
        candidates = list(Path(root).rglob(f"{stem}.TextGrid"))
        return str(candidates[0]) if candidates else None

    if args.wav:
        tg = args.textgrid
        if tg is None and args.textgrid_root:
            tg = find_matching_textgrid(Path(args.wav).stem, args.textgrid_root)
        print(make_bundle(
            args.wav, tg, out_root=args.out_root,
            frame_ms=args.frame_ms, vad_aggr=args.vad,
            phones_tier=args.phones_tier, words_tier=args.words_tier,
            bg_cmap=args.bg_cmap, bg_alpha=args.bg_alpha,
            f0_method=args.f0_method, f0_floor=args.f0_floor, f0_ceiling=args.f0_ceiling,
            silence_threshold=args.silence_threshold, voicing_threshold=args.voicing_threshold,
            octave_cost=args.octave_cost, octave_jump_cost=args.octave_jump_cost,
            voiced_unvoiced_cost=args.voiced_unvoiced_cost,
            ymax_hz=args.ymax_hz, clean_dir=args.clean_dir,
            praat_alg=args.praat_alg, f0_axis_max=args.f0_axis_max
        ))
    else:
        audio_exts = {".wav", ".flac"}
        for p in Path(args.wav_dir).rglob("*"):
            if p.suffix.lower() in audio_exts:
                tg = find_matching_textgrid(p.stem, args.textgrid_root)
                print(make_bundle(
                    str(p), tg, out_root=args.out_root,
                    frame_ms=args.frame_ms, vad_aggr=args.vad,
                    phones_tier=args.phones_tier, words_tier=args.words_tier,
                    bg_cmap=args.bg_cmap, bg_alpha=args.bg_alpha,
                    f0_method=args.f0_method, f0_floor=args.f0_floor, f0_ceiling=args.f0_ceiling,
                    silence_threshold=args.silence_threshold, voicing_threshold=args.voicing_threshold,
                    octave_cost=args.octave_cost, octave_jump_cost=args.octave_jump_cost,
                    voiced_unvoiced_cost=args.voiced_unvoiced_cost,
                    ymax_hz=args.ymax_hz, clean_dir=args.clean_dir,
                    praat_alg=args.praat_alg, f0_axis_max=args.f0_axis_max
                ))