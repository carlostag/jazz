import io, random, os, math
import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from mido import Message, MidiFile, MidiTrack, MetaMessage, bpm2tempo
from scipy.io import wavfile
from moviepy.editor import VideoClip, AudioFileClip

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Jazz Shorts Generator", page_icon="🎷")
st.title("🎷 Generador de Jazz Shorts/Reels")

DURATION_SEC = 60
FPS = 30
W, H = 1080, 1920  # formato vertical

# =========================
# SÍNTESIS SIMPLE (vibráfono fake)
# =========================
def synth_sine(freq, dur, sr=44100):
    n = int(sr*dur)
    t = np.linspace(0, dur, n, endpoint=False)
    env = np.exp(-3*t/dur)  # decay exponencial
    y = np.sin(2*np.pi*freq*t) * env
    return y.astype(np.float32)

def midi_to_freq(m):
    return 440.0 * (2**((m-69)/12.0))

# =========================
# GENERADOR AUDIO + EVENTOS
# =========================
def generate_music(bpm=100, root=60):
    beats_per_bar = 4
    bars = int((bpm/120)*30)  # ≈60s
    total_beats = beats_per_bar * bars
    q = 60.0/bpm

    audio = np.zeros(int(44100*DURATION_SEC), dtype=np.float32)
    events = []

    # progresión simple: I–vi–ii–V
    progression = [[root, root+4, root+7],
                   [root+9, root+12, root+16],
                   [root+2, root+5, root+9],
                   [root+7, root+11, root+14]] * (bars//4)

    t = 0.0
    for bar, chord in enumerate(progression):
        t_bar = bar*beats_per_bar*q
        for n in chord:
            f = midi_to_freq(n)
            y = 0.2*synth_sine(f, q*beats_per_bar)
            s = int(t_bar*44100); e = s+len(y)
            if e < len(audio):
                audio[s:e] += y
                events.append((t_bar, q*beats_per_bar, n))
    # Final marcado
    f = midi_to_freq(root)
    y = 0.5*synth_sine(f, 3.0)
    s = int(DURATION_SEC*44100- len(y))
    audio[s:s+len(y)] += y
    events.append((DURATION_SEC-3.0, 3.0, root))

    # normalizar
    audio /= np.max(np.abs(audio)+1e-12)
    return audio, events, bars

# =========================
# VISUAL VIDEO
# =========================
def make_frame(events, duration):
    def frame(t):
        img = Image.new("RGB", (W,H), (10,10,20))
        draw = ImageDraw.Draw(img)
        # título
        draw.text((W//2-200,50),"Random Jazz 🎷",
                  fill=(220,220,220), anchor="lt")
        # dibujar notas que caen
        for t0, dur, m in events:
            y = int((t - t0)*200) + 400
            if 0 < y < H:
                x = int((m-48)*20) % W
                color = (200,100+(m%5)*30,150)
                draw.ellipse([x-20,y-20,x+20,y+20], fill=color)
        return np.array(img)
    return frame

# =========================
# UI
# =========================
if st.button("🎬 Generar Jazz Short aleatorio"):
    bpm = random.randint(80,130)
    root = random.choice([48, 50, 52, 53, 55, 57, 59, 60])  # varias tonalidades
    audio, events, bars = generate_music(bpm, root)

    # export wav temporal
    wav_path = "out.wav"
    wavfile.write(wav_path, 44100, (audio*32767).astype(np.int16))

    # generar video
    clip = VideoClip(make_frame(events,DURATION_SEC), duration=DURATION_SEC)
    audio_clip = AudioFileClip(wav_path)
    clip = clip.set_audio(audio_clip)
    clip = clip.set_fps(FPS)

    out_path = "jazz_short.mp4"
    clip.write_videofile(out_path, codec="libx264", audio_codec="aac", fps=FPS)

    st.success(f"¡Short generado a {bpm} bpm!")
    st.video(out_path)
