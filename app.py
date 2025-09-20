import os, math, random, io
import numpy as np
import streamlit as st
from tempfile import NamedTemporaryFile
from PIL import Image, ImageDraw, ImageFont
from scipy.io import wavfile
from mido import Message, MidiFile, MidiTrack, MetaMessage, bpm2tempo

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Generador Musical Simple", page_icon="🎵")
st.title("🎵 Generador Musical Simple")

SR = 44100

NOTE_TO_SEMITONE = {
    "C":0,"C#":1,"Db":1,"D":2,"D#":3,"Eb":3,
    "E":4,"F":5,"F#":6,"Gb":6,"G":7,"G#":8,"Ab":8,
    "A":9,"A#":10,"Bb":10,"B":11
}

def midi_from_note(note_name, octave=4):
    return 12*(octave+1)+NOTE_TO_SEMITONE[note_name]

def freq_from_midi(m):
    return 440.0*(2.0**((m-69)/12.0))

# =========================
# SÍNTESIS
# =========================
def adsr_envelope(n_samples, sr, attack=0.01, decay=0.2, sustain=0.5, release=0.2):
    a = int(attack * sr); d = int(decay * sr); r = int(release * sr)
    total = a + d + r
    if total > n_samples and total > 0:
        factor = n_samples / total
        a = max(1, int(a * factor)); d = max(1, int(d * factor)); r = max(1, int(r * factor))
    s = max(n_samples - (a + d + r), 0)
    env = np.zeros(n_samples, dtype=np.float32)
    if a > 0: env[:a] = np.linspace(0, 1, a, endpoint=False)
    if d > 0: env[a:a+d] = np.linspace(1, sustain, d, endpoint=False)
    env[a+d:a+d+s] = sustain
    env[a+d+s:a+d+s+r] = np.linspace(sustain, 0, r, endpoint=True)
    return env

def mallet_note(freq, dur, sr):
    n = int(sr * dur)
    t = np.linspace(0, dur, n, endpoint=False)
    x = np.sin(2*np.pi*freq*t).astype(np.float32)
    env = adsr_envelope(n, sr, 0.01, 0.1, 0.7, 0.1)
    return x*env

# =========================
# GENERADOR
# =========================
def chord_voicing(root, quality="maj"):
    return [root, root+4, root+7] if quality=="maj" else [root, root+3, root+7]

def simple_progression(root="C"):
    I  = midi_from_note(root, 4)
    V  = I + 7
    vi = I + 9
    IV = I + 5
    return [(I,"maj"), (V,"maj"), (vi,"min"), (IV,"maj")]

def generate_music(duration=30, bpm=90, root="C", sr=SR):
    q = 60.0/bpm
    beats_per_bar = 4
    total_bars = int(duration/(q*beats_per_bar))
    progression = simple_progression(root)

    audio = np.zeros(int(sr*duration), dtype=np.float32)
    notes = []

    last_mel = None
    for bar in range(total_bars):
        chord_root, quality = progression[bar % len(progression)]
        t_bar = bar*beats_per_bar*q
        dur_ch = beats_per_bar*q

        # Acorde
        for m in chord_voicing(chord_root, quality):
            f = freq_from_midi(m)
            y = 0.2*mallet_note(f, dur_ch, sr)
            s = int(t_bar*sr); audio[s:s+len(y)] += y
            notes.append((t_bar, dur_ch, m, "chord"))

        # Bajo
        f = freq_from_midi(chord_root-24)
        y = 0.3*mallet_note(f, dur_ch, sr)
        s = int(t_bar*sr); audio[s:s+len(y)] += y
        notes.append((t_bar, dur_ch, chord_root-24, "bass"))

        # Melodía
        scale = [chord_root, chord_root+2, chord_root+4, chord_root+7, chord_root+9]
        subbeat = 0.0
        while subbeat < beats_per_bar:
            m = random.choice(scale)
            dur_beats = random.choice([0.25,0.5,1.0])
            dur_time = dur_beats*q
            t_mel = t_bar+subbeat*q
            f = freq_from_midi(m)
            y = 0.4*mallet_note(f, dur_time, sr)
            s = int(t_mel*sr); audio[s:s+len(y)] += y
            notes.append((t_mel, dur_time, m, "mel"))
            subbeat += dur_beats

    audio /= np.max(np.abs(audio)+1e-12)
    return audio, notes

# =========================
# MIDI + PARTITURA SIMPLE
# =========================
def notes_to_midi(notes, bpm=90):
    mid = MidiFile(); track = MidiTrack(); mid.tracks.append(track)
    track.append(MetaMessage('set_tempo', tempo=bpm2tempo(bpm)))
    for t0, dur, m, role in notes:
        tick_on = int(t0*bpm*480/60)
        tick_off = int((t0+dur)*bpm*480/60)
        track.append(Message('note_on', note=m, velocity=80, time=tick_on))
        track.append(Message('note_off', note=m, velocity=64, time=tick_off-tick_on))
    return mid

def draw_score(melody):
    width = 800; height = 200
    img = Image.new("RGB",(width,height),(255,255,255))
    d = ImageDraw.Draw(img)
    y_staff = 100; spacing = 10
    for i in range(5):
        d.line([(50,y_staff+i*spacing),(width-50,y_staff+i*spacing)],fill=(0,0,0))
    x=60
    for _,_,m,_ in melody:
        d.ellipse([x-5,y_staff-(m-60)*2-5,x+5,y_staff-(m-60)*2+5],fill=(0,0,0))
        x+=20
    return img

# =========================
# UI
# =========================
root = st.selectbox("Tonalidad", list(NOTE_TO_SEMITONE.keys()), index=0)
bpm = st.slider("Tempo (BPM)",60,160,90)
duration = st.slider("Duración (segundos)",20,60,30)
seed = st.number_input("Semilla",min_value=0,max_value=9999,value=42)

if st.button("🎶 Generar música"):
    random.seed(seed); np.random.seed(seed)
    audio, notes = generate_music(duration=duration,bpm=bpm,root=root)

    # WAV
    wav_bytes = io.BytesIO()
    wav = np.int16(audio*32767)
    wavfile.write(wav_bytes,SR,wav); wav_bytes.seek(0)

    # MIDI
    midi_bytes = io.BytesIO()
    notes_to_midi(notes,bpm).save(file=midi_bytes); midi_bytes.seek(0)

    # Partitura
    melody=[n for n in notes if n[3]=="mel"]
    img=draw_score(melody[:30])
    buf=io.BytesIO(); img.save(buf,format="PNG"); buf.seek(0)

    st.audio(wav_bytes,format="audio/wav")
    st.image(buf,caption="Partitura simplificada",use_container_width=True)
    st.download_button("⬇️ WAV",data=wav_bytes,file_name="musica.wav")
    st.download_button("⬇️ MIDI",data=midi_bytes,file_name="musica.mid")
