import io, random
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from mido import Message, MidiFile, MidiTrack, MetaMessage, bpm2tempo

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Generador Jazz Combo", page_icon="🎷")
st.title("🎷 Generador de Combo de Jazz (MIDI + Partitura)")

TICKS_PER_BEAT = 480
BEATS_PER_BAR = 4
BARS = 8
TOTAL_BEATS = BEATS_PER_BAR * BARS

# Escalas diatónicas
MAJOR_STEPS = [0, 2, 4, 5, 7, 9, 11]
MINOR_STEPS = [0, 2, 3, 5, 7, 8, 10]

KEY_OPTIONS = {
    "Do mayor (C major)": {"root_midi": 60, "mode": "major"},
    "La menor (A minor)": {"root_midi": 57, "mode": "minor"},
}

INSTRUMENTS = {
    "Vibráfono": 11,
    "Trompeta": 56,
}

# =========================
# UTILIDADES MUSICALES
# =========================
def scale_steps(mode):
    return MAJOR_STEPS if mode == "major" else MINOR_STEPS

def degree_to_midi(root_midi, degree, mode):
    steps = scale_steps(mode)
    diatonic = steps[degree % 7] + 12 * (degree // 7)
    return root_midi + diatonic

def walking_bass(root_midi, mode):
    notes = []
    deg = 0
    for bar in range(BARS):
        for beat in range(BEATS_PER_BAR):
            t = bar * BEATS_PER_BAR + beat
            if beat == 0:
                deg = 0
            else:
                step = random.choice([-2, -1, 1, 2])
                deg += step
            midi = degree_to_midi(root_midi - 12, deg, mode)
            notes.append((t, 1.0, midi))
    return notes

def chords(root_midi, mode):
    chords = []
    for bar in range(BARS):
        for beat in [0, 2]:
            t = bar * BEATS_PER_BAR + beat
            root = root_midi
            chord = [root, root+4, root+7, root+11] if mode=="major" else [root, root+3, root+7, root+10]
            for n in chord:
                chords.append((t, 2.0, n))
    return chords

def melody(root_midi, mode, seed):
    random.seed(seed)
    deg = 7
    events = []
    t = 0
    while t < TOTAL_BEATS:
        dur = random.choice([0.5, 1.0])
        step = random.choice([-2, -1, 1, 2])
        deg += step
        midi = degree_to_midi(root_midi, deg, mode)
        if t + dur > TOTAL_BEATS: dur = TOTAL_BEATS - t
        events.append((t, dur, midi))
        t += dur
    return events

def swing_drums():
    events = []
    for bar in range(BARS):
        for beat in range(BEATS_PER_BAR):
            t = bar*BEATS_PER_BAR + beat
            # Ride (51)
            if beat in [0,1,2,3]:
                dur = 0.5 if beat%2==0 else 0.25
                events.append((t, dur, 51))
            if beat in [1,3]:
                events.append((t, 0.25, 36))  # bombo
                events.append((t, 0.25, 38))  # caja
    return events

# =========================
# MIDI GENERATOR
# =========================
def events_to_midi(melody_ev, bass_ev, chord_ev, drum_ev, bpm, instr_melody):
    mid = MidiFile(ticks_per_beat=TICKS_PER_BEAT)

    mel_track = MidiTrack(); mid.tracks.append(mel_track)
    mel_track.append(MetaMessage('set_tempo', tempo=bpm2tempo(bpm)))
    mel_track.append(Message('program_change', program=instr_melody, channel=0, time=0))

    bass_track = MidiTrack(); mid.tracks.append(bass_track)
    bass_track.append(Message('program_change', program=32, channel=1, time=0))

    chord_track = MidiTrack(); mid.tracks.append(chord_track)
    chord_track.append(Message('program_change', program=0, channel=2, time=0))

    drum_track = MidiTrack(); mid.tracks.append(drum_track)

    def add_events(track, events, channel):
        evts = []
        for t, dur, midi in events:
            on = ('on', int(round(t*TICKS_PER_BEAT)), midi)
            off = ('off', int(round((t+dur)*TICKS_PER_BEAT)), midi)
            evts += [on, off]
        evts.sort(key=lambda x: x[1])
        cursor = 0
        for typ, tick, midi in evts:
            delta = tick - cursor
            cursor = tick
            if typ == 'on':
                track.append(Message('note_on', note=int(midi), velocity=90, time=delta, channel=channel))
            else:
                track.append(Message('note_off', note=int(midi), velocity=64, time=delta, channel=channel))

    add_events(mel_track, melody_ev, 0)
    add_events(bass_track, bass_ev, 1)
    add_events(chord_track, chord_ev, 2)
    add_events(drum_track, drum_ev, 9)

    return mid

# =========================
# PARTITURA PNG (melodía)
# =========================
def draw_score(events, key_name, bpm):
    width, height = 1000, 220
    img = Image.new("RGB", (width,height),(255,255,255))
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 20)
    except:
        font = None
    d.text((40,10), f"Melodía ({key_name}, {bpm} bpm)", fill=(0,0,0), font=font)
    y_staff = 120
    sp = 12
    for i in range(5):
        y = y_staff+i*sp
        d.line([(40,y),(width-40,y)], fill=(0,0,0))
    x=60
    for t,dur,m in events[:32]:
        y = y_staff+2*sp - (m-60)*0.5*sp
        d.ellipse([x-5,y-5,x+5,y+5], fill=(0,0,0))
        x+=20
    return img

# =========================
# UI
# =========================
col1,col2,col3=st.columns(3)
with col1:
    key_name = st.selectbox("Tonalidad", list(KEY_OPTIONS.keys()))
with col2:
    bpm = st.slider("BPM",60,140,100)
with col3:
    seed = st.number_input("Semilla",0,99999,42)

mel_instr_name = st.radio("Instrumento de la melodía", list(INSTRUMENTS.keys()))
instr_melody = INSTRUMENTS[mel_instr_name]

if st.button("🎶 Generar combo de jazz (MIDI + Partitura)"):
    cfg = KEY_OPTIONS[key_name]
    mel = melody(cfg["root_midi"], cfg["mode"], seed)
    bass = walking_bass(cfg["root_midi"], cfg["mode"])
    chords_ev = chords(cfg["root_midi"], cfg["mode"])
    drums = swing_drums()

    mid = events_to_midi(mel, bass, chords_ev, drums, bpm, instr_melody)

    midi_bytes = io.BytesIO(); mid.save(file=midi_bytes); midi_bytes.seek(0)

    img = draw_score(mel, key_name, bpm)
    buf = io.BytesIO(); img.save(buf, format="PNG"); buf.seek(0)

    st.success("¡Listo! Aquí tienes tu combo de jazz 🎷🥁🎶")
    st.image(buf, caption="Partitura simplificada de la melodía")
    st.download_button("⬇️ MIDI", data=midi_bytes, file_name="combo.mid")
