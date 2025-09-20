import io, random
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from mido import Message, MidiFile, MidiTrack, MetaMessage, bpm2tempo
# music21 está importado por compatibilidad futura (PDF/MusicXML si luego lo usas)
from music21 import stream as m21stream, note as m21note, chord as m21chord
from music21 import instrument as m21inst, meter as m21meter, tempo as m21tempo
from music21 import key as m21key, duration as m21duration, metadata as m21metadata

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Generador Jazz Combo", page_icon="🎷")
st.title("🎷 Generador de Combo de Jazz (MIDI + Partitura)")

TICKS_PER_BEAT = 480
BEATS_PER_BAR = 4

KEY_OPTIONS = {
    "Do mayor (C major)": {"root_midi": 60, "mode": "major", "tonic": "C", "scale_mode": "major"},
    "La menor (A minor)": {"root_midi": 57, "mode": "minor", "tonic": "A", "scale_mode": "minor"},
}

INSTRUMENTS = {
    "Vibráfono": 11,
    "Trompeta": 56,
}

# Escalas diatónicas
MAJOR_STEPS = [0, 2, 4, 5, 7, 9, 11]
MINOR_STEPS = [0, 2, 3, 5, 7, 8, 10]

# =========================
# UTILIDADES MUSICALES
# =========================
def scale_steps(mode):
    return MAJOR_STEPS if mode == "major" else MINOR_STEPS

def degree_to_midi(root_midi, degree, mode):
    steps = scale_steps(mode)
    diatonic = steps[degree % 7] + 12 * (degree // 7)
    return root_midi + diatonic

def clamp_to_range(midi, lo=60, hi=84):
    while midi < lo:
        midi += 12
    while midi > hi:
        midi -= 12
    return midi

# =========================
# PROGRESIONES
# =========================
def build_progression(root_midi, mode, progression_type="I-vi-ii-V"):
    if progression_type == "I-vi-ii-V":
        bars = 8
        if mode == "major":
            four = [
                [root_midi, root_midi+4, root_midi+7, root_midi+11],       # Cmaj7
                [root_midi+9, root_midi+12, root_midi+16, root_midi+19],   # Am7
                [root_midi+2, root_midi+5, root_midi+9, root_midi+12],     # Dm7
                [root_midi+7, root_midi+11, root_midi+14, root_midi+17],   # G7
            ]
        else:
            four = [
                [root_midi, root_midi+3, root_midi+7, root_midi+10],       # Am7
                [root_midi+8, root_midi+12, root_midi+15, root_midi+19],   # Fmaj7
                [root_midi+2, root_midi+5, root_midi+10, root_midi+14],    # Bm7b5
                [root_midi+7, root_midi+11, root_midi+14, root_midi+17],   # E7
            ]
        return four * 2, bars

    elif progression_type == "Blues en C":
        bars = 12
        I7  = [root_midi, root_midi+4, root_midi+7, root_midi+10]
        IV7 = [root_midi+5, root_midi+9, root_midi+12, root_midi+15]
        V7  = [root_midi+7, root_midi+11, root_midi+14, root_midi+17]
        prog = [I7, I7, I7, I7, IV7, IV7, I7, I7, V7, IV7, I7, V7]
        return prog, bars

# =========================
# GENERADORES
# =========================
def melody(root_midi, mode, seed, total_bars):
    random.seed(seed)
    deg = 7
    events = []
    TOTAL_BEATS = total_bars * BEATS_PER_BAR
    t = 0.0
    while t < TOTAL_BEATS:
        dur = random.choice([0.5, 1.0])
        step = random.choice([-2, -1, 1, 2])
        deg += step
        midi = degree_to_midi(root_midi, deg, mode)
        midi = clamp_to_range(midi, 60, 84)  # vibrafono rango cómodo
        if t + dur > TOTAL_BEATS:
            dur = TOTAL_BEATS - t
        events.append((t, dur, midi))
        t += dur
    return events

def walking_bass(prog):
    events = []
    for bar, chord in enumerate(prog):
        root = chord[0] - 24
        for beat in range(BEATS_PER_BAR):
            t = bar*BEATS_PER_BAR + beat
            events.append((t, 1.0, root))
    return events

def chord_hits(prog):
    events = []
    for bar, chord in enumerate(prog):
        for beat in [0, 2]:
            t = bar*BEATS_PER_BAR + beat
            for n in chord:
                events.append((t, 2.0, n))
    return events

def swing_drums(total_bars):
    events = []
    for bar in range(total_bars):
        for beat in range(BEATS_PER_BAR):
            t = bar*BEATS_PER_BAR + beat
            if beat in [0,1,2,3]:
                dur = 0.5 if beat % 2 == 0 else 0.25
                events.append((t, dur, 51))  # ride
            if beat in [1,3]:
                events.append((t, 0.25, 36)) # bombo
                events.append((t, 0.25, 38)) # caja
    return events

# =========================
# MIDI BUILDER
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
        evts.sort(key=lambda x: (x[1], 0 if x[0]=='off' else 1))
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
# PARTITURA SIMPLIFICADA PNG (opcional)
# =========================
def draw_score_preview(events, key_name, bpm):
    width, height = 1000, 220
    img = Image.new("RGB", (width, height), (255, 255, 255))
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 20)
    except:
        font = None
    d.text((40, 10), f"Melodía ({key_name}, {bpm} bpm)", fill=(0, 0, 0), font=font)
    y_staff = 120
    sp = 12
    for i in range(5):
        y = y_staff + i * sp
        d.line([(40, y), (width - 40, y)], fill=(0, 0, 0))
    x = 60
    for t, dur, m in events[:32]:
        y = y_staff + 2 * sp - (m - 60) * 0.5 * sp
        d.ellipse([x - 5, y - 5, x + 5, y + 5], fill=(0, 0, 0))
        x += 20
    return img

# =========================
# UI
# =========================
col1, col2, col3 = st.columns(3)
with col1:
    key_name = st.selectbox("Tonalidad", list(KEY_OPTIONS.keys()), key="sel_key")
with col2:
    bpm = st.slider("BPM", 60, 140, 100, key="slider_bpm")
with col3:
    seed = st.number_input("Semilla", 0, 99999, 42, key="num_seed")

progression_type = st.radio("Tipo de progresión", ["I-vi-ii-V", "Blues en C"], key="radio_prog")
mel_instr_name = st.radio("Instrumento de la melodía (MIDI)", list(INSTRUMENTS.keys()), index=0, key="radio_instr")
instr_melody = INSTRUMENTS[mel_instr_name]

# Opción de imagen estática
use_static = st.checkbox("Usar imagen estática en lugar del pentagrama", value=False, key="chk_static")
static_img_file = None
if use_static:
    static_img_file = st.file_uploader("Sube una imagen (PNG/JPG)", type=["png", "jpg", "jpeg"], key="upl_img")

if st.button("🎶 Generar combo de jazz (MIDI + Visual)", key="btn_generate"):
    cfg = KEY_OPTIONS[key_name]
    prog, total_bars = build_progression(cfg["root_midi"], cfg["mode"], progression_type)
    mel = melody(cfg["root_midi"], cfg["mode"], seed, total_bars)
    bass = walking_bass(prog)
    chords_ev = chord_hits(prog)
    drums = swing_drums(total_bars)

    mid = events_to_midi(mel, bass, chords_ev, drums, bpm, instr_melody)
    midi_bytes = io.BytesIO()
    mid.save(file=midi_bytes); midi_bytes.seek(0)

    # Visual: imagen estática o partitura simplificada
    if use_static and static_img_file is not None:
        st.image(static_img_file, caption="Imagen estática")
    else:
        preview_img = draw_score_preview(mel, key_name, bpm)
        buf = io.BytesIO(); preview_img.save(buf, format="PNG"); buf.seek(0)
        st.image(buf, caption="Vista previa rápida (melodía)")

    st.success(f"¡Listo! 🎷 Combo con progresión {progression_type}.")
    st.download_button("⬇️ Descargar MIDI", data=midi_bytes, file_name="combo.mid", mime="audio/midi")
