import io, random, math
import streamlit as st
from PIL import Image, ImageDraw
from mido import Message, MidiFile, MidiTrack, MetaMessage, bpm2tempo
from music21 import stream as m21stream, note as m21note, chord as m21chord
from music21 import instrument as m21inst, meter as m21meter, tempo as m21tempo
from music21 import key as m21key, duration as m21duration, metadata as m21metadata

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Generador Jazz Combo", page_icon="🎷")
st.title("🎷 Generador de Combo de Jazz (MIDI + Visual)")

TICKS_PER_BEAT = 480
BEATS_PER_BAR = 4
TARGET_DURATION = 60  # segundos

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
        base = [
            [root_midi, root_midi+4, root_midi+7, root_midi+11],       # Cmaj7
            [root_midi+9, root_midi+12, root_midi+16, root_midi+19],   # Am7
            [root_midi+2, root_midi+5, root_midi+9, root_midi+12],     # Dm7
            [root_midi+7, root_midi+11, root_midi+14, root_midi+17],   # G7
        ] if mode == "major" else [
            [root_midi, root_midi+3, root_midi+7, root_midi+10],       # Am7
            [root_midi+8, root_midi+12, root_midi+15, root_midi+19],   # Fmaj7
            [root_midi+2, root_midi+5, root_midi+10, root_midi+14],    # Bm7b5
            [root_midi+7, root_midi+11, root_midi+14, root_midi+17],   # E7
        ]
        return base
    elif progression_type == "Blues en C":
        I7  = [root_midi, root_midi+4, root_midi+7, root_midi+10]
        IV7 = [root_midi+5, root_midi+9, root_midi+12, root_midi+15]
        V7  = [root_midi+7, root_midi+11, root_midi+14, root_midi+17]
        return [I7, I7, I7, I7, IV7, IV7, I7, I7, V7, IV7, I7, V7]

def extend_to_duration(prog, bpm):
    bar_len = (BEATS_PER_BAR * 60.0) / bpm  # duración compás
    bars_needed = int(round(TARGET_DURATION / bar_len))
    prog_extended = (prog * ((bars_needed // len(prog)) + 1))[:bars_needed]
    return prog_extended, bars_needed

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
        midi = clamp_to_range(midi, 60, 84)
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
# VISUAL RANDOM (polígonos o fractales)
# =========================
def random_polygon_image(width=800, height=600, n_sides=5):
    img = Image.new("RGB", (width, height), (255,255,255))
    d = ImageDraw.Draw(img)
    pts = [(random.randint(50, width-50), random.randint(50, height-50)) for _ in range(n_sides)]
    d.polygon(pts, fill=(random.randint(0,255), random.randint(0,255), random.randint(0,255)))
    return img

def fractal_tree(draw, x, y, angle, depth, length):
    if depth == 0: return
    x2 = x + int(math.cos(math.radians(angle)) * length)
    y2 = y - int(math.sin(math.radians(angle)) * length)
    draw.line((x, y, x2, y2), fill=(0,0,0), width=2)
    fractal_tree(draw, x2, y2, angle-20, depth-1, length*0.7)
    fractal_tree(draw, x2, y2, angle+20, depth-1, length*0.7)

def fractal_image(width=800, height=600):
    img = Image.new("RGB", (width,height), (255,255,255))
    d = ImageDraw.Draw(img)
    fractal_tree(d, width//2, height-50, -90, 8, 80)
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

visual_type = st.radio("Visual", ["Pentagrama simplificado", "Polígono aleatorio", "Fractal"], key="radio_visual")

if st.button("🎶 Generar combo de jazz (60s)", key="btn_generate"):
    cfg = KEY_OPTIONS[key_name]
    prog_base = build_progression(cfg["root_midi"], cfg["mode"], progression_type)
    prog, total_bars = extend_to_duration(prog_base, bpm)

    mel = melody(cfg["root_midi"], cfg["mode"], seed, total_bars)
    bass = walking_bass(prog)
    chords_ev = chord_hits(prog)
    drums = swing_drums(total_bars)

    mid = events_to_midi(mel, bass, chords_ev, drums, bpm, instr_melody)
    midi_bytes = io.BytesIO()
    mid.save(file=midi_bytes); midi_bytes.seek(0)

    if visual_type == "Pentagrama simplificado":
        img = Image.new("RGB", (1000,220), (255,255,255))
        d = ImageDraw.Draw(img)
        d.text((40,10), f"Melodía ({key_name}, {bpm} bpm)", fill=(0,0,0))
        for i in range(5):
            y = 120+i*12; d.line([(40,y),(960,y)], fill=(0,0,0))
        x=60
        for t,dur,m in mel[:32]:
            y = 120+24-(m-60)*6
            d.ellipse([x-5,y-5,x+5,y+5], fill=(0,0,0))
            x+=20
    elif visual_type == "Polígono aleatorio":
        img = random_polygon_image()
    else:
        img = fractal_image()

    buf = io.BytesIO(); img.save(buf, format="PNG"); buf.seek(0)

    st.success(f"¡Listo! 🎷 Combo de {total_bars} compases ≈ {TARGET_DURATION}s")
    st.image(buf, caption="Visual generado")
    st.download_button("⬇️ Descargar MIDI", data=midi_bytes, file_name="combo.mid", mime="audio/midi")
