import io, random, os
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from mido import Message, MidiFile, MidiTrack, MetaMessage, bpm2tempo

# music21 para partitura completa (MusicXML / PDF si disponible)
from music21 import stream as m21stream, note as m21note, chord as m21chord
from music21 import instrument as m21inst, meter as m21meter, tempo as m21tempo
from music21 import key as m21key, duration as m21duration, metadata as m21metadata

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Generador Jazz Combo", page_icon="🎷")
st.title("🎷 Generador de Combo de Jazz (MIDI + Partitura completa)")

TICKS_PER_BEAT = 480
BEATS_PER_BAR = 4
BARS = 8
TOTAL_BEATS = BEATS_PER_BAR * BARS

KEY_OPTIONS = {
    "Do mayor (C major)": {"root_midi": 60, "mode": "major", "key_sig": "C"},
    "La menor (A minor)": {"root_midi": 57, "mode": "minor", "key_sig": "A minor"},
}

INSTRUMENTS = {
    "Vibráfono": 11,  # GM
    "Trompeta": 56,   # GM
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

def build_progression(root_midi, mode):
    """
    Devuelve una lista de 8 compases, cada uno con un voicing (lista de MIDIs).
    Mayor:   | Cmaj7 | Am7 | Dm7 | G7 | x2
    Menor:   | Am7 | Fmaj7 | Bm7b5 | E7 | x2
    """
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
    return four * 2  # 8 compases

# =========================
# GENERADORES DE EVENTOS
# =========================
def melody(root_midi, mode, seed):
    """
    Melodía diatónica (solo una voz), con rango fijo C4–C6 para que destaque (vibráfono protagonista).
    """
    random.seed(seed)
    deg = 7  # arranca aprox en tónica una octava arriba
    events = []
    t = 0.0
    while t < TOTAL_BEATS:
        dur = random.choice([0.5, 1.0])
        step = random.choice([-2, -1, 1, 2])
        deg += step
        midi = degree_to_midi(root_midi, deg, mode)
        midi = clamp_to_range(midi, 60, 84)  # C4–C6
        if t + dur > TOTAL_BEATS:
            dur = TOTAL_BEATS - t
        events.append((t, dur, midi))
        t += dur
    return events  # [(t_beats, dur_beats, midi)]

def walking_bass(prog):
    """
    Contrabajo en negras: raíz de cada acorde, 1 nota por pulso, dos octavas abajo.
    """
    events = []
    for bar in range(BARS):
        chord = prog[bar]
        root = chord[0] - 24
        for beat in range(BEATS_PER_BAR):
            t = bar*BEATS_PER_BAR + beat
            events.append((t, 1.0, root))
    return events

def chord_hits(prog):
    """
    Acordes de acompañamiento: golpes en 1 y 3 (blancas).
    """
    events = []
    for bar in range(BARS):
        chord = prog[bar]
        for beat in [0, 2]:
            t = bar*BEATS_PER_BAR + beat
            for n in chord:
                events.append((t, 2.0, n))
    return events

def swing_drums():
    """
    Patrón swing muy básico (solo para el MIDI, no lo incluimos en la partitura impresa).
    Ride (51) en patrón; bombo (36) y caja (38) en 2 y 4.
    """
    events = []
    for bar in range(BARS):
        for beat in range(BEATS_PER_BAR):
            t = bar*BEATS_PER_BAR + beat
            # Ride
            if beat in [0,1,2,3]:
                dur = 0.5 if beat % 2 == 0 else 0.25
                events.append((t, dur, 51))
            # 2 y 4
            if beat in [1,3]:
                events.append((t, 0.25, 36))  # bombo
                events.append((t, 0.25, 38))  # caja
    return events

# =========================
# MIDI BUILDER
# =========================
def events_to_midi(melody_ev, bass_ev, chord_ev, drum_ev, bpm, instr_melody):
    mid = MidiFile(ticks_per_beat=TICKS_PER_BEAT)

    # Melody
    mel_track = MidiTrack(); mid.tracks.append(mel_track)
    mel_track.append(MetaMessage('set_tempo', tempo=bpm2tempo(bpm)))
    mel_track.append(Message('program_change', program=instr_melody, channel=0, time=0))

    # Bass
    bass_track = MidiTrack(); mid.tracks.append(bass_track)
    bass_track.append(Message('program_change', program=32, channel=1, time=0))  # Acoustic Bass

    # Chords
    chord_track = MidiTrack(); mid.tracks.append(chord_track)
    chord_track.append(Message('program_change', program=0, channel=2, time=0))  # Acoustic Grand

    # Drums (channel 9)
    drum_track = MidiTrack(); mid.tracks.append(drum_track)

    def add_events(track, events, channel):
        evts = []
        for t, dur, midi in events:
            on = ('on', int(round(t*TICKS_PER_BEAT)), midi)
            off = ('off', int(round((t+dur)*TICKS_PER_BEAT)), midi)
            evts += [on, off]
        # offs antes que ons si coinciden
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
# PARTITURA (PNG PREVIEW SOLO MELODÍA)
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
# PARTITURA COMPLETA (music21)
# =========================
def build_full_score_music21(prog, melody_ev, bpm, key_sig_name, mode):
    """
    Construye una partitura con:
    - Part Melody (Vibraphone por defecto)
    - Part Bass (Contrabass)
    - Part Piano (Acordes en bloque en 1 y 3)
    Devuelve un music21.stream.Score listo para exportar a MusicXML (y PDF si hay backend).
    """
    sc = m21stream.Score()
    sc.insert(0, m21metadata.Metadata())
    sc.metadata.title = "Combo Jazz – I–vi–ii–V"
    sc.metadata.composer = "Generador Streamlit"

    # Tempo y compás
    ts = m21meter.TimeSignature("4/4")
    mm = m21tempo.MetronomeMark(number=bpm)

    # Tonalidad
    k = m21key.Key(key_sig_name)

    # ===== Melody (Vibraphone)
    p_mel = m21stream.Part()
    p_mel.id = "Melody"
    p_mel.insert(0, m21inst.Vibraphone())
    p_mel.append(ts)
    p_mel.append(mm)
    p_mel.append(k)

    for t, dur, midi in melody_ev:
        n = m21note.Note(midi)
        n.duration = m21duration.Duration(dur)  # quarterLength
        p_mel.insert(t, n)

    # ===== Bass (Contrabass)
    p_bass = m21stream.Part()
    p_bass.id = "Bass"
    p_bass.insert(0, m21inst.Contrabass())
    p_bass.append(m21meter.TimeSignature("4/4"))
    p_bass.append(m21tempo.MetronomeMark(number=bpm))
    p_bass.append(k)

    for bar in range(BARS):
        root = prog[bar][0] - 24
        for beat in range(BEATS_PER_BAR):
            t = bar*BEATS_PER_BAR + beat
            n = m21note.Note(root)
            n.duration = m21duration.Duration(1.0)  # negra
            p_bass.insert(t, n)

    # ===== Piano (Chords)
    p_pno = m21stream.Part()
    p_pno.id = "Piano"
    p_pno.insert(0, m21inst.Piano())
    p_pno.append(m21meter.TimeSignature("4/4"))
    p_pno.append(m21tempo.MetronomeMark(number=bpm))
    p_pno.append(k)

    for bar in range(BARS):
        chord_pitches = prog[bar]
        for beat in [0, 2]:
            t = bar*BEATS_PER_BAR + beat
            ch = m21chord.Chord(chord_pitches)
            ch.duration = m21duration.Duration(2.0)  # blanca
            p_pno.insert(t, ch)

    sc.insert(0, p_mel)
    sc.insert(0, p_bass)
    sc.insert(0, p_pno)
    return sc

def write_musicxml_bytes(score_obj):
    """
    Exporta a MusicXML (XML) y devuelve bytes.
    Si el entorno soporta PDF (MuseScore/LilyPond), lo intentamos aparte.
    """
    from tempfile import NamedTemporaryFile
    # MusicXML (.musicxml)
    with NamedTemporaryFile(delete=False, suffix=".musicxml") as tmp:
        fp = tmp.name
    try:
        score_obj.write("musicxml", fp=fp)
        with open(fp, "rb") as f:
            data = f.read()
    finally:
        try:
            os.remove(fp)
        except Exception:
            pass
    return data

def try_write_pdf_bytes(score_obj):
    """
    Intenta exportar PDF via music21. Si no hay backend configurado, devuelve None.
    """
    from tempfile import NamedTemporaryFile
    try:
        with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            fp = tmp.name
        score_obj.write("musicxml.pdf", fp=fp)  # requiere backend (MuseScore/LilyPond)
        with open(fp, "rb") as f:
            pdf_data = f.read()
        try:
            os.remove(fp)
        except Exception:
            pass
        return pdf_data
    except Exception:
        # Backend no disponible en este entorno (normal en Streamlit Cloud)
        return None

# =========================
# UI
# =========================
col1, col2, col3 = st.columns(3)
with col1:
    key_name = st.selectbox("Tonalidad", list(KEY_OPTIONS.keys()))
with col2:
    bpm = st.slider("BPM", 60, 140, 100)
with col3:
    seed = st.number_input("Semilla", 0, 99999, 42)

mel_instr_name = st.radio("Instrumento de la melodía (MIDI)", list(INSTRUMENTS.keys()), index=0)
instr_melody = INSTRUMENTS[mel_instr_name]

if st.button("🎶 Generar combo de jazz (MIDI + Partitura completa)"):
    cfg = KEY_OPTIONS[key_name]
    prog = build_progression(cfg["root_midi"], cfg["mode"])
    mel = melody(cfg["root_midi"], cfg["mode"], seed)
    bass = walking_bass(prog)
    chords_ev = chord_hits(prog)
    drums = swing_drums()

    # MIDI
    mid = events_to_midi(mel, bass, chords_ev, drums, bpm, instr_melody)
    midi_bytes = io.BytesIO()
    mid.save(file=midi_bytes); midi_bytes.seek(0)

    # Partitura completa (MusicXML + intento de PDF)
    score = build_full_score_music21(prog, mel, bpm, KEY_OPTIONS[key_name]["key_sig"], cfg["mode"])
    mxl_bytes = write_musicxml_bytes(score)
    pdf_bytes = try_write_pdf_bytes(score)

    # Preview PNG (melodía)
    preview_img = draw_score_preview(mel, key_name, bpm)
    buf = io.BytesIO(); preview_img.save(buf, format="PNG"); buf.seek(0)

    st.success("¡Listo! 🎷 Partitura completa (MusicXML) + MIDI generado.")
    st.image(buf, caption="Vista previa rápida (melodía)")

    st.download_button("⬇️ Descargar MIDI", data=midi_bytes, file_name="combo.mid", mime="audio/midi")
    st.download_button("⬇️ Descargar MusicXML", data=mxl_bytes, file_name="partitura.musicxml", mime="application/vnd.recordare.musicxml+xml")

    if pdf_bytes:
        st.download_button("⬇️ Descargar PDF (si disponible)", data=pdf_bytes, file_name="partitura.pdf", mime="application/pdf")
    else:
        st.info("Para exportar PDF automáticamente, instala MuseScore o LilyPond en el entorno y configura music21. Mientras tanto, puedes abrir el MusicXML en MuseScore y exportar a PDF.")
