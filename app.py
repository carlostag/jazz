import io, math, random
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from scipy.io import wavfile
from mido import Message, MidiFile, MidiTrack, MetaMessage, bpm2tempo

# =========================
# CONFIG BÁSICA
# =========================
st.set_page_config(page_title="Melodías Diatónicas (Simple)", page_icon="🎼", layout="centered")
st.title("🎼 Generador de Melodías (diatónico, sin acordes)")

SR = 44100
TICKS_PER_BEAT = 480
BEATS_PER_BAR = 4
BARS = 8          # frase de 8 compases
TOTAL_BEATS = BEATS_PER_BAR * BARS

# Solo Do mayor y La menor para notación limpia sin alteraciones
KEY_OPTIONS = {
    "Do mayor (C major)": {"root_midi": 60, "mode": "major"},  # C4
    "La menor (A minor)": {"root_midi": 57, "mode": "minor"},  # A3
}

# Escalas diatónicas (en semitonos desde la tónica)
MAJOR_STEPS = [0, 2, 4, 5, 7, 9, 11]   # 1 2 3 4 5 6 7
MINOR_STEPS = [0, 2, 3, 5, 7, 8, 10]   # 1 2 b3 4 5 b6 b7

# Rango cómodo para melodía (evita saltos fuera del pentagrama)
MIDI_MIN = 60  # C4
MIDI_MAX = 79  # G5

# =========================
# UTILIDADES MUSICALES
# =========================
def scale_steps(mode):
    return MAJOR_STEPS if mode == "major" else MINOR_STEPS

def degree_to_midi(root_midi, degree, mode):
    """degree: entero (puede subir/bajar de octava)."""
    steps = scale_steps(mode)
    diatonic = steps[degree % 7] + 12 * (degree // 7)
    return root_midi + diatonic

def clamp_to_range(m):
    while m < MIDI_MIN: m += 12
    while m > MIDI_MAX: m -= 12
    return m

# =========================
# GENERADOR RÍTMICO
# =========================
def generate_bar_rhythm(beats=4):
    """Crea una lista de duraciones que suman 'beats' (en negra=1, corchea=0.5)."""
    durs = []
    remaining = beats
    # Preferir negras y corcheas, con alguna figura más larga
    choices = [0.5, 1.0, 1.5, 2.0]
    weights = [0.35, 0.45, 0.10, 0.10]
    while remaining > 0:
        # Asegurar cierre exacto
        opts = [d for d in choices if d <= remaining + 1e-6]
        if not opts:
            durs.append(remaining)
            break
        d = random.choices(opts, weights=[weights[choices.index(x)] for x in opts])[0]
        durs.append(d)
        remaining = round(remaining - d, 3)
    return durs

# =========================
# GENERADOR MELÓDICO DIATÓNICO
# =========================
def generate_melody(key_cfg, bpm, seed):
    random.seed(seed); np.random.seed(seed)

    root_midi = key_cfg["root_midi"]
    mode = key_cfg["mode"]

    # Estructura por grados relativos a la tónica (empieza y termina en 1ª)
    # Centro cómodo: alrededor de 8-10 grados (≈ una octava sobre tónica)
    start_deg = 7  # tónica una octava arriba aprox.
    deg = start_deg

    # Objetivos de frase: clímax hacia compás 4 y cierre en tónica compás 8
    target_mid = start_deg + 3  # subir un poco
    target_climax = start_deg + 5  # pico moderado

    events = []  # (t0_beats, dur_beats, midi)

    t = 0.0
    for bar in range(BARS):
        bar_rhythm = generate_bar_rhythm(BEATS_PER_BAR)
        for i, dur in enumerate(bar_rhythm):
            # Elegir siguiente movimiento en grados
            # Paso conjunto predominante, saltos ocasionales con resolución contraria
            if random.random() < 0.80:
                step = random.choices([-1, 0, 1, 2, -2], weights=[0.28, 0.12, 0.40, 0.10, 0.10])[0]
            else:
                step = random.choice([3, -3, 4, -4])  # salto ocasional

            # Acercarse a objetivos (bar 0-3 subir; 3-4 clímax; 5-7 descender)
            if bar < 3 and deg < target_mid and random.random() < 0.6:
                step = max(1, step)  # favorece subir
            if bar == 3 and deg < target_climax and random.random() < 0.7:
                step = abs(step)     # subir
            if bar >= 5 and deg > start_deg and random.random() < 0.65:
                step = -abs(step)    # bajar

            # Aplicar y limitar rango (en MIDI)
            cand_deg = deg + step
            cand_midi = degree_to_midi(root_midi, cand_deg, mode)
            cand_midi = clamp_to_range(cand_midi)

            # Evitar repeticiones largas
            if events and abs(cand_midi - events[-1][2]) == 0 and random.random() < 0.5:
                # fuerza un paso mínimo
                cand_deg += random.choice([-1, 1])
                cand_midi = clamp_to_range(degree_to_midi(root_midi, cand_deg, mode))

            # Semicadencia al final del 4º compás: dominante (G en mayor, E/G en menor natural)
            is_last_note_of_bar4 = (bar == 3) and (i == len(bar_rhythm) - 1)
            if is_last_note_of_bar4:
                if mode == "major":
                    cand_midi = clamp_to_range(root_midi + 7)   # grado 5
                else:
                    cand_midi = clamp_to_range(root_midi + 7)   # 5 también funciona bien
                # alargar si es muy corta
                if dur < 1.0: dur = 1.0

            # Cadencia final en tónica (último compás, última nota)
            is_last_note = (bar == BARS - 1) and (i == len(bar_rhythm) - 1)
            if is_last_note:
                cand_midi = clamp_to_range(root_midi)  # tónica
                dur = max(dur, 2.0)  # sostener cierre

            events.append((t, dur, cand_midi))
            deg = cand_deg
            t += dur

    # Ajuste exacto a TOTAL_BEATS
    if t < TOTAL_BEATS:
        events.append((t, TOTAL_BEATS - t, clamp_to_range(root_midi)))

    return events  # [(t_beats, dur_beats, midi)]

# =========================
# SÍNTESIS MONOFÓNICA SUAVE
# =========================
def adsr(n, sr, a=0.01, d=0.06, s=0.75, r=0.08):
    A = int(a*sr); D = int(d*sr); R = int(r*sr)
    S = max(n - (A+D+R), 0)
    env = np.zeros(n, np.float32)
    if A>0: env[:A] = np.linspace(0, 1, A, endpoint=False)
    if D>0: env[A:A+D] = np.linspace(1, s, D, endpoint=False)
    if S>0: env[A+D:A+D+S] = s
    if R>0: env[A+D+S:A+D+S+R] = np.linspace(s, 0, R, endpoint=True)
    return env

def synth_melody(events, bpm, sr=SR):
    q = 60.0 / bpm
    total_time = sum(d for _, d, _ in events) * q
    y = np.zeros(int(total_time * sr) + sr//2, np.float32)

    prev_end = 0
    for t_beats, dur_beats, midi in events:
        t0 = int(t_beats * q * sr)
        dur = dur_beats * q
        n = max(1, int(dur * sr))
        f = 440.0 * (2.0 ** ((midi - 69) / 12.0))

        # seno + triángulo suave
        t = np.linspace(0, dur, n, endpoint=False, dtype=np.float32)
        x = 0.7*np.sin(2*np.pi*f*t) + 0.3*(2*np.abs(2*(t*f - np.floor(0.5 + t*f))) - 1.0)  # tri

        env = adsr(n, sr, a=0.006, d=0.05, s=0.82, r=0.07)
        note = (x * env).astype(np.float32)

        # crossfade leve si solapan (legato)
        s = t0; e = t0 + n
        if s < prev_end:
            overlap = prev_end - s
            if overlap > 0:
                fade = np.linspace(0.0, 1.0, overlap, dtype=np.float32)
                y[s:prev_end] = y[s:prev_end]*(1.0 - 0.35*fade) + note[:overlap]*(0.35*fade)
                y[prev_end:e] += note[overlap:overlap + (e-prev_end)]
            else:
                y[s:e] += note
        else:
            y[s:e] += note
        prev_end = max(prev_end, e)

    # Normalizar suave
    peak = float(np.max(np.abs(y)) + 1e-9)
    y *= (0.9 / peak)
    return y

# =========================
# EXPORT MIDI (delta times correctos)
# =========================
def events_to_midi(events, bpm):
    mid = MidiFile(ticks_per_beat=TICKS_PER_BEAT)
    track = MidiTrack()
    mid.tracks.append(track)
    track.append(MetaMessage('set_tempo', tempo=bpm2tempo(bpm)))

    # Construir eventos on/off en ticks absolutos
    evts = []
    for t_beats, dur_beats, midi in events:
        on_tick = int(round(t_beats * TICKS_PER_BEAT))
        off_tick = int(round((t_beats + dur_beats) * TICKS_PER_BEAT))
        evts.append(('on', on_tick, midi))
        evts.append(('off', off_tick, midi))
    evts.sort(key=lambda e: (e[1], 0 if e[0]=='off' else 1))  # offs antes que ons si simultáneos

    cursor = 0
    for typ, tick, m in evts:
        delta = tick - cursor
        cursor = tick
        if typ == 'on':
            track.append(Message('note_on', note=int(m), velocity=84, time=delta))
        else:
            track.append(Message('note_off', note=int(m), velocity=64, time=delta))
    return mid

# =========================
# DIBUJO DE PARTITURA (PNG)
# =========================
STAFF_LINE_SP = 12
LEFT = 40
RIGHT = 1020
TOP = 60
BOTTOM = TOP + 4*STAFF_LINE_SP

def midi_to_y(m):
    """E4=64 es línea inferior. Cada semitono = medio espacio de línea."""
    pos = (m - 64) * 0.5  # 2 semitonos = 1 línea/espacio
    y = BOTTOM - pos * STAFF_LINE_SP
    return y

def draw_ledger_lines(draw, x, y, m):
    # Dibuja líneas adicionales cada "línea" por fuera del pentagrama
    top_line_y = TOP
    bot_line_y = BOTTOM
    line_every = STAFF_LINE_SP  # distancia entre líneas

    if y < top_line_y:
        k = 0
        while top_line_y - k*line_every > y - 2:
            y_line = top_line_y - k*line_every
            draw.line([(x-10, y_line), (x+10, y_line)], fill=(20,20,20), width=2)
            k += 1
    elif y > bot_line_y:
        k = 0
        while bot_line_y + k*line_every < y + 2:
            y_line = bot_line_y + k*line_every
            draw.line([(x-10, y_line), (x+10, y_line)], fill=(20,20,20), width=2)
            k += 1

def draw_staff_image(events, bpm, key_name):
    # Layout horizontal simple: repartir por compases
    width = 1080
    height = 260
    img = Image.new("RGB", (width, height), (250, 250, 250))
    d = ImageDraw.Draw(img)

    # Título
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 20)
    except:
        font = None
    d.text((LEFT, 20), f"Melodía – {key_name} – {bpm} bpm", fill=(30,30,30), font=font)

    # Pentagrama (5 líneas)
    for i in range(5):
        y = TOP + i * STAFF_LINE_SP
        d.line([(LEFT, y), (RIGHT, y)], fill=(20,20,20), width=2)

    # Clave de sol simple
    d.ellipse([LEFT-18, TOP-10, LEFT+10, TOP+18], outline=(20,20,20), width=2)

    # Compases
    bar_width = (RIGHT - LEFT - 20) / BARS
    for b in range(BARS + 1):
        xbar = LEFT + 10 + int(b * bar_width)
        d.line([(xbar, TOP), (xbar, BOTTOM)], fill=(60,60,60), width=2)

    # Notas
    # Distribuir notas según su t0 relativo dentro de cada compás
    for t_beats, dur_beats, midi in events:
        bar = int(t_beats // BEATS_PER_BAR)
        t_in_bar = t_beats - bar * BEATS_PER_BAR
        x0 = LEFT + 10 + bar * bar_width
        x1 = LEFT + 10 + (bar + 1) * bar_width
        x = int(np.interp(t_in_bar, [0, BEATS_PER_BAR], [x0 + 12, x1 - 12]))

        y = midi_to_y(midi)
        # Cabeza
        rx, ry = 7, 5
        d.ellipse([x - rx, y - ry, x + rx, y + ry], fill=(10,10,10))

        # Plica (arriba si por debajo de la tercera línea)
        midline_y = TOP + 2 * STAFF_LINE_SP
        if y > midline_y:
            # plica arriba
            d.line([(x + rx, y), (x + rx, y - 28)], fill=(10,10,10), width=2)
        else:
            # plica abajo
            d.line([(x - rx, y), (x - rx, y + 28)], fill=(10,10,10), width=2)

        # Bandera si es corchea (dur < 1.0 beat)
        if dur_beats < 1.0 - 1e-6:
            if y > midline_y:
                d.line([(x + rx, y - 28), (x + rx + 10, y - 18)], fill=(10,10,10), width=2)
            else:
                d.line([(x - rx, y + 28), (x - rx - 10, y + 18)], fill=(10,10,10), width=2)

        # Líneas adicionales
        draw_ledger_lines(d, x, y, midi)

    # Compás final más grueso
    x_end = LEFT + 10 + BARS * bar_width
    d.line([(x_end-2, TOP), (x_end-2, BOTTOM)], fill=(10,10,10), width=4)

    return img

# =========================
# UI
# =========================
col1, col2, col3 = st.columns(3)
with col1:
    key_name = st.selectbox("Tonalidad", list(KEY_OPTIONS.keys()), index=0)
with col2:
    bpm = st.slider("Tempo (BPM)", 60, 140, 88, step=1)
with col3:
    seed = st.number_input("Semilla", min_value=0, max_value=999999, value=42, step=1)

if st.button("🎶 Generar melodía"):
    cfg = KEY_OPTIONS[key_name]
    events = generate_melody(cfg, bpm, seed)
    audio = synth_melody(events, bpm)

    # WAV
    wav_bytes = io.BytesIO()
    wav = np.int16(np.clip(audio, -1.0, 1.0) * 32767)
    wavfile.write(wav_bytes, SR, wav); wav_bytes.seek(0)

    # MIDI
    mid = events_to_midi(events, bpm)
    midi_bytes = io.BytesIO()
    mid.save(file=midi_bytes); midi_bytes.seek(0)

    # Partitura
    score_img = draw_staff_image(events, bpm, key_name)
    img_buf = io.BytesIO()
    score_img.save(img_buf, format="PNG"); img_buf.seek(0)

    st.success("¡Hecho! Melodía diatónica generada sin acordes de fondo.")
    st.image(img_buf, caption="Partitura (melodía, 8 compases)", use_container_width=True)
    st.audio(wav_bytes, format="audio/wav")
    st.download_button("⬇️ Descargar WAV", data=wav_bytes, file_name="melodia.wav", mime="audio/wav")
    st.download_button("⬇️ Descargar MIDI", data=midi_bytes, file_name="melodia.mid", mime="audio/midi")
