import os, math, random, io, json
import numpy as np
import streamlit as st
from tempfile import NamedTemporaryFile
from PIL import Image, ImageDraw, ImageFont
from scipy.io import wavfile
from moviepy.editor import VideoClip, AudioFileClip
from mido import Message, MidiFile, MidiTrack, MetaMessage, bpm2tempo
from langchain_groq import ChatGroq

# =========================
# STREAMLIT + LLM CONFIG
# =========================
st.set_page_config(page_title="Generador Musical + Partitura", page_icon="🎼", layout="centered")
st.title("🎼 Generador musical con partitura + LLM (Groq)")

# Clave Groq desde .streamlit/secrets.toml
# [secrets]
# GROQ_API_KEY="tu_clave"
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", None)

llm = None
if GROQ_API_KEY:
    try:
        llm = ChatGroq(model_name="llama-3.3-70b-versatile", api_key=GROQ_API_KEY)
    except Exception as e:
        st.warning(f"No se pudo inicializar el modelo Groq: {e}")

# =========================
# PARÁMETROS BASE
# =========================
W, H = 1080, 1920
FPS = 30
SR = 44100

DEFAULTS = {
    "duration_sec": 45,
    "bpm": 90,
    "root": "C",
    "key_low": 48,   # C3
    "key_high": 76,  # E5
}

NOTE_TO_SEMITONE = {
    "C":0,"C#":1,"Db":1,"D":2,"D#":3,"Eb":3,
    "E":4,"F":5,"F#":6,"Gb":6,"G":7,"G#":8,"Ab":8,
    "A":9,"A#":10,"Bb":10,"B":11
}

PC_COLORS = {
    0:(255,140,140), 1:(255,170,120), 2:(255,200,120),
    3:(240,220,120), 4:(200,240,130), 5:(150,250,160),
    6:(140,230,220), 7:(140,190,255), 8:(170,160,255),
    9:(210,150,255),10:(240,140,220),11:(255,140,180)
}

# =========================
# UTILIDADES MUSICALES
# =========================
def midi_from_note(note_name, octave=4):
    return 12*(octave+1)+NOTE_TO_SEMITONE[note_name]

def freq_from_midi(m):
    return 440.0*(2.0**((m-69)/12.0))

def clamp_vis_to_range(m, low=DEFAULTS["key_low"], high=DEFAULTS["key_high"]):
    while m < low:  m += 12
    while m > high: m -= 12
    return m

def adsr_envelope(n_samples, sr, attack=0.01, decay=0.2, sustain=0.5, release=0.2):
    a = int(attack * sr); d = int(decay * sr); r = int(release * sr)
    total = a + d + r
    if total > n_samples and total > 0:
        factor = n_samples / total
        a = max(1, int(a * factor))
        d = max(1, int(d * factor))
        r = max(1, int(r * factor))
    s = max(n_samples - (a + d + r), 0)
    env = np.zeros(n_samples, dtype=np.float32)
    if a > 0: env[:a] = np.linspace(0, 1, a, endpoint=False)
    start = a
    if d > 0 and start < n_samples:
        end = min(start + d, n_samples); env[start:end] = np.linspace(1, sustain, end-start, endpoint=False)
    start = a + d
    if s > 0 and start < n_samples:
        end = min(start + s, n_samples); env[start:end] = sustain
    start = a + d + s
    if r > 0 and start < n_samples:
        end = min(start + r, n_samples); env[start:end] = np.linspace(sustain, 0, end-start, endpoint=True)
    return env

def apply_fades(y, sr, ms=3):
    n = len(y)
    f = max(1, int((ms/1000.0) * sr))
    if f*2 >= n:
        f = n//4 if n >= 4 else 1
    if f > 1:
        win = 0.5 - 0.5*np.cos(np.linspace(0, math.pi, f))
        y[:f] *= win; y[-f:] *= win[::-1]
    return y

def mallet_note(freq, dur, sr):
    n = int(sr * dur)
    if n <= 0: return np.zeros(0, dtype=np.float32)
    t = np.linspace(0, dur, n, endpoint=False)
    nyq = sr/2.0
    mults = [1.0, 2.0, 3.0, 4.0]
    amps  = [1.00, 0.35, 0.20, 0.12]
    x = np.zeros(n, dtype=np.float32)
    for m,a in zip(mults, amps):
        f = freq*m
        if f < nyq: x += a*np.sin(2*np.pi*f*t)
    env = adsr_envelope(n, sr, attack=0.008, decay=0.18, sustain=0.45, release=0.18)
    y = apply_fades(x * env, sr, ms=3)
    # LP 1er orden ~7kHz
    rc = 1.0 / (2*math.pi*7000.0); dt = 1.0 / sr; alp = dt/(rc+dt)
    z = np.zeros_like(y)
    for i in range(1, n):
        z[i] = z[i-1] + alp*(y[i]-z[i-1])
    return z

def bass_sine(freq, dur, sr):
    n = int(sr*dur)
    if n <= 0: return np.zeros(0, dtype=np.float32)
    t = np.linspace(0, dur, n, endpoint=False)
    x = np.sin(2*np.pi*freq*t).astype(np.float32)
    env = adsr_envelope(n, sr, attack=0.005, decay=0.12, sustain=0.65, release=0.12)
    y = apply_fades(x*env, sr, ms=3)
    return y

def chord_voicing_simple(midi_root, quality="maj"):
    return [midi_root, midi_root+4, midi_root+7] if quality=="maj" else [midi_root, midi_root+3, midi_root+7]

def simple_progression(root="C"):
    I  = midi_from_note(root, 4)    # C4=60
    V  = I + 7                      # G
    vi = I + 9                      # Am
    IV = I + 5                      # F
    return [(I,"maj"), (V,"maj"), (vi,"min"), (IV,"maj")]

def generate_track(duration_s=DEFAULTS["duration_sec"], sr=SR, bpm=DEFAULTS["bpm"], root=DEFAULTS["root"]):
    q = 60.0 / bpm
    beats_per_bar = 4
    total_beats = int(duration_s / q)
    total_bars  = max(1, total_beats // beats_per_bar)
    progression = simple_progression(root)

    audio = np.zeros(int(sr*duration_s), dtype=np.float32)
    # notes: (t0, dur, midi, role)
    notes = []

    def chord_scale_for(ch_root, quality):
        if quality == "maj":
            return [ch_root, ch_root+2, ch_root+4, ch_root+7, ch_root+9]
        else:
            return [ch_root, ch_root+3, ch_root+5, ch_root+7, ch_root+10]

    last_mel = None
    for bar in range(total_bars):
        chord_root, quality = progression[bar % len(progression)]
        t_bar = bar * beats_per_bar * q
        last_bar = (bar == total_bars-1)

        dur_ch = beats_per_bar * q
        chord = chord_voicing_simple(chord_root, quality)
        for m in chord:
            f = freq_from_midi(m)
            y = 0.22 * mallet_note(f, dur_ch, sr)
            s = int(t_bar*sr); e = min(len(audio), s+len(y))
            if s < e:
                seg = min(e-s, len(y))
                audio[s:s+seg] += y[:seg]
                notes.append((t_bar, dur_ch, m, "chord"))

        if not last_bar:
            f = freq_from_midi(chord_root - 24)
            y = 0.32 * bass_sine(f, dur_ch, sr)
            s = int(t_bar*sr); e = min(len(audio), s+len(y))
            if s < e:
                seg = min(e-s, len(y))
                audio[s:s+seg] += y[:seg]
                notes.append((t_bar, dur_ch, chord_root-24, "bass"))

        if not last_bar:
            scale = chord_scale_for(chord_root, quality)
            subbeat = 0.0
            goal = random.choice([chord_root, chord_root+4, chord_root+7])
            while subbeat < beats_per_bar:
                if random.random() < 0.82:
                    if last_mel is None:
                        m = random.choice(scale)
                    else:
                        pool = []
                        for n in scale:
                            diff = abs(n - last_mel)
                            if diff == 0: continue
                            dist_goal = abs(n - goal)
                            weight = 1.0/(1.0 + dist_goal)
                            if diff <= 5: pool.append((n, weight))
                        m = random.choices([c for c,_ in pool], weights=[w for _,w in pool])[0] if pool else random.choice(scale)
                    last_mel = m

                    dur_beats = random.choices([0.25,0.5,1.0,1.5],[0.22,0.40,0.30,0.08])[0]
                    dur_beats = min(dur_beats, beats_per_bar - subbeat)
                    dur_time = dur_beats * q
                    t_mel = t_bar + subbeat * q

                    f = freq_from_midi(m)
                    y = 0.40 * mallet_note(f, dur_time, sr)
                    s = int(t_mel*sr); e = min(len(audio), s+len(y))
                    if s < e:
                        seg = min(e-s, len(y))
                        audio[s:s+seg] += y[:seg]
                        notes.append((t_mel, dur_time, m, "mel"))
                    subbeat += dur_beats
                else:
                    subbeat += 0.5

    peak = np.max(np.abs(audio)) + 1e-12
    audio = audio * (0.89 / peak)
    return audio, notes, q, beats_per_bar

# =========================
# MIDI EXPORT
# =========================
def notes_to_midi(notes, bpm=90, ticks_per_beat=480):
    mid = MidiFile(ticks_per_beat=ticks_per_beat)
    track = MidiTrack(); mid.tracks.append(track)
    track.append(MetaMessage('set_tempo', tempo=bpm2tempo(bpm)))
    last_tick = 0
    # Convert each note (mel, chord, bass) into on/off
    events = []
    for t0, dur, m, role in notes:
        on = ('on', int(round(t0 * bpm * ticks_per_beat / 60.0)), m, role)
        off = ('off', int(round((t0+dur) * bpm * ticks_per_beat / 60.0)), m, role)
        events.append(on); events.append(off)
    events.sort(key=lambda e: e[1])
    cursor = 0
    for typ, tick, m, role in events:
        delta = tick - cursor; cursor = tick
        if typ == 'on':
            track.append(Message('note_on', note=int(m), velocity=80, time=delta))
        else:
            track.append(Message('note_off', note=int(m), velocity=64, time=delta))
    return mid

# =========================
# PARTITURA SENCILLA (imagen)
# =========================
STAFF_LINE_SPACING = 14
STAFF_MARGIN = 28
STAFF_WIDTH = 980
NOTE_SPACING = 34  # distancia horizontal entre notas
# Mapeo MIDI -> línea/espacio en clave de Sol aprox. (E4 como línea 1)
def midi_to_staff_pos(m):
    # E4 (64) = línea 1; cada 1 paso = semitono ~ 1/2 línea (aprox visual)
    return (m - 64) * 0.5

def draw_staff_image(melody_notes, title="Melodía", bpm=90, root="C"):
    # melody_notes: [(t0, dur, midi)] solo role "mel"
    # Ordenar por tiempo
    melody_notes = sorted(melody_notes, key=lambda x: x[0])
    # Crear imagen
    lines = 5
    width = STAFF_MARGIN*2 + STAFF_WIDTH
    height = 320
    img = Image.new("RGB", (width, height), (245,245,245))
    d = ImageDraw.Draw(img)

    # Título
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 24)
    except:
        font = None
    d.text((STAFF_MARGIN, 8), f"{title}  (bpm {bpm}, tonalidad {root})", fill=(20,20,20), font=font)

    # Pentagrama
    top = 80
    for i in range(lines):
        y = top + i*STAFF_LINE_SPACING
        d.line([(STAFF_MARGIN, y), (STAFF_MARGIN+STAFF_WIDTH, y)], fill=(30,30,30), width=2)

    # Clave de Sol simplificada (G swirl fake)
    d.ellipse([STAFF_MARGIN-6, top-8, STAFF_MARGIN+18, top+16], outline=(30,30,30), width=2)

    # Notas (solo cabezas con plica simple; negra/ corchea)
    x = STAFF_MARGIN + 30
    for i, (t0, dur, m) in enumerate(melody_notes):
        # Posición vertical: 0 = en E4 (línea 1); hacia arriba disminuye y
        pos = midi_to_staff_pos(m)
        y_center = top + (4*STAFF_LINE_SPACING) - pos*STAFF_LINE_SPACING/1.0
        # Cabeza de nota
        rx, ry = 8, 6
        d.ellipse([x-rx, y_center-ry, x+rx, y_center+ry], fill=(10,10,10))
        # Plica
        d.line([(x+rx, y_center), (x+rx, y_center-28)], fill=(10,10,10), width=2)
        # Indicador de duración: corchea si dur < negra
        # Aprox: negra ~ 1 beat; q = 60/bpm
        beat = 60.0/ max(1, bpm)
        if dur < beat * 0.75:
            # “banderín” simple
            d.line([(x+rx, y_center-28), (x+rx+10, y_center-18)], fill=(10,10,10), width=2)
        # Avanzar
        x += NOTE_SPACING

    return img

# =========================
# UI: CONTROL POR LLM
# =========================
st.subheader("🧠 Describe el estilo que quieres (el LLM ajusta los parámetros)")
desc = st.text_area("Ejemplos: 'más rápido y alegre en Sol mayor', 'balada lenta y suave', 'estilo épico en La menor, 60s'")

def llm_to_params(description):
    # Devuelve bpm, root, duration_sec a partir de la descripción natural
    # Fallback a defaults si no hay LLM
    base = DEFAULTS.copy()
    if not llm or not description.strip():
        return base["bpm"], base["root"], base["duration_sec"]
    prompt = f"""
Eres un asistente musical. A partir de esta descripción, sugiere parámetros JSON compactos (sin texto extra):
campos: bpm (entero 60-180), root (nota en C,C#,D,Db,...), duration_sec (entre 20 y 120).
Descripción: "{description}"
Responde SOLO con un JSON. Ej: {{"bpm": 110, "root": "A", "duration_sec": 40}}
"""
    try:
        out = llm.invoke(prompt)
        txt = out.content if hasattr(out, "content") else str(out)
        data = json.loads(txt)
        bpm = int(data.get("bpm", DEFAULTS["bpm"]))
        root = str(data.get("root", DEFAULTS["root"])).replace("♭","b").replace("♯","#")
        dur = int(data.get("duration_sec", DEFAULTS["duration_sec"]))
        bpm = min(180, max(60, bpm))
        dur = min(120, max(20, dur))
        if root not in NOTE_TO_SEMITONE: root = DEFAULTS["root"]
        return bpm, root, dur
    except Exception:
        return base["bpm"], base["root"], base["duration_sec"]

col1, col2, col3 = st.columns(3)
with col1:
    seed = st.number_input("Semilla aleatoria", min_value=0, max_value=999999, value=42, step=1)
with col2:
    gen_video = st.checkbox("Crear vídeo piano-roll (MP4)", value=False)
with col3:
    only_melody_in_score = st.checkbox("Partitura solo melodía", value=True)

if st.button("🎵 Generar música"):
    random.seed(seed); np.random.seed(seed)

    bpm, root, duration_sec = llm_to_params(desc)
    with st.spinner("Generando audio y notas..."):
        audio, notes, q, beats_per_bar = generate_track(duration_s=duration_sec, bpm=bpm, root=root)

    # WAV en memoria
    wav_bytes = io.BytesIO()
    wav = np.int16(np.clip(audio, -1.0, 1.0) * 32767)
    wavfile.write(wav_bytes, SR, wav); wav_bytes.seek(0)

    # MIDI
    mid = notes_to_midi(notes, bpm=bpm)
    midi_bytes = io.BytesIO()
    mid.save(file=midi_bytes); midi_bytes.seek(0)

    # Selección para partitura (melodía)
    melody = [(t0, dur, m) for (t0, dur, m, role) in notes if role == "mel"]
    if not melody:
        # fallback: usa las notas de acorde más altas
        chords_tops = []
        for (t0, dur, m, role) in notes:
            if role == "chord":
                chords_tops.append((t0, dur, m))
        melody = chords_tops[:16]

    score_img = draw_staff_image(melody, title="Generado", bpm=bpm, root=root)
    buf = io.BytesIO()
    score_img.save(buf, format="PNG"); buf.seek(0)

    st.success(f"¡Listo! bpm={bpm}, tonalidad={root}, duración≈{duration_sec}s")

    st.image(buf, caption="Partitura simplificada (melodía)", use_container_width=True)
    st.audio(wav_bytes, format="audio/wav")
    st.download_button("⬇️ Descargar WAV", data=wav_bytes, file_name="generado.wav", mime="audio/wav")
    st.download_button("⬇️ Descargar MIDI", data=midi_bytes, file_name="generado.mid", mime="audio/midi")

    if gen_video:
        with st.spinner("Creando vídeo MP4..."):
            # vídeo simple con el piano-roll del código original
            def make_frame_factory(notes, duration, bpm=bpm):
                q_local = 60.0 / bpm
                beats_per_bar_local = 4
                bar_time = beats_per_bar_local * q_local
                keys = list(range(DEFAULTS["key_low"], DEFAULTS["key_high"]+1))
                key_w = max(4, W // len(keys))
                roll_top = int(H * 0.1); kb_top = int(H * 0.75)
                roll_h = kb_top - roll_top
                lead_time = 3.0
                speed = roll_h / lead_time

                def make_frame(t):
                    img = Image.new("RGB", (W, H), (10, 10, 20))
                    draw = ImageDraw.Draw(img)
                    # compases
                    for bar in range(int(duration // bar_time) + 1):
                        t_bar = bar * bar_time; dt = t_bar - t
                        if -bar_time <= dt <= lead_time:
                            y_line = kb_top - int(max(0, dt) * speed)
                            draw.line([(0, y_line), (W, y_line)], fill=(200, 200, 200), width=2)
                    # teclado
                    for i, m in enumerate(keys):
                        x0, x1 = i * key_w, (i+1) * key_w
                        is_black = (m % 12) in [1,3,6,8,10]
                        col = (50,50,50) if is_black else (230,230,230)
                        draw.rectangle([x0, kb_top, x1, H], fill=col)
                    # notas
                    for role_order in ["chord","bass","mel"]:
                        for t0, d, m, role in notes:
                            if role != role_order: continue
                            m_vis = clamp_vis_to_range(m)
                            dt = t0 - t
                            if -d <= dt <= lead_time:
                                yb = kb_top - int(max(0, dt) * speed)
                                ht = max(8 if role=="chord" else 20, int(d * speed))
                                yt = yb - ht
                                idx = keys.index(m_vis)
                                if role == "chord":
                                    x0 = idx * key_w + key_w//3; x1 = (idx+1) * key_w - key_w//3; col = (120,160,200)
                                elif role == "bass":
                                    x0 = idx * key_w + key_w//4; x1 = (idx+1) * key_w - key_w//4; col = (80,80,200)
                                else:
                                    x0 = idx * key_w + 2; x1 = (idx+1) * key_w - 2; col = PC_COLORS[m % 12]
                                draw.rectangle([x0, yt, x1, yb], fill=col)
                    return np.array(img)
                return make_frame

            make_frame = make_frame_factory(notes, duration_sec)
            clip = VideoClip(make_frame, duration=duration_sec).set_fps(FPS)

            # audio temp -> AudioFileClip
            with NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                wavfile.write(tmp.name, SR, wav)
                audio_clip = AudioFileClip(tmp.name)
            clip = clip.set_audio(audio_clip)

            with NamedTemporaryFile(delete=False, suffix=".mp4") as out:
                clip.write_videofile(out.name, codec="libx264", audio_codec="aac", fps=FPS, verbose=False, logger=None)
                with open(out.name, "rb") as f:
                    video_bytes = f.read()
            st.download_button("⬇️ Descargar MP4", data=video_bytes, file_name="piano_roll.mp4", mime="video/mp4")

            # limpiar temporales
            try:
                os.remove(tmp.name)
                os.remove(out.name)
            except Exception:
                pass

# =========================
# CHAT DE MEJORA CON LLM
# =========================
st.divider()
st.subheader("💬 Pide mejoras al LLM sobre la música generada")

if "chat" not in st.session_state:
    st.session_state.chat = []

for turn in st.session_state.chat:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])

user_msg = st.chat_input("Ej.: 'hazlo más rápido y cambia a La menor'")
if user_msg:
    st.session_state.chat.append({"role":"user", "content":user_msg})
    with st.chat_message("user"):
        st.markdown(user_msg)

    if llm:
        # Pedimos SOLO JSON de nuevos parámetros; si quiere cambios en melodía,
        # bastará con regenerar con otra semilla.
        prompt = f"""
Eres un asistente musical. El usuario pide cambios. Devuelve SOLO JSON con posibles nuevos parámetros:
- bpm (60-180, int)
- root (C,C#,D,Db,Eb,F#,Gb,G#,Ab,A,Bb,B)
- reseed (bool) para forzar una nueva melodía
Si no menciona algo, conserva valor actual. Petición: "{user_msg}"
Ejemplo: {{"bpm":120,"root":"Am","reseed":true}}
"""
        try:
            out = llm.invoke(prompt)
            txt = out.content if hasattr(out, "content") else str(out)
            data = json.loads(txt)
            pretty = "```json\n" + json.dumps(data, indent=2, ensure_ascii=False) + "\n```"
            assistant_msg = "Sugerencia de parámetros:\n" + pretty + "\nPulsa **Generar música** para aplicarlos."
        except Exception as e:
            assistant_msg = f"No pude interpretar la respuesta del LLM. Error: {e}"
    else:
        assistant_msg = "No hay LLM inicializado (falta GROQ_API_KEY). Puedes añadir tu clave en `.streamlit/secrets.toml`."

    st.session_state.chat.append({"role":"assistant", "content":assistant_msg})
    with st.chat_message("assistant"):
        st.markdown(assistant_msg)
