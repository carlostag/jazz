import os, math, random
import numpy as np
import streamlit as st
from tempfile import NamedTemporaryFile
from PIL import Image, ImageDraw
from scipy.io import wavfile
from moviepy.editor import VideoClip, AudioFileClip

# =========================
# CONFIG
# =========================
DURATION_SEC = 60
SR = 44100
W, H = 1080, 1920
FPS = 30
KEY_LOW, KEY_HIGH = 48, 76

NOTE_TO_SEMITONE = {"C":0,"C#":1,"Db":1,"D":2,"D#":3,"Eb":3,"E":4,"F":5,
    "F#":6,"Gb":6,"G":7,"G#":8,"Ab":8,"A":9,"A#":10,"Bb":10,"B":11}

PC_COLORS = {
    0:(255,140,140), 1:(255,170,120), 2:(255,200,120), 3:(240,220,120),
    4:(200,240,130), 5:(150,250,160), 6:(140,230,220), 7:(140,190,255),
    8:(170,160,255), 9:(210,150,255),10:(240,140,220),11:(255,140,180)
}

# =========================
# UTILIDADES
# =========================
def midi_from_note(note_name, octave=4):
    return 12*(octave+1)+NOTE_TO_SEMITONE[note_name]

def freq_from_midi(m):
    return 440.0*(2.0**((m-69)/12.0))

def clamp_vis_to_range(m, low=KEY_LOW, high=KEY_HIGH):
    while m < low:  m += 12
    while m > high: m -= 12
    return m

def adsr_envelope(n_samples, sr, attack=0.01, decay=0.2, sustain=0.5, release=0.2):
    a = int(attack*sr); d = int(decay*sr); r = int(release*sr)
    total = a+d+r
    if total > n_samples:
        factor = n_samples/total
        a = max(1,int(a*factor)); d = max(1,int(d*factor)); r = max(1,int(r*factor))
    s = max(n_samples-(a+d+r),0)
    env = np.zeros(n_samples,dtype=np.float32)
    if a>0: env[:a]=np.linspace(0,1,a,endpoint=False)
    if d>0: env[a:a+d]=np.linspace(1,sustain,d,endpoint=False)
    if s>0: env[a+d:a+d+s]=sustain
    if r>0: env[a+d+s:a+d+s+r]=np.linspace(sustain,0,r,endpoint=True)
    return env

def apply_fades(y,sr,ms=3):
    n=len(y); f=max(1,int((ms/1000.0)*sr))
    if f*2>=n: f=n//4 if n>=4 else 1
    if f>1:
        win=0.5-0.5*np.cos(np.linspace(0,math.pi,f))
        y[:f]*=win; y[-f:]*=win[::-1]
    return y

def mallet_note(freq,dur,sr):
    n=int(sr*dur); 
    if n<=0: return np.zeros(0,dtype=np.float32)
    t=np.linspace(0,dur,n,endpoint=False)
    nyq=sr/2.0
    mults=[1,2,3,4]; amps=[1.0,0.35,0.2,0.12]
    x=np.zeros(n,dtype=np.float32)
    for m,a in zip(mults,amps):
        f=freq*m
        if f<nyq: x+=a*np.sin(2*np.pi*f*t)
    env=adsr_envelope(n,sr,0.008,0.18,0.45,0.18)
    y=apply_fades(x*env,sr,3)
    return y

def bass_sine(freq,dur,sr):
    n=int(sr*dur)
    if n<=0: return np.zeros(0,dtype=np.float32)
    t=np.linspace(0,dur,n,endpoint=False)
    x=np.sin(2*np.pi*freq*t).astype(np.float32)
    env=adsr_envelope(n,sr,0.005,0.12,0.65,0.12)
    return apply_fades(x*env,sr,3)

# =========================
# PROGRESIONES ALEATORIAS
# =========================
def random_progression(root="C"):
    progs = [
        [("I","maj"),("vi","min"),("ii","min"),("V","maj")],
        [("I","maj"),("IV","maj"),("I","maj"),("V","maj")],
        [("ii","min"),("V","maj"),("I","maj"),("I","maj")],
    ]
    return random.choice(progs)

def chord_root_from_degree(root_midi, degree):
    steps={"I":0,"ii":2,"iii":4,"IV":5,"V":7,"vi":9}
    return root_midi+steps[degree]

def chord_voicing_simple(midi_root, quality):
    return [midi_root, midi_root+(4 if quality=="maj" else 3), midi_root+7]

# =========================
# GENERADOR MUSICAL
# =========================
def generate_track(duration_s, sr, bpm, root="C"):
    q=60.0/bpm; beats_per_bar=4
    total_bars=int(duration_s/(beats_per_bar*q))
    root_midi=midi_from_note(root,4)
    prog=random_progression(root)
    audio=np.zeros(int(sr*duration_s),dtype=np.float32)
    notes=[]
    last_mel=None

    for bar in range(total_bars):
        deg,quality=prog[bar%len(prog)]
        chord_root=chord_root_from_degree(root_midi,deg)
        t_bar=bar*beats_per_bar*q
        last_bar=(bar==total_bars-1)

        # acordes
        dur_ch=beats_per_bar*q
        for m in chord_voicing_simple(chord_root,quality):
            y=0.2*mallet_note(freq_from_midi(m),dur_ch,sr)
            s=int(t_bar*sr); e=s+len(y)
            audio[s:e]+=y; notes.append((t_bar,dur_ch,m,"chord"))

        # bajo
        if not last_bar:
            f=freq_from_midi(chord_root-24)
            y=0.3*bass_sine(f,dur_ch,sr)
            s=int(t_bar*sr); e=s+len(y)
            audio[s:e]+=y; notes.append((t_bar,dur_ch,chord_root-24,"bass"))

        # melodía
        if not last_bar:
            scale=[chord_root,chord_root+2,chord_root+4,chord_root+7]
            sub=0.0
            while sub<beats_per_bar:
                if random.random()<0.8:
                    m=random.choice(scale)
                    dur_b=random.choice([0.5,1.0])
                    if sub+dur_b>beats_per_bar: dur_b=beats_per_bar-sub
                    t_mel=t_bar+sub*q
                    y=0.4*mallet_note(freq_from_midi(m),dur_b*q,sr)
                    s=int(t_mel*sr); e=s+len(y)
                    audio[s:e]+=y; notes.append((t_mel,dur_b*q,m,"mel"))
                    sub+=dur_b
                else:
                    sub+=0.5
    audio/=np.max(np.abs(audio)+1e-12)
    return audio,notes

# =========================
# VISUAL
# =========================
def make_frame_factory(notes,duration,bpm):
    q=60.0/bpm; beats_per_bar=4; bar_time=beats_per_bar*q
    keys=list(range(KEY_LOW,KEY_HIGH+1))
    key_w=max(4,W//len(keys))
    roll_top=int(H*0.1); kb_top=int(H*0.75)
    roll_h=kb_top-roll_top; lead_time=3.0
    speed=roll_h/lead_time
    def make_frame(t):
        img=Image.new("RGB",(W,H),(10,10,20))
        draw=ImageDraw.Draw(img)
        # compases
        for bar in range(int(duration//bar_time)+1):
            t_bar=bar*bar_time; dt=t_bar-t
            if -bar_time<=dt<=lead_time:
                y_line=kb_top-int(max(0,dt)*speed)
                draw.line([(0,y_line),(W,y_line)],fill=(200,200,200),width=2)
        # teclado
        for i,m in enumerate(keys):
            x0,x1=i*key_w,(i+1)*key_w
            col=(50,50,50) if m%12 in [1,3,6,8,10] else (230,230,230)
            draw.rectangle([x0,kb_top,x1,H],fill=col)
        # notas
        for role_order in ["chord","bass","mel"]:
            for t0,d,m,role in notes:
                if role!=role_order: continue
                m_vis=clamp_vis_to_range(m)
                if m_vis not in keys: continue
                dt=t0-t
                if -d<=dt<=lead_time:
                    yb=kb_top-int(max(0,dt)*speed)
                    ht=max(8 if role=="chord" else 20,int(d*speed))
                    yt=yb-ht
                    idx=keys.index(m_vis)
                    if role=="chord":
                        x0=idx*key_w+key_w//3; x1=(idx+1)*key_w-key_w//3; col=(120,160,200)
                    elif role=="bass":
                        x0=idx*key_w+key_w//4; x1=(idx+1)*key_w-key_w//4; col=(80,80,200)
                    else:
                        x0=idx*key_w+2; x1=(idx+1)*key_w-2; col=PC_COLORS[m%12]
                    draw.rectangle([x0,yt,x1,yb],fill=col)
        return np.array(img)
    return make_frame

# =========================
# STREAMLIT UI
# =========================
st.title("🎷 Generador de Jazz Shorts (60s)")
bpm=st.slider("BPM",70,130,100)
root=st.selectbox("Tonalidad",["C","D","E","F","G","A","Bb"])
if st.button("🎬 Generar vídeo aleatorio"):
    y,notes=generate_track(DURATION_SEC,SR,bpm,root)
    with NamedTemporaryFile(delete=False,suffix=".wav") as tmp:
        wav= np.int16(np.clip(y,-1.0,1.0)*32767)
        wavfile.write(tmp.name,SR,wav)
        audio_path=tmp.name
    make_frame=make_frame_factory(notes,DURATION_SEC,bpm)
    clip=VideoClip(make_frame,duration=DURATION_SEC).set_fps(FPS)
    audio=AudioFileClip(audio_path)
    clip=clip.set_audio(audio)
    out="jazz_short.mp4"
    clip.write_videofile(out,codec="libx264",audio_codec="aac",fps=FPS)
    st.video(out)
    os.remove(audio_path)
import os, math, random
import numpy as np
import streamlit as st
from tempfile import NamedTemporaryFile
from PIL import Image, ImageDraw
from scipy.io import wavfile
from moviepy.editor import VideoClip, AudioFileClip

# =========================
# CONFIG
# =========================
DURATION_SEC = 60
SR = 44100
W, H = 1080, 1920
FPS = 30
KEY_LOW, KEY_HIGH = 48, 76

NOTE_TO_SEMITONE = {"C":0,"C#":1,"Db":1,"D":2,"D#":3,"Eb":3,"E":4,"F":5,
    "F#":6,"Gb":6,"G":7,"G#":8,"Ab":8,"A":9,"A#":10,"Bb":10,"B":11}

PC_COLORS = {
    0:(255,140,140), 1:(255,170,120), 2:(255,200,120), 3:(240,220,120),
    4:(200,240,130), 5:(150,250,160), 6:(140,230,220), 7:(140,190,255),
    8:(170,160,255), 9:(210,150,255),10:(240,140,220),11:(255,140,180)
}

# =========================
# UTILIDADES
# =========================
def midi_from_note(note_name, octave=4):
    return 12*(octave+1)+NOTE_TO_SEMITONE[note_name]

def freq_from_midi(m):
    return 440.0*(2.0**((m-69)/12.0))

def clamp_vis_to_range(m, low=KEY_LOW, high=KEY_HIGH):
    while m < low:  m += 12
    while m > high: m -= 12
    return m

def adsr_envelope(n_samples, sr, attack=0.01, decay=0.2, sustain=0.5, release=0.2):
    a = int(attack*sr); d = int(decay*sr); r = int(release*sr)
    total = a+d+r
    if total > n_samples:
        factor = n_samples/total
        a = max(1,int(a*factor)); d = max(1,int(d*factor)); r = max(1,int(r*factor))
    s = max(n_samples-(a+d+r),0)
    env = np.zeros(n_samples,dtype=np.float32)
    if a>0: env[:a]=np.linspace(0,1,a,endpoint=False)
    if d>0: env[a:a+d]=np.linspace(1,sustain,d,endpoint=False)
    if s>0: env[a+d:a+d+s]=sustain
    if r>0: env[a+d+s:a+d+s+r]=np.linspace(sustain,0,r,endpoint=True)
    return env

def apply_fades(y,sr,ms=3):
    n=len(y); f=max(1,int((ms/1000.0)*sr))
    if f*2>=n: f=n//4 if n>=4 else 1
    if f>1:
        win=0.5-0.5*np.cos(np.linspace(0,math.pi,f))
        y[:f]*=win; y[-f:]*=win[::-1]
    return y

def mallet_note(freq,dur,sr):
    n=int(sr*dur); 
    if n<=0: return np.zeros(0,dtype=np.float32)
    t=np.linspace(0,dur,n,endpoint=False)
    nyq=sr/2.0
    mults=[1,2,3,4]; amps=[1.0,0.35,0.2,0.12]
    x=np.zeros(n,dtype=np.float32)
    for m,a in zip(mults,amps):
        f=freq*m
        if f<nyq: x+=a*np.sin(2*np.pi*f*t)
    env=adsr_envelope(n,sr,0.008,0.18,0.45,0.18)
    y=apply_fades(x*env,sr,3)
    return y

def bass_sine(freq,dur,sr):
    n=int(sr*dur)
    if n<=0: return np.zeros(0,dtype=np.float32)
    t=np.linspace(0,dur,n,endpoint=False)
    x=np.sin(2*np.pi*freq*t).astype(np.float32)
    env=adsr_envelope(n,sr,0.005,0.12,0.65,0.12)
    return apply_fades(x*env,sr,3)

# =========================
# PROGRESIONES ALEATORIAS
# =========================
def random_progression(root="C"):
    progs = [
        [("I","maj"),("vi","min"),("ii","min"),("V","maj")],
        [("I","maj"),("IV","maj"),("I","maj"),("V","maj")],
        [("ii","min"),("V","maj"),("I","maj"),("I","maj")],
    ]
    return random.choice(progs)

def chord_root_from_degree(root_midi, degree):
    steps={"I":0,"ii":2,"iii":4,"IV":5,"V":7,"vi":9}
    return root_midi+steps[degree]

def chord_voicing_simple(midi_root, quality):
    return [midi_root, midi_root+(4 if quality=="maj" else 3), midi_root+7]

# =========================
# GENERADOR MUSICAL
# =========================
def generate_track(duration_s, sr, bpm, root="C"):
    q=60.0/bpm; beats_per_bar=4
    total_bars=int(duration_s/(beats_per_bar*q))
    root_midi=midi_from_note(root,4)
    prog=random_progression(root)
    audio=np.zeros(int(sr*duration_s),dtype=np.float32)
    notes=[]
    last_mel=None

    for bar in range(total_bars):
        deg,quality=prog[bar%len(prog)]
        chord_root=chord_root_from_degree(root_midi,deg)
        t_bar=bar*beats_per_bar*q
        last_bar=(bar==total_bars-1)

        # acordes
        dur_ch=beats_per_bar*q
        for m in chord_voicing_simple(chord_root,quality):
            y=0.2*mallet_note(freq_from_midi(m),dur_ch,sr)
            s=int(t_bar*sr); e=s+len(y)
            audio[s:e]+=y; notes.append((t_bar,dur_ch,m,"chord"))

        # bajo
        if not last_bar:
            f=freq_from_midi(chord_root-24)
            y=0.3*bass_sine(f,dur_ch,sr)
            s=int(t_bar*sr); e=s+len(y)
            audio[s:e]+=y; notes.append((t_bar,dur_ch,chord_root-24,"bass"))

        # melodía
        if not last_bar:
            scale=[chord_root,chord_root+2,chord_root+4,chord_root+7]
            sub=0.0
            while sub<beats_per_bar:
                if random.random()<0.8:
                    m=random.choice(scale)
                    dur_b=random.choice([0.5,1.0])
                    if sub+dur_b>beats_per_bar: dur_b=beats_per_bar-sub
                    t_mel=t_bar+sub*q
                    y=0.4*mallet_note(freq_from_midi(m),dur_b*q,sr)
                    s=int(t_mel*sr); e=s+len(y)
                    audio[s:e]+=y; notes.append((t_mel,dur_b*q,m,"mel"))
                    sub+=dur_b
                else:
                    sub+=0.5
    audio/=np.max(np.abs(audio)+1e-12)
    return audio,notes

# =========================
# VISUAL
# =========================
def make_frame_factory(notes,duration,bpm):
    q=60.0/bpm; beats_per_bar=4; bar_time=beats_per_bar*q
    keys=list(range(KEY_LOW,KEY_HIGH+1))
    key_w=max(4,W//len(keys))
    roll_top=int(H*0.1); kb_top=int(H*0.75)
    roll_h=kb_top-roll_top; lead_time=3.0
    speed=roll_h/lead_time
    def make_frame(t):
        img=Image.new("RGB",(W,H),(10,10,20))
        draw=ImageDraw.Draw(img)
        # compases
        for bar in range(int(duration//bar_time)+1):
            t_bar=bar*bar_time; dt=t_bar-t
            if -bar_time<=dt<=lead_time:
                y_line=kb_top-int(max(0,dt)*speed)
                draw.line([(0,y_line),(W,y_line)],fill=(200,200,200),width=2)
        # teclado
        for i,m in enumerate(keys):
            x0,x1=i*key_w,(i+1)*key_w
            col=(50,50,50) if m%12 in [1,3,6,8,10] else (230,230,230)
            draw.rectangle([x0,kb_top,x1,H],fill=col)
        # notas
        for role_order in ["chord","bass","mel"]:
            for t0,d,m,role in notes:
                if role!=role_order: continue
                m_vis=clamp_vis_to_range(m)
                if m_vis not in keys: continue
                dt=t0-t
                if -d<=dt<=lead_time:
                    yb=kb_top-int(max(0,dt)*speed)
                    ht=max(8 if role=="chord" else 20,int(d*speed))
                    yt=yb-ht
                    idx=keys.index(m_vis)
                    if role=="chord":
                        x0=idx*key_w+key_w//3; x1=(idx+1)*key_w-key_w//3; col=(120,160,200)
                    elif role=="bass":
                        x0=idx*key_w+key_w//4; x1=(idx+1)*key_w-key_w//4; col=(80,80,200)
                    else:
                        x0=idx*key_w+2; x1=(idx+1)*key_w-2; col=PC_COLORS[m%12]
                    draw.rectangle([x0,yt,x1,yb],fill=col)
        return np.array(img)
    return make_frame

# =========================
# STREAMLIT UI
# =========================
st.title("🎷 Generador de Jazz Shorts (60s)")

bpm = st.slider("BPM", 70, 130, 100, key="slider_bpm")
root = st.selectbox("Tonalidad", ["C","D","E","F","G","A","Bb"], key="select_root")

if st.button("🎬 Generar vídeo aleatorio", key="btn_generate"):
    y, notes = generate_track(DURATION_SEC, SR, bpm, root)
    with NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        wav = np.int16(np.clip(y, -1.0, 1.0) * 32767)
        wavfile.write(tmp.name, SR, wav)
        audio_path = tmp.name

    make_frame = make_frame_factory(notes, DURATION_SEC, bpm)
    clip = VideoClip(make_frame, duration=DURATION_SEC).set_fps(FPS)
    audio = AudioFileClip(audio_path)
    clip = clip.set_audio(audio)

    out = "jazz_short.mp4"
    clip.write_videofile(out, codec="libx264", audio_codec="aac", fps=FPS)

    st.video(out)
    os.remove(audio_path)

