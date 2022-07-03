from collections import deque
import numpy as np
import pyaudio
import soundfile as sf


CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
DURATION = 1.1
FRAMES = DURATION * RATE
VAD_FRAMES = 480


def record():
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    buffer = b''

    for i in range(0, int(FRAMES / CHUNK)):
        buffer += stream.read(CHUNK)

    stream.stop_stream()
    stream.close()
    p.terminate()

    int16_data = np.frombuffer(buffer, dtype=np.int16)
    float32_data = int16_data.astype(dtype=np.float32) / 32768

    return float32_data


def save(frames):
    sf.write('last.wav', frames, RATE)


def monitor(vad):
    recent = deque(maxlen=int(FRAMES / VAD_FRAMES))
    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    for i in range(0, int(FRAMES / VAD_FRAMES)):
        frames = stream.read(VAD_FRAMES)
        recent.append((frames, vad.is_speech(frames, RATE)))

    triggered = False

    while not triggered:
        num_voiced = len([f for f, speech in recent if speech])

        if num_voiced >= 0.8 * recent.maxlen:
            triggered = True
        else:
            frames = stream.read(VAD_FRAMES)
            recent.append((frames, vad.is_speech(frames, RATE)))

    stream.stop_stream()
    stream.close()
    p.terminate()

    buffer = b''.join([f for f, speech in recent])

    int16_data = np.frombuffer(buffer, dtype=np.int16)
    float32_data = int16_data.astype(dtype=np.float32) / 32768

    return float32_data
