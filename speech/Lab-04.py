import os
# Suppress TensorFlow info/warning logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import librosa
import crepe
import matplotlib.pyplot as plt
import numpy as np

def compute_all_pitches(y, sr):
    # YIN Algorithm
    f0_yin = librosa.yin(y, fmin=80, fmax=400, sr=sr)
    # PYIN Algorithm
    f0_pyin, _, _ = librosa.pyin(y, fmin=80, fmax=400, sr=sr)
    # CREPE Algorithm
    time, f0_crepe, confidence, _ = crepe.predict(y, sr, viterbi=True)

    return {
        'yin': f0_yin,
        'pyin': f0_pyin,
        'crepe': (time, f0_crepe)
    }

def plot_pitch_comparison(pitches, sr):
    plt.figure(figsize=(15, 8))
    times = librosa.times_like(pitches['yin'], sr=sr)
    plt.plot(times, pitches['yin'], label='YIN', alpha=0.7)
    plt.plot(times, pitches['pyin'], label='PYIN', alpha=0.7)
    plt.plot(pitches['crepe'][0], pitches['crepe'][1], label='CREPE', alpha=0.7)

    plt.title("Pitch Estimation Comparison")
    plt.ylabel("Frequency (Hz)")
    plt.xlabel("Time (s)")
    plt.legend()
    plt.grid()
    plt.savefig('pitch_comparison.png')
    plt.show()

if __name__ == "__main__":
    audio_file = "speech/speech_sample.wav"
    if os.path.exists(audio_file):
        y, sr = librosa.load(audio_file, sr=16000, duration=5)
        pitches = compute_all_pitches(y, sr)
        plot_pitch_comparison(pitches, sr)
    else:
        print(f"Audio file '{audio_file}' not found. Please provide a valid file path.")



#Required Command(Python 3.10)
# python -m pip install --upgrade pip
#pip install librosa matplotlib numpy tensorflow crepe
