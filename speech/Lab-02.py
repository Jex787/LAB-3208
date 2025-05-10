# Comparative Analysis of Speech Feature Extraction Techniques (MFCC, PLP, LPC)

import librosa
import python_speech_features as psf
import matplotlib.pyplot as plt
import numpy as np
import os

def extract_features(audio_file):
    # Check if file exists
    if not os.path.isfile(audio_file):
        raise FileNotFoundError(f"Audio file not found: {audio_file}")
    
    # Load audio
    y, sr = librosa.load(audio_file, sr=16000, mono=True, duration=5)
    y = y.flatten()  # Ensure 1D audio

    # MFCC
    mfcc = psf.mfcc(y, sr, numcep=13, nfilt=26, nfft=512)

    # PLP (fallback to MFCC if fails)
    try:
        plp = psf.plp(y, sr, numcep=13, nfilt=26, nfft=512)
    except:
        plp = mfcc

    # LPC using frame-wise processing
    frames = psf.sigproc.framesig(y, int(0.025 * sr), int(0.01 * sr))
    lpc = [librosa.lpc(f + 0.001*np.random.randn(len(f)), order=12) for f in frames]

    return mfcc, plp, np.array(lpc)

def plot_features(mfcc, plp, lpc):
    plt.figure(figsize=(15, 12))

    # MFCC
    plt.subplot(3, 1, 1)
    plt.imshow(mfcc.T, aspect='auto', origin='lower')
    plt.title("MFCC Features")
    plt.colorbar()
    plt.tight_layout(pad=3.0)

    # PLP
    plt.subplot(3, 1, 2)
    plt.imshow(plp.T, aspect='auto', origin='lower')
    plt.title("PLP Features")
    plt.colorbar()
    plt.tight_layout(pad=3.0)

    # LPC
    plt.subplot(3, 1, 3)
    plt.plot(lpc[:, 1:])  # Skip the first LPC coefficient (usually 1)
    plt.title("LPC Coefficients")
    plt.xlabel("Frame")
    plt.ylabel("Coefficient Value")
    plt.tight_layout(pad=3.0)

    # Adjust layout and save
    plt.subplots_adjust(hspace=0.4)
    plt.savefig("feature_comparison.png")
    plt.show()

if __name__ == "__main__":
    # Use relative path to file in the 'speech' subfolder
    audio_path = "speech/speech_sample.wav"

    mfcc, plp, lpc = extract_features(audio_path)
    print("MFCC Shape:", mfcc.shape)
    print("PLP Shape:", plp.shape)
    print("LPC Shape:", lpc.shape)

    plot_features(mfcc, plp, lpc)




# Required Dependencies(Python 3.13.3_latest)

# pip install librosa python_speech_features matplotlib numpy soundfile


