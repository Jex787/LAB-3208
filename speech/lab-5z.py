import os
import numpy as np
import librosa
from sklearn.metrics.pairwise import cosine_similarity as cs

def extract(x):
    a, b = librosa.load(x, sr=16000)
    return np.mean(librosa.feature.mfcc(y=a, sr=b, n_mfcc=13), axis=1)

def identify(base, target):
    ref_data = {
        i.rsplit('.', 1)[0]: extract(os.path.join(base, i))
        for i in os.listdir(base) if i.endswith('.wav')
    }
    t_feat = extract(target)
    sim = {k: cs([t_feat], [v])[0][0] for k, v in ref_data.items()}
    top = max(sim, key=sim.get)
    for k, v in sim.items():
        print(f"{k}: {v:.4f}")
    print(f"Identified Speaker: {top}")

if __name__ == "__main__":
    folder = "E:/VS Code/He"
    probe = "E:/VS Code/Test/alice_real_2.wav"
    identify(folder, probe)
