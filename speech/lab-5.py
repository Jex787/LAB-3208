# Text-Dependent Speaker Identification Using MFCC and Cosine Similarity with GMM

import numpy as np
import scipy.io.wavfile as wav
from python_speech_features import mfcc
import speech_recognition as sr
from sklearn.mixture import GaussianMixture
import pickle

class SpeakerVerifier:
    def __init__(self, phrase="open sesame"):
        self.phrase = phrase.lower()
        self.speaker_models = {}
        self.ubm = None
        self.threshold = -15  # Tunable

    def extract_features(self, file):
        rate, signal = wav.read(file)
        return mfcc(signal, rate, numcep=13, nfilt=26, nfft=512)

    def train_ubm(self, bg_files):
        features = np.vstack([self.extract_features(f) for f in bg_files])
        self.ubm = GaussianMixture(n_components=64, covariance_type='diag').fit(features)
        print("UBM trained.")

    def enroll(self, speaker_id, files):
        features = np.vstack([self.extract_features(f) for f in files])
        model = GaussianMixture(n_components=16, covariance_type='diag').fit(features)
        self.speaker_models[speaker_id] = model
        print(f"{speaker_id} enrolled.")

    def verify(self, file, speaker_id):
        # Phrase verification
        rec = sr.Recognizer()
        with sr.AudioFile(file) as src:
            audio = rec.record(src)
        try:
            if rec.recognize_google(audio).lower() != self.phrase:
                print("Wrong phrase.")
                return False
        except:
            print("Speech recognition failed.")
            return False

        # Speaker verification
        feat = self.extract_features(file)
        spk_score = self.speaker_models[speaker_id].score(feat)
        ubm_score = self.ubm.score(feat)
        print(f"Speaker score: {spk_score:.2f}, UBM score: {ubm_score:.2f}")
        return (spk_score - ubm_score) > self.threshold

    def save_model(self, path="verifier.pkl"):
        with open(path, "wb") as f:
            pickle.dump({
                "phrase": self.phrase,
                "ubm": self.ubm,
                "models": self.speaker_models,
                "threshold": self.threshold
            }, f)
        print("Model saved.")

    def load_model(self, path="verifier.pkl"):
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.phrase = data["phrase"]
            self.ubm = data["ubm"]
            self.speaker_models = data["models"]
            self.threshold = data["threshold"]
        print("Model loaded.")

if __name__ == "__main__":
    verifier = SpeakerVerifier("open sesame")

    # Train and enroll
    verifier.train_ubm(["bg1.wav", "bg2.wav"])
    verifier.enroll("alice", ["alice1.wav", "alice2.wav"])

    # Save model
    verifier.save_model()

    # Load model later (optional)
    # verifier.load_model()

    # Verify
    result = verifier.verify("test.wav", "alice")
    print("Verification:", "Accepted" if result else "Rejected")
