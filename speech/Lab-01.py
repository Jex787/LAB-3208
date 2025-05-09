import sounddevice as sd
from scipy.io.wavfile import write
import speech_recognition as sr

# Record audio from microphone
def record_audio(filename="my_recording.wav", duration=5, fs=16000):
    print("Recording... Please speak.")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    write(filename, fs, audio)
    print("Recording saved.")

# Convert recorded audio to text using Google API
def speech_to_text(filename="my_recording.wav"):
    recognizer = sr.Recognizer()
    with sr.AudioFile(filename) as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data)
            print("Recognized Text:", text)
        except sr.UnknownValueError:
            print("Sorry, could not understand the audio.")
        except sr.RequestError as e:
            print(f"API Error: {e}")

# Main process
if __name__ == "__main__":
    record_audio()
    speech_to_text()



# Required Dependencies(Python 3.13.3_latest)

# pip install sounddevice scipy SpeechRecognition
                # OR(IF NOT THEN)
#pip install pipwin
#pip install pyaudio
