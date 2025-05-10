# Implementation of Real-Time Continuous Speech Recognition System Using VAD and Google API

import speech_recognition as sr

def main():
    r = sr.Recognizer()
    with sr.Microphone(sample_rate=16000) as source:
        # 1) Calibrate to ambient noise
        print("Calibrating ambient noise… please stay silent")
        r.adjust_for_ambient_noise(source, duration=1)
        print("Calibration complete. Press Ctrl+C to stop.\n")

        # 2) Continuous listen/recognize loop
        while True:
            try:
                print("Listening…")
                audio = r.listen(source, phrase_time_limit=5)
                text = r.recognize_google(audio)
                print("Recognized:", text)
            except sr.WaitTimeoutError:
                # nothing heard in this window
                continue
            except sr.UnknownValueError:
                # speech unintelligible
                print("…")
                continue
            except sr.RequestError as e:
                print(f"API error: {e}")
                break
            except KeyboardInterrupt:
                print("\nStopping.")
                break

if __name__ == "__main__":
    main()

# pip install SpeechRecognition pyaudio
# ctrl+c to stop listening