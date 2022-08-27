import speech_recognition as sr
import os
import playsound
from gtts import gTTS


def Speak(txt):

    tts = gTTS(text=txt)
    file = "voice.mp3"
    tts.save(file)
    playsound.playsound(file)
    os.remove(file)


def GetAudio():
    # Instance of speech recognizer.
    r = sr.Recognizer()

    with sr.Microphone() as source:    
        # Adjust for ambient noises.
        r.adjust_for_ambient_noise(source)

        # Intake audio and store.
        print("\nSay Something: ")
            
        try:
            # Listen.
            inAudio = r.listen(source)

            # Use google API to understand speech.
            inTxt = r.recognize_google(inAudio)
            print("You said: {}".format(inTxt), "\n")

        except Exception as e:

            print("Exception: " +str(e))

    return inAudio