import os
import playsound
import speech_recognition as sr
from gtts import gTTS
from datetime import datetime


def speak(text, play_aloud=True):
    tts = gTTS(text=text, lang='en')
    filename = 'voice.mp3'
    os.remove(filename)  # removes file if it already exists from previous execution
    tts.save(filename)
    if play_aloud:
        playsound.playsound(filename)


def get_audio(print_words=False):
    r = sr.Recognizer()
    with sr.Microphone() as source:
        audio = r.listen(source=source)
        said = ""
        try:
            said = r.recognize_google(audio_data=audio)  # uses Google speech recognition
            if print_words:
                print(said)
        except Exception as e:
            pass
            # print("Exception: " + str(e))
    return said


class VoiceCommands:
    def __init__(self):
        self.keywords = "okay robot"
        self.passive_listen = ""
        self.active_listen = ""

    def start_passive_listen(self):
        self.passive_listen = get_audio()
        print("Listening for wake commands...")

        def heard_keyword():
            if self.passive_listen.count(self.keywords) > 0:
                speak("Yes?")
                self.passive_listen = ""
                print("Actively listening...")
                self.active_listen = get_audio(print_words=True)
                return True

        return heard_keyword()

    def stop_stream(self):
        pass

    def predict_command(self):
        # take speech that it actively hears and predicts which command is suggested
        listen = self.active_listen

        # print the command robot thinks is being suggested. Maybe add a voice command confirm clause?

        # if listen == "can you tell me the time":
        #     c_time = datetime.now().strftime("%H %M %S").split()
        #     hours = c_time[0]
        #     mins = c_time[1]
        #     seconds = c_time[2]
        #     speak(f"The current time is {hours} hours, {mins} minutes, and {seconds} seconds.", [hours, mins, seconds])
        # pass active_listen tokenizer/whatever and then into model


if __name__ == '__main__':
    vc = VoiceCommands()
    vc.start_passive_listen()
    while True:
        if vc.start_passive_listen():
            vc.predict_command()

