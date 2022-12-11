import tkinter
from tkinter import filedialog

import customtkinter
import sounddevice
import numpy as np
import librosa
import librosa.display
from tensorflow import keras
from threading import Thread
from time import sleep
import PIL.Image
from sklearn.preprocessing import LabelEncoder


class App(customtkinter.CTk):
    AUDIO_FILE_EXTENSIONS = '.m4a .flac .mp3 .mp4 .wav .wma .aac'
    AUDIO_TYPES = [('Wszystkie pliki audio', AUDIO_FILE_EXTENSIONS)]

    FONT_TITLE_SIZE = 35
    FONT_INFO_SIZE = 30

    def __init__(self):
        super().__init__()
        customtkinter.set_appearance_mode("Dark")
        customtkinter.set_default_color_theme("green")

        default_font = tkinter.font.nametofont("TkDefaultFont")
        self.TITLE_FONT = (default_font, App.FONT_TITLE_SIZE)
        self.INFO_FONT = (default_font, App.FONT_INFO_SIZE)

        self.fs = 44100
        self.seconds = 5
        self.n_mfcc = 40
        self.top_db = 20

        self.loaded_model = keras.models.load_model('neural_network')
        self.label_encoder = LabelEncoder()
        self.label_encoder.classes_ = np.load(r'neural_network/labels.npy')
        image_record = customtkinter.CTkImage(light_image=PIL.Image.open(r'images/transparent_microphone.png'),
                                              size=(100, 100))

        self.title("System identyfikacji osoby za pomocą głosu")
        self.geometry("800x600")

        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        self.panel_import = customtkinter.CTkFrame(self)
        self.panel_import.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)

        self.panel_record = customtkinter.CTkFrame(self)
        self.panel_record.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)

        self.panel_result = customtkinter.CTkFrame(self)
        self.panel_result.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)

        self.panel_import.tkraise()

        # panel główny
        self.panel_import.rowconfigure((0, 1, 2), weight=1)
        self.panel_import.columnconfigure(0, weight=1)

        self.label_import_title = customtkinter.CTkLabel(master=self.panel_import,
                                                         text="System identyfikacji osoby na podstawie głosu",
                                                         text_color="#2CC985",
                                                         font=self.TITLE_FONT)
        self.label_import_title.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)

        self.button_record = customtkinter.CTkButton(master=self.panel_import,
                                                     text="Nagraj głos",
                                                     text_color="gray14",
                                                     font=self.INFO_FONT,
                                                     command=self.go_to_record_panel)
        self.button_record.grid(row=1, column=0, sticky="nsew", padx=20, pady=20)

        self.button_import = customtkinter.CTkButton(master=self.panel_import,
                                                     text="Importuj nagranie",
                                                     text_color="gray14",
                                                     font=self.INFO_FONT,
                                                     command=self.import_audio)
        self.button_import.grid(row=2, column=0, sticky="nsew", padx=20, pady=20)

        # panel nagrywania
        self.panel_record.rowconfigure((0, 1, 2, 3), weight=1)
        self.panel_record.columnconfigure(0, weight=1)

        self.label_record_title = customtkinter.CTkLabel(master=self.panel_record,
                                                         text="Naciśnij przycisk mikrofonu, aby nagrać głos.",
                                                         text_color="#2CC985",
                                                         font=self.TITLE_FONT)
        self.label_record_title.grid(row=0, column=0, padx=20, pady=20)

        self.button_record = customtkinter.CTkButton(master=self.panel_record,
                                                     text="",
                                                     width=120,
                                                     height=120,
                                                     text_color="black",
                                                     # corner_radius=20,
                                                     image=image_record,
                                                     command=self.record_voice)
        self.button_record.grid(row=1, column=0, padx=20, pady=20)

        self.label_record_info = customtkinter.CTkLabel(master=self.panel_record,
                                                        text="",
                                                        text_color="#2CC985",
                                                        font=self.INFO_FONT)
        self.label_record_info.grid(row=2, column=0, padx=20, pady=20)

        self.button_return_1 = customtkinter.CTkButton(master=self.panel_record,
                                                       text="Powrót",
                                                       text_color="gray14",
                                                       text_color_disabled="gray14",
                                                       font=self.INFO_FONT,
                                                       command=self.return_to_main)
        self.button_return_1.grid(row=3, column=0, columnspan=2, sticky="nsew", padx=20, pady=20)

        # panel z wynikami
        self.panel_result.rowconfigure((0, 1, 2, 3), weight=1)
        self.panel_result.columnconfigure(0, weight=1)

        self.empty_label_1 = customtkinter.CTkLabel(master=self.panel_result,
                                                    text="")
        self.empty_label_1.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)

        self.label_result = customtkinter.CTkLabel(master=self.panel_result,
                                                   text="Wykryto osobę: ",
                                                   text_color="#2CC985",
                                                   font=self.TITLE_FONT)
        self.label_result.grid(row=1, column=0, sticky="nsew", padx=20, pady=20)

        self.empty_label_2 = customtkinter.CTkLabel(master=self.panel_result,
                                                    text="")
        self.empty_label_2.grid(row=2, column=0, sticky="nsew", padx=20, pady=20)

        self.button_return_2 = customtkinter.CTkButton(master=self.panel_result,
                                                       text="Spróbuj jeszcze raz",
                                                       text_color="gray14",
                                                       font=self.INFO_FONT,
                                                       command=self.return_to_main)
        self.button_return_2.grid(row=3, column=0, sticky="nsew", padx=20, pady=20)

    def go_to_record_panel(self):
        self.button_record.configure(state=customtkinter.NORMAL)
        self.button_return_1.configure(state=customtkinter.NORMAL)
        self.panel_record.tkraise()

    def return_to_main(self):
        self.panel_import.tkraise()

    def countdown(self):
        seconds = 5
        while seconds >= 0:
            self.label_record_info.configure(text="Pozostało sekund: "+str(seconds))
            sleep(1)
            seconds -= 1
        self.label_record_info.configure(text="")

    def record(self):
        recorded_voice = sounddevice.rec(int(self.seconds * self.fs), samplerate=self.fs, channels=1)
        sounddevice.wait()
        data = np.squeeze(recorded_voice)
        data = librosa.util.normalize(data)
        data = librosa.effects.trim(data, top_db=self.top_db)[0]
        self.predict(data)

    def record_voice(self):
        thread_countdown = Thread(target=self.countdown)
        thread_record = Thread(target=self.record)
        thread_countdown.start()
        thread_record.start()
        self.button_return_1.configure(state=customtkinter.DISABLED)
        self.button_record.configure(state=customtkinter.DISABLED)

    def import_audio(self):
        path = filedialog.askopenfilename(title="Wybierz obraz", filetypes=App.AUDIO_TYPES)
        if path:
            data, sample_rate = librosa.load(path, sr=self.fs)
            data = librosa.util.normalize(data)
            data = librosa.effects.trim(data, top_db=self.top_db)[0]
            self.predict(data)

    def predict(self, data):
        mfcc = np.mean(librosa.feature.mfcc(y=data, sr=self.fs, n_mfcc=self.n_mfcc).T, axis=0)
        mfcc_scaled_features = mfcc.reshape(1, -1)
        predicted_label = self.loaded_model.predict(mfcc_scaled_features)
        prediction_index = np.argmax(predicted_label, axis=1)
        result = self.label_encoder.inverse_transform(prediction_index)[0]
        result_string = "Witaj, "+result
        self.label_result.configure(text=result_string)
        self.panel_result.tkraise()


if __name__ == '__main__':
    app = App()
    app.mainloop()
