import tkinter as tk
import customtkinter as ctk
import cv2
import numpy as np
import tensorflow as tf
import joblib
import mediapipe as mp
import time
from googletrans import Translator, LANGUAGES
from PIL import Image, ImageTk
import threading
import asyncio

single_hand_model = tf.keras.models.load_model("models/asl_single_hand_model.keras")
single_hand_label_encoder = joblib.load("models/asl_single_hand_label_encoder.pkl")
double_hand_model = tf.keras.models.load_model("models/asl_double_hand_model.keras")
double_hand_label_encoder = joblib.load("models/asl_double_hand_label_encoder.pkl")

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2)

sign_map = {
    'how are you?': ['how', 'you'],
    'what is your name?': ['you', 'name', "what"],
    "my name is": ['me', 'name'],
    "i'm good": ['me', 'good'],
    "i'm fine": ['me', 'fine'],
}

sign_language = {
    "American Sign Language (ASL)": "en",
    "Arabic Sign Language (ArSL)": "ar",
}

models = {
            "American Sign Language (ASL)": {
                "single_hand_model": "models/asl_single_hand_model.keras",
                "single_hand_label_encoder": "models/asl_single_hand_label_encoder.pkl",
                "double_hand_model": "models/asl_double_hand_model.keras",
                "double_hand_label_encoder": "models/asl_double_hand_label_encoder.pkl"
            },
            "Arabic Sign Language (ArSL)": {
                "single_hand_model": "models/arsl_single_hand_model.keras",
                "single_hand_label_encoder": "models/arsl_single_hand_label_encoder.pkl",
                "double_hand_model": None,
                "double_hand_label_encoder": None
            }
        }

translator = Translator()

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.geometry("1020x560")
        self.title("Sign Language Translator")
        self.running = False
        self.recognized_text = ""
        self.frame_buffer = []
        self.rh_frames = []
        self.lh_frames = []
        self.seq = []
        self.last_write_time = 0
        self.write_interval = 0
        self.current_character = None
        self.last_character = None
        self.model_type_single = True
        self.translation_text = ""
        self.models = models
        self.single_hand_model = None
        self.single_hand_label_encoder = None
        self.double_hand_model = None
        self.double_hand_label_encoder = None
        self.sign_to_text = True
        self.protocol("WM_DELETE_WINDOW", self.close)
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")


        self.load_models("American Sign Language (ASL)")

        
        main_frame = ctk.CTkFrame(self)
        main_frame.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="nsew")

        # Sign Language Frame
        self.sign_language_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        self.sign_language_frame.grid(row=0, column=0, padx=(50, 0), pady=10, sticky="ew")

        from_label = ctk.CTkLabel(self.sign_language_frame, text="From:", font=("Arial", 13))
        from_label.grid(row=0, column=0, padx=(0, 5), pady=0)

        self.sign_language_var = ctk.StringVar(value="American Sign Language (ASL)")
        self.sign_language_options = [lang for lang in models.keys()]
        self.sign_language_menu = ctk.CTkOptionMenu(self.sign_language_frame, values=self.sign_language_options, variable=self.sign_language_var, font=("Arial", 13))
        self.sign_language_menu.grid(row=0, column=1, padx=0, pady=0)
        self.sign_language_menu.bind("<Configure>", lambda e: self.on_language_change(self.sign_language_var.get()))

        # Toggle Button
        toggle_btn = ctk.CTkButton(main_frame, text="⇄", width=40, command=lambda: print("Toggle"))
        toggle_btn.grid(row=0, column=1, padx=30, pady=10)

        # Translation Frame
        self.translation_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        self.translation_frame.grid(row=0, column=2, padx=0, pady=10)

        to_label = ctk.CTkLabel(self.translation_frame, text="To:", font=("Arial", 13))
        to_label.grid(row=0, column=0, padx=(0, 5), pady=0)

        self.language_var = ctk.StringVar(value="Arabic")
        self.language_menu = ctk.CTkOptionMenu(self.translation_frame, values=[lang.title() for lang in LANGUAGES.values()], variable=self.language_var, width=10, font=("Arial", 13))
        self.language_menu.grid(row=0, column=1, padx=0, pady=0)

        # Camera Capture Frame
        self.cap_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        self.cap_frame.grid(row=1, column=0, pady=(30, 0))

        self.prediction_label = ctk.CTkLabel(self.cap_frame, text="Prediction Paused", font=("Arial", 21), text_color="Red")
        self.prediction_label.grid(row=0, column=0, padx=20, pady=(0, 10), sticky="nsew")

        # Create Canvas inside the cap_frame for video feed
        self.canvas = ctk.CTkCanvas(self.cap_frame, width=880, height=660, bg="#212121", bd=0, highlightthickness=0)
        self.canvas.grid(row=1, column=0, padx=20, pady=(0, 20), sticky="nsew")

        self.capture = cv2.VideoCapture(0)

        self.update_frame()

        # Recognized Text Frame
        self.recognized_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        self.recognized_frame.grid(row=1, column=2)

        # Recognized Text
        self.recognized_label = ctk.CTkLabel(self.recognized_frame, text="Recognized Text", font=("Arial", 21))
        self.recognized_label.grid(row=0, column=0, padx=20, pady=0)

        self.recognized_text_box = ctk.CTkTextbox(self.recognized_frame, width=400, height=120, corner_radius=10, font=("Arial", 18), state="disabled")
        self.recognized_text_box.grid(row=1, column=0, padx=20, pady=(0, 20), sticky="nsew")

        # Translated Text
        self.translated_label = ctk.CTkLabel(self.recognized_frame, text="Translated Text", font=("Arial", 21))
        self.translated_label.grid(row=2, column=0, padx=0, pady=0)

        self.translated_text_box = ctk.CTkTextbox(self.recognized_frame, width=400, height=120, corner_radius=10, font=("Arial", 18), state="disabled")
        self.translated_text_box.grid(row=3, column=0, padx=20, pady=(0, 20), sticky="nsew")

        # Buttons Frame
        buttons_frame = ctk.CTkFrame(self, fg_color="transparent")
        buttons_frame.grid(row=3, column=0, padx=20, pady=(0, 20))

        self.start_button = ctk.CTkButton(buttons_frame, text="Start", height=40, width=80, fg_color="Green", hover_color="#006400", border_spacing=5,
                                     command=self.start)
        self.start_button.grid(row=0, column=0, padx=(75, 10), pady=10, sticky="ew")

        self.stop_button = ctk.CTkButton(buttons_frame, text="Stop", height=40, width=80, fg_color="Red", hover_color="#8B0000", border_spacing=5,
                                     command=self.stop)
        self.stop_button.grid(row=0, column=1, padx=10, pady=10, sticky="ew")

        clear_button = ctk.CTkButton(buttons_frame, text="Clear", height=40, width=80, border_spacing=5, command=self.clear)
        clear_button.grid(row=0, column=2, padx=10, pady=10, sticky="ew")

        translate_button = ctk.CTkButton(buttons_frame, text="Translate", height=40, width=80, border_spacing=5, command=self.translate)
        translate_button.grid(row=0, column=3, padx=10, pady=10, sticky="ew")


    def normalize_keypoints(self, keypoints):
        wrist_x, wrist_y, wrist_z = keypoints[0]
        return [(x - wrist_x, y - wrist_y, z - wrist_z) for (x, y, z) in keypoints]

    def scale_normalize(self, keypoints):
        middle_finger_x, middle_finger_y, middle_finger_z = keypoints[12]
        scale = np.sqrt((middle_finger_x) ** 2 + (middle_finger_y) ** 2 + (middle_finger_z) ** 2)
        return [(x / scale, y / scale, z / scale) for (x, y, z) in keypoints]

    def preprocess_landmarks(self, hand_landmarks):
        keypoints = [(landmark.x, landmark.y, landmark.z) for landmark in hand_landmarks.landmark]
        keypoints = self.normalize_keypoints(keypoints)
        keypoints = self.scale_normalize(keypoints)
        return np.array(keypoints).flatten().reshape(1, -1)

    def update_frame(self):
        if self.sign_to_text: 
            ret, frame = self.capture.read()

            if not ret:
                return

            frame_rgb = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)

            if self.running:
                results_hands = hands.process(frame_rgb)
                if results_hands.multi_hand_landmarks:
                    current_time = time.time()
                    for hand_id, hand_landmarks in enumerate(results_hands.multi_hand_landmarks):
                        landmarks = self.preprocess_landmarks(hand_landmarks)

                        if len(results_hands.multi_hand_landmarks) == 2:
                            if self.model_type_single: 
                                self.last_write_time = 0
                                self.frame_buffer.clear()
                                self.last_character = None
                                self.model_type_single = False

                            if self.double_hand_model != None:
                                hand_label = None
                                if results_hands.multi_handedness:
                                    hand_label = results_hands.multi_handedness[hand_id].classification[0].label

                                if hand_label == "Right":
                                    self.rh_frames.append(landmarks)
                                elif hand_label == "Left":
                                    self.lh_frames.append(landmarks)

                                if len(self.rh_frames) > 20:
                                    self.rh_frames = self.rh_frames[-20:]

                                if len(self.lh_frames) > 20:
                                    self.lh_frames = self.lh_frames[-20:]

                                if len(self.lh_frames) == 20 and len(self.rh_frames) == 20:
                                    for i in range(20):
                                        self.frame_buffer.append(self.rh_frames[i])
                                        self.frame_buffer.append(self.lh_frames[i])

                                    model_input = np.array(self.frame_buffer).flatten().reshape(1, 20, 126)
                                    prediction_probabilities = self.double_hand_model.predict(model_input)
                                    predicted_index = np.argmax(prediction_probabilities)
                                    confidence = prediction_probabilities[0][predicted_index]

                                    if confidence >= 0.9: 
                                        predicted_label = self.double_hand_label_encoder.inverse_transform([predicted_index])[0]
                                        self.prediction_label.configure(text=f"{predicted_label} ({confidence*100:.2f})", text_color="green")
                                        self.current_character = predicted_label
                                        if current_time - self.last_write_time >= self.write_interval and self.current_character is self.last_character:
                                            if len(self.seq) != 0 and (len(predicted_label) > 1 or len(self.seq[-1]) > 1):
                                                self.write_text(" ")
                                            self.seq.append(predicted_label)
                                            self.write_text(predicted_label)
                                            self.last_write_time = current_time
                                            self.last_character = None
                                            self.frame_buffer.clear()
                                            self.rh_frames.clear()
                                            self.lh_frames.clear()
                                        else:
                                            if self.last_character is not self.current_character:
                                                self.last_write_time = current_time 
                                            self.last_character = self.current_character
                                    else:
                                        self.prediction_label.configure(text="Low confidence", text_color="green")

                                    self.frame_buffer.clear()
                        elif len(results_hands.multi_hand_landmarks) == 1:
                            if not self.model_type_single:
                                self.frame_buffer.clear()
                                self.last_write_time = 0
                                self.last_character = None
                                self.model_type_single = True

                            if self.single_hand_model != None:
                                self.frame_buffer.append(landmarks)

                                if len(self.frame_buffer) > 20:
                                    self.frame_buffer = self.frame_buffer[-20:]

                                if len(self.frame_buffer) == 20:
                                    model_input = np.array(self.frame_buffer).reshape(1, 20, 63)
                                    prediction_probabilities = self.single_hand_model.predict(model_input)
                                    predicted_index = np.argmax(prediction_probabilities)
                                    confidence = prediction_probabilities[0][predicted_index]

                                    if confidence >= 0.9:
                                        predicted_label = self.single_hand_label_encoder.inverse_transform([predicted_index])[0]
                                        self.prediction_label.configure(text=f"{predicted_label} ({confidence*100:.2f})", text_color="green")
                                        self.current_character = predicted_label
                                        if current_time - self.last_write_time >= self.write_interval and self.current_character is self.last_character:
                                            if len(self.seq) != 0 and (len(predicted_label) > 1 or len(self.seq[-1]) > 1):
                                                self.write_text(" ")
                                            self.seq.append(predicted_label)
                                            self.write_text(predicted_label)
                                            self.last_write_time = current_time
                                            self.last_character = None
                                            self.frame_buffer.clear()
                                        else:
                                            if self.last_character is not self.current_character:
                                                self.last_write_time = current_time 
                                            self.last_character = self.current_character
                                    else:
                                        self.prediction_label.configure(text="Low confidence", text_color="green")
                        mp_drawing.draw_landmarks(frame_rgb, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                else:
                    self.frame_buffer.clear()
                    self.rh_frames.clear()
                    self.lh_frames.clear()
            else:
                self.prediction_label.configure(text="Prediction Paused", text_color="red")

            frame_resized = cv2.resize(frame_rgb, (880, 660))

            image = Image.fromarray(frame_resized)
            photo = ImageTk.PhotoImage(image=image)

            self.canvas.create_image(0, 0, image=photo, anchor=tk.NW)

            self.canvas.image = photo

            self.after(30, self.update_frame)

    
    def write_text(self, word):
        self.recognized_text_box.configure(state="normal")
        self.recognized_text_box.insert(ctk.END, word)
        self.recognized_text_box.configure(state="disabled")

    def start(self):
        self.running = True
        self.prediction_label.configure(text="Prediction Started", text_color="green")

    def stop(self):
        self.running = False

    def clear(self):
        self.translated_text_box.configure(state="normal")
        self.recognized_text_box.configure(state="normal")
        self.translated_text_box.delete("0.0", ctk.END)
        self.recognized_text_box.delete("0.0", ctk.END)
        self.translation_text = ""
        self.recognized_text = ""
        self.translated_text_box.configure(state="disabled")
        self.recognized_text_box.configure(state="disabled")
        self.frame_buffer.clear()
        self.rh_frames.clear()
        self.lh_frames.clear()
        self.seq.clear()

    def load_models(self, language):
        model_info = self.models[language]
    
        # Load single hand model and label encoder
        if model_info["single_hand_model"] is not None and model_info["single_hand_label_encoder"] is not None:
            try:
                self.single_hand_model = tf.keras.models.load_model(model_info["single_hand_model"])
                self.single_hand_label_encoder = joblib.load(model_info["single_hand_label_encoder"])
            except (OSError, IOError) as e:
                print(f"Error loading single hand model or label encoder for {language}: {e}")
                self.single_hand_model = None
                self.single_hand_label_encoder = None
        else:
            self.single_hand_model = None
            self.single_hand_label_encoder = None

        # Load double hand model and label encoder
        if model_info["double_hand_model"] is not None and model_info["double_hand_label_encoder"] is not None:
            try:
                self.double_hand_model = tf.keras.models.load_model(model_info["double_hand_model"])
                self.double_hand_label_encoder = joblib.load(model_info["double_hand_label_encoder"])
            except (OSError, IOError) as e:
                print(f"Error loading double hand model or label encoder for {language}: {e}")
                self.double_hand_model = None
                self.double_hand_label_encoder = None
        else:
            self.double_hand_model = None
            self.double_hand_label_encoder = None

       
    def on_language_change(self, language):
        self.load_models(language)
        self.clear()

    def mapping(self, seq, keywords, i=0, matched=None):
        if matched is None:
            matched = []

        if not seq:
            return matched
    
        candidates = {key: words for key, words in keywords.items() if len(words) > i and words[i] == seq[0]}

        if not candidates:
            matched.append(seq[0])
            return self.mapping(seq[1:], sign_map, 0, matched)

        for candidate, words in candidates.items():
            if len(words) == len(seq[:len(words)]) and words == seq[:len(words)]:
                matched.append(candidate)
                return self.mapping(seq[len(words):], sign_map, 0, matched)

        return self.mapping(seq, candidates, i + 1, matched)

    def translate(self):
        self.running = False
        sentence = self.mapping(self.seq, sign_map)

        for word in sentence:
            if len(word) == 1:
                self.recognized_text += word
            else:
                if self.recognized_text and self.recognized_text[-1] != ' ':
                    self.recognized_text += ' '
                self.recognized_text += word + ' '
                
        self.recognized_text = self.recognized_text.strip()

        self.recognized_text_box.configure(state="normal")
        self.recognized_text_box.delete("0.0", ctk.END)
        self.write_text(self.recognized_text)
        threading.Thread(target=asyncio.run(self.translate_background)).start()

    async def translate_background(self):
        src_lang = sign_language[self.sign_language_var.get()]
        dest_lang = self.language_var.get()
        if self.recognized_text:
            
            result = await translator.translate(self.recognized_text, src=src_lang, dest=dest_lang)
            self.translation_text = result.text

            
            self.after(1, self.update_translation_text_box)

    def update_translation_text_box(self):
        self.translated_text_box.configure(state="normal")
        self.translated_text_box.delete("0.0", ctk.END)
        self.translated_text_box.insert(ctk.END, self.translation_text)
        self.translated_text_box.configure(state="disabled")

    def toggle(self):
        self.sign_to_text = not self.sign_to_text

        if self.sign_to_text:
            if self.capture is None: 
                self.capture = cv2.VideoCapture(0)
            
            self.update_frame()

            self.cap_frame.grid(row=1, column=0)

            self.prediction_label.configure(text="Prediction Paused", text_color="red")

            self.sign_language_frame.grid(column=0)
            self.translation_frame.grid(column=2)
            self.recognized_frame.grid(row=1, column=2)

            if not hasattr(self, "text_to_translation_frame"):
                self.text_to_translation_frame = ctk.CTkFrame(self, fg_color="transparent")
                self.text_to_translation_frame.grid(row=2, column=2) 

            self.start_button.grid(row=0, column=0)
            self.stop_button.grid(row=0, column=1)

        else:
            if self.capture is not None:  #
                self.capture.release()
                self.capture = None

            self.cap_frame.grid(row=1, column=2)

            self.prediction_label.configure(text="")

            self.sign_language_frame.grid(column=2)
            self.translation_frame.grid(column=0)
            self.recognized_frame.grid(row=1, column=0)

            self.start_button.grid_forget()
            self.stop_button.grid_forget()


    def close(self):
        if self.capture.isOpened():
            self.capture.release()

        self.destroy()
        
app = App()
app.mainloop()