import pickle
import cv2
import mediapipe as mp
import numpy as np
import time
import pyttsx3

model_dict = pickle.load(open('./model_ASL.p', 'rb'))
model = model_dict['model']

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

cap = cv2.VideoCapture(0)  

if not cap.isOpened():  
    print("Kamera başlatılamadı!")
    exit()

labels_dict = {
    0: 'A',
    1: 'B',
    2: 'C',
    3: 'D',
    4: 'E',
    5: 'F',
    6: 'G',
    7: 'H',
    8: 'I',
    9: 'J',
    10: 'K',
    11: 'L',
    12: 'M',
    13: 'N',
    14: 'O',
    15: 'P',
    16: 'Q',
    17: 'R',
    18: 'S',
    19: 'T',
    20: 'U',
    21: 'V',
    22: 'W',
    23: 'X',
    24: 'Y',
    25: 'Z'
}

predicted_letters = []  # List to store predicted letters
sentence = ""
start_time = time.time()  # Start time for the 2-second window

engine = pyttsx3.init()

while True:
    ret, frame = cap.read()  
    if not ret:
        print("Kare okunamadı!")
        break

    H, W, _ = frame.shape  

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS, 
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in results.multi_hand_landmarks:
            data_aux = []
            x_ = []
            y_ = []

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))
            
            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10

            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            # Ensure data_aux has 84 features
            while len(data_aux) < 84:
                data_aux.append(0.0)  # Padding with zeros if necessary

            prediction = model.predict([np.asarray(data_aux)])
            
            if isinstance(prediction[0], int) and prediction[0] in labels_dict:
                predicted_character = labels_dict[prediction[0]]
            else:
                predicted_character = prediction[0]
                predicted_letters.append(predicted_character)
                print(predicted_character)
                cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,cv2.LINE_AA)
                
    # Check if 2 seconds have passed
    if time.time() - start_time > 4:
        if len(predicted_letters) > 0:
            # Count occurrences of each letter
            letter_counts = {letter: predicted_letters.count(letter) for letter in labels_dict.values()}
            # Get the most predicted letter
            most_predicted_letter = max(letter_counts, key=letter_counts.get)
            # Append the most predicted letter to the sentence
            sentence += most_predicted_letter
            # Clear predicted_letters list for the next window
            print("most predicted letter:", most_predicted_letter)
            predicted_letters = []
            # Update start time
            print(sentence)
        else:
            # If there's no input, clear both sentence and most_predicted_letter
            sentence = ""
            most_predicted_letter = ""
        start_time = time.time()

    cv2.putText(frame, "Sentence: " + sentence, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, "Press 's' to Speak", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    key = cv2.waitKey(1) & 0xFF
    cv2.imshow('frame', frame)
    if key == ord('q'):
        break
    elif key == ord(' '):  # Space button
        sentence += ' '  # Add space character to the sentence
    elif key == 8:  # Backspace key
        if len(sentence) > 0:
            sentence = sentence[:-1]  # Delete the last character
    elif key == ord('s'):  # Speak button
        engine.say(sentence)
        engine.runAndWait()

cap.release()
cv2.destroyAllWindows()
