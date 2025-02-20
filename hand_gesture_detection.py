# import necessary packages

import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import tensorflow as tf

# initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Load the gesture recognizer model
from keras.utils import custom_object_scope

# Definir la capa personalizada (o importarla si está en otro script)
class KerasModelWrapper(tf.keras.Model):  # Asegúrate de que hereda de Model
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def call(self, inputs):
        return inputs  # Lógica real del modelo

# Cargar el modelo dentro de un custom_object_scope
with custom_object_scope({'KerasModelWrapper': KerasModelWrapper}):
    model = tf.keras.models.load_model("mp_hand_gesture_manual.h5")

print(model.input_dtype)
# model = tensorflow.saved_model.load('mp_hand_gesture_manual')

# Load class names
f = open('gesture.names', 'r')
classNames = f.read().split('\n')
f.close()
print(classNames)


# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read each frame from the webcam
    _, frame = cap.read()

    x, y, c = frame.shape

    # Flip the frame vertically
    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get hand landmark prediction
    result = hands.process(framergb)

    # print(result)
    
    className = ''

    # post process the result
    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                # print(id, lm)
                lmx = int(lm.x * x)
                lmy = int(lm.y * y)

                landmarks.append([lmx, lmy])

            # Drawing landmarks on frames
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)
            

            # Convierte landmarks a numpy array
            landmarks_array = np.array(landmarks, dtype=np.float32)

            # Asegúrate de que tenga la forma esperada (None, 21, 2)
            landmarks_array = np.expand_dims(landmarks_array, axis=0)

            # print(landmarks)
            # Predict gesture
            prediction = model.predict(landmarks_array)
            # print(prediction)
            classID = np.argmax(prediction)
            className = classNames[classID]

    # show the prediction on the frame
    cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (0,0,255), 2, cv2.LINE_AA)

    # Show the final output
    cv2.imshow("Output", frame) 

    if cv2.waitKey(1) == ord('q'):
        break

# release the webcam and destroy all active windows
cap.release()

cv2.destroyAllWindows()
