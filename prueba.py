# Importar paquetes necesarios
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
from keras.utils import custom_object_scope

# Inicializar MediaPipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Definir la capa personalizada si es necesaria

class KerasModelWrapper(tf.keras.Model):
    def __init__(self, tfm_model, trainable=True, dtype=None, **kwargs):
        super().__init__(trainable=trainable, dtype=dtype, **kwargs)
        self.tfm_model = tfm_model

    def call(self, inputs):
        return self.tfm_model.signatures["serving_default"](flatten_2_input=inputs)["dense_16"]

    def get_config(self):
        return {"trainable": self.trainable}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

# Cargar el modelo con custom_object_scope
with tf.keras.utils.custom_object_scope({'KerasModelWrapper': KerasModelWrapper}):
    model = tf.keras.models.load_model("mp_hand_gesture_manual.h5")


# Leer nombres de gestos y asegurarse de que no haya líneas vacías
with open('gesture.names', 'r') as f:
    classNames = [line.strip() for line in f if line.strip()]
print(classNames)

# Inicializar la webcam
cap = cv2.VideoCapture(0)

while True:
    # Leer frame de la webcam
    ret, frame = cap.read()
    if not ret:
        continue

    h, w, c = frame.shape

    # Voltear el frame y convertir a RGB
    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Obtener predicción de landmarks de MediaPipe
    result = hands.process(framergb)

    className = ''

    # Procesar los resultados
    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                lmx = lm.x  # Normalizado entre 0 y 1
                lmy = lm.y  # Normalizado entre 0 y 1
                landmarks.append([lmx, lmy])  # Usar valores normalizados

            # Dibujar los landmarks
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

        # Convertir a numpy array con la forma correcta
        landmarks_array = np.array(landmarks, dtype=np.float32).reshape(1, 21, 2)

        # Hacer la predicción
        prediction = model.predict(landmarks_array)
        print("Predicción:", prediction)

        # Obtener el índice de la clase con mayor probabilidad
        classID = np.argmax(prediction)
        if classID < len(classNames):
            className = classNames[classID]
        else:
            className = "Desconocido"

    # Mostrar el resultado en el frame
    cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0, 0, 255), 2, cv2.LINE_AA)

    # Mostrar la salida
    cv2.imshow("Output", frame)

    if cv2.waitKey(1) == ord('q'):
        break

# Liberar la cámara y cerrar ventanas
cap.release()
cv2.destroyAllWindows()
