import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import time
st.markdown("### TECNOLOGIAS DE RECONHECIMENTO FACIAL: estratégia para avaliação do engajamento e aproveitamento de alunos do ensino fundamental em escolas públicas")
# Inicializa mediapipe
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Função para determinar se o olhar está focado à frente
def is_looking_forward(face_landmarks):
    left_eye = face_landmarks[0]
    right_eye = face_landmarks[1]
    nose_tip = face_landmarks[2]

    eye_center = (np.array(left_eye) + np.array(right_eye)) / 2
    direction = eye_center - np.array(nose_tip)

    return abs(direction[0]) < 10  # Ajuste o limiar conforme necessário

# Inicializa a captura de vídeo
cap = None

# Variáveis para contar o tempo de atenção e distração
total_attention_time = 0
total_distraction_time = 0
current_attention_time = 0
previous_time = 0
attention_state = None
start_attention_time = 0

# Cria o botão para iniciar a webcam
if st.button('Iniciar Câmera'):
    if cap is None:
        cap = cv2.VideoCapture(0)
        previous_time = time.time()  # Inicia a contagem do tempo
    else:
        cap.release()
        cap = None

# Exibe a imagem da webcam se ela estiver ligada
if cap is not None and cap.isOpened():
    stframe = st.empty()
    attention_text = st.empty()
    distraction_text = st.empty()

    with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.warning("Não foi possível capturar o vídeo.")
                break

            # Tempo atual
            current_time = time.time()
            elapsed_time = current_time - previous_time
            previous_time = current_time

            # Converter para RGB
            rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detectar rostos usando mediapipe
            results = face_detection.process(rgb_img)

            if results.detections:
                for detection in results.detections:
                    keypoints = detection.location_data.relative_keypoints
                    left_eye = keypoints[0]
                    right_eye = keypoints[1]
                    nose_tip = keypoints[2]

                    ih, iw, _ = frame.shape
                    left_eye = (int(left_eye.x * iw), int(left_eye.y * ih))
                    right_eye = (int(right_eye.x * iw), int(right_eye.y * ih))
                    nose_tip = (int(nose_tip.x * iw), int(nose_tip.y * ih))

                    # Verificar se o olhar está focado à frente
                    if is_looking_forward([left_eye, right_eye, nose_tip]):
                        color = (0, 255, 0)  # Verde para atenção
                        if attention_state != 'attention':
                            attention_state = 'attention'
                            start_attention_time = time.time()  # Inicia o tempo atual de atenção
                    else:
                        color = (0, 0, 255)  # Vermelho para distração
                        if attention_state == 'attention':
                            attention_state = 'distraction'
                            current_attention_time = time.time() - start_attention_time
                            total_attention_time += current_attention_time

                    bboxC = detection.location_data.relative_bounding_box
                    x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # Atualiza o tempo de atenção ou distração
            if attention_state == 'attention':
                total_attention_time += elapsed_time
                current_attention_time = time.time() - start_attention_time
            elif attention_state == 'distraction':
                total_distraction_time += elapsed_time

            # Converter tempos para minutos
            attention_time_minutes = total_attention_time / 60
            distraction_time_minutes = total_distraction_time / 60

            # Exibir a imagem processada
            stframe.image(frame, channels="BGR")

            # Exibir os tempos de atenção e distração
            attention_text.markdown(f"**Tempo Total de Atenção:** {attention_time_minutes:.2f} minutos")
            distraction_text.markdown(f"**Tempo Total de Distração:** {distraction_time_minutes:.2f} minutos")

    cap.release()

else:
    st.write("Webcam desligada. Clique no botão acima para ligar.")
