import base64
from flask import Flask, render_template, Response, request, jsonify, send_file, send_from_directory
from flask_socketio import SocketIO
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import os
import time
import numpy as np

app = Flask(__name__, static_folder='static')
socketio = SocketIO(app)
CORS(app)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi'}
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

ANNOTATED_IMAGES_FOLDER = 'annotated_images'
if not os.path.exists(ANNOTATED_IMAGES_FOLDER):
    os.makedirs(ANNOTATED_IMAGES_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DETECTION_STARTED'] = False
stop_detection_flag = False

model = YOLO('best.pt')  # Asegúrate de tener el modelo YOLO en el mismo directorio

# Crear un conjunto global para rastrear IDs únicos de detección
unique_detection_ids = set()

# Crear un contador global para el número de detecciones únicas
detection_counter = 0
tracking_threshold = 50  # Distancia máxima para considerar que una nueva detección es la misma que una anterior

def euclidean_distance(box1, box2):
    return np.linalg.norm(np.array(box1) - np.array(box2))

def detect_objects(frame):
    global unique_detection_ids
    global detection_counter
    results = model(frame)
    detections = []
    new_detection_ids = set()

    # Contadores de tipos de detección
    papaya_madura_count = 0
    papaya_verde_count = 0

    for result in results:
        for box in result.boxes:
            if box.conf > 0.5:
                label = model.names[int(box.cls)]
                confidence = box.conf.item()
                bbox = box.xyxy[0].tolist()
                bbox = [round(coord, 2) for coord in bbox]
                detection_id = f"{label}_{'_'.join(map(str, bbox))}"

                # Verificar si la detección está cerca de una detección anterior
                is_new_detection = True
                for existing_id in unique_detection_ids:
                    existing_bbox = list(map(float, existing_id.split('_')[1:]))
                    if euclidean_distance(bbox, existing_bbox) < tracking_threshold:
                        is_new_detection = False
                        break

                if is_new_detection:
                    unique_detection_ids.add(detection_id)
                    new_detection_ids.add(detection_id)
                    detection_counter += 1

                # Contar las detecciones por tipo
                if label == "Papaya Madura":
                    papaya_madura_count += 1
                elif label == "Papaya Verde":
                    papaya_verde_count += 1

                detections.append({
                    'label': label,
                    'confidence': confidence,
                    'bbox': bbox
                })

    # Crear el mensaje a mostrar
    detection_summary = f"{papaya_madura_count} Papayas Maduras: , {papaya_verde_count} Papayas Verdes: "

    return detections, detection_counter, len(new_detection_ids), detection_summary

def save_annotated_frame(frame, file_path):
    """Guardar el marco anotado como imagen en la ruta especificada."""
    cv2.imwrite(file_path, frame)

def gen(video_source):
    global stop_detection_flag
    cap = cv2.VideoCapture(video_source)
    frame_count = 0  # Contador de cuadros para nombrar archivos de manera única
    frame = None  # Inicializar frame

    while cap.isOpened():
        if stop_detection_flag:
            cap.release()
            if frame is not None:
                # Guardar solo la última imagen si frame no es None
                last_frame_path = os.path.join(ANNOTATED_IMAGES_FOLDER, 'last_frame.jpg')
                save_annotated_frame(frame, last_frame_path)
            break

        ret, frame = cap.read()
        if not ret:
            break

        # Detección de objetos
        start_time = time.time()
        detections, unique_detections_count, new_detections, detection_summary = detect_objects(frame)
        inference_time = (time.time() - start_time) * 1000  # Tiempo de inferencia en milisegundos

        # Preprocesamiento y postprocesamiento (placeholder, ajustar según sea necesario)
        preprocess_time = 0.0
        postprocess_time = 0.0

        num_detections = len(detections)
        image_shape = frame.shape[:2]

        # Enviar datos al cliente a través de SocketIO
        socketio.emit('detection', {
            'detections': detections,
            'num_detections': num_detections,
            'unique_detections_count': unique_detections_count,
            'detection_counter': detection_counter,
            'detection_summary': detection_summary,
            'preprocess_time': preprocess_time,
            'inference_time': inference_time,
            'postprocess_time': postprocess_time,
            'image_shape': image_shape
        })

        # Dibujar detecciones en el frame
        for detection in detections:
            bbox = detection['bbox']
            label = detection['label']
            confidence = detection['confidence']
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
            cv2.putText(frame, f'{label} {confidence:.2f}', (int(bbox[0]), int(bbox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)

        # Agregar información adicional en el video
        cv2.putText(frame, f'Detecciones: {num_detections}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                    cv2.LINE_AA)
        cv2.putText(frame, f'Estimacion: {unique_detections_count}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255), 2, cv2.LINE_AA)
        #cv2.putText(frame, f'Counter: {detection_counter}', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1,
                   # (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, detection_summary, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 0, 0), 2, cv2.LINE_AA)

        # Guardar los resultados de detección
        frame_count += 1
        annotated_image_path = os.path.join(ANNOTATED_IMAGES_FOLDER, f'frame_{frame_count}.jpg')
        save_annotated_frame(frame, annotated_image_path)

        # Guardar continuamente la última imagen procesada
        last_frame_path = os.path.join(ANNOTATED_IMAGES_FOLDER, 'last_frame.jpg')
        save_annotated_frame(frame, last_frame_path)

        # Reducir el tamaño de los frames antes de enviar
        frame = cv2.resize(frame, (640, 480))

        # Codificar el frame como JPEG y enviar al cliente
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    global stop_detection_flag
    stop_detection_flag = False
    return Response(gen(0), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/file_video_feed')
def file_video_feed():
    global stop_detection_flag
    stop_detection_flag = False
    video_path = os.path.join(UPLOAD_FOLDER, 'video.mp4')
    return Response(gen(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'success': False, 'error': 'No file part'})

    file = request.files['video']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No selected file'})

    if file and allowed_file(file.filename):
        filename = 'video.mp4'
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return jsonify({'success': True})

    return jsonify({'success': False, 'error': 'File not allowed'})

@app.route('/stop_detection', methods=['POST'])
def stop_detection():
    global stop_detection_flag
    global unique_detection_ids
    global detection_counter
    stop_detection_flag = True

    # Reiniciar las variables de conteo
    unique_detection_ids = set()
    detection_counter = 0

    # Ruta al archivo de imagen que quieres enviar
    last_frame_path = 'annotated_images/last_frame.jpg'

    # Enviar el archivo como respuesta
    return send_file(
        last_frame_path,
        mimetype='image/jpeg',
        download_name='last_frame.jpg'  # Cambia attachment_filename por download_name en Flask 2.x
    )

@app.route('/annotated_images/<path:filename>')
def serve_annotated_image(filename):
    return send_from_directory(ANNOTATED_IMAGES_FOLDER, filename)

@app.route('/get_image_paths')
def get_image_paths():
    image_files = os.listdir(ANNOTATED_IMAGES_FOLDER)
    image_paths = [f'/annotated_images/{filename}' for filename in image_files]
    return jsonify(image_paths)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=8080, debug=True, allow_unsafe_werkzeug=True)

