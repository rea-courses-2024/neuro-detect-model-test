import cv2
from ultralytics import YOLO
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import numpy as np
import base64
from collections import deque

# Инициализация модели YOLO
model = YOLO('yolov8m-seg.pt')

# Инициализация Dash приложения
app = dash.Dash(__name__)

# Хранение количества детекций
detection_history = deque(maxlen=10)  # Хранить последние 10 значений
time_intervals = deque(maxlen=10)  # Хранить временные метки

# Макет приложения
app.layout = html.Div([
    html.H1("Система мониторинга в реальном времени"),
    dcc.Graph(id='live-update-graph'),
    dcc.Interval(
        id='interval-component',
        interval=1 * 1000,  # обновление каждую секунду
        n_intervals=0
    )
])


# Функция для захвата и обработки видео
def capture_frame():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if ret:
        result = model(frame, iou=0.4, conf=0.6)
        detect_frame = result[0].plot()
        _, buffer = cv2.imencode('.jpg', detect_frame)
        frame_b64 = base64.b64encode(buffer).decode('utf-8')

        # Подсчет детектированных объектов
        detections = len(result[0].boxes)
        cap.release()
        return frame_b64, detections
    cap.release()
    return None, 0


# Обновление графика в Dash
@app.callback(Output('live-update-graph', 'figure'),
              Input('interval-component', 'n_intervals'))
def update_graph(n):
    frame_b64, detections = capture_frame()

    # Обновление истории детекций
    detection_history.append(detections)
    time_intervals.append(n)

    if len(time_intervals) == 10:  # Каждые 10 секунд
        avg_detections = np.mean(detection_history)
        detection_history.clear()  # Сбросить историю
        time_intervals.clear()  # Сбросить временные метки
    else:
        avg_detections = np.mean(detection_history) if detection_history else 0

    if frame_b64:
        return {
            'data': [{
                'x': list(range(len(detection_history))),  # Индексы для графика
                'y': [avg_detections] * len(detection_history),  # Средние детекции
                'type': 'bar',
                'name': 'Средние детекции'
            }],
            'layout': {
                'images': [{
                    'source': frame_b64,
                    'xref': 'paper',
                    'yref': 'paper',
                    'x': 0,
                    'y': 1,
                    'sizex': 1,
                    'sizey': 1,
                    'xanchor': 'left',
                    'yanchor': 'top',
                    'layer': 'below'
                }],
                'title': 'Мониторинг в реальном времени',
                'xaxis': {'title': 'Время', 'visible': True},
                'yaxis': {'title': 'Среднее количество детекций', 'visible': True},
            }
        }
    return {}


if __name__ == '__main__':
    app.run_server(debug=True)
