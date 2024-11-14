import cv2
from ultralytics import YOLO
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from collections import deque
import os

# Инициализация модели YOLO
model = YOLO('yolov8m-seg.pt')

# Инициализация Dash приложения
app = dash.Dash(__name__)

# Хранение общего количества детекций
detection_history = deque(maxlen=10)  # Хранить последние 10 значений
image_dir = 'images'

# Обработка изображений один раз
def process_images():
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    detections_count = []

    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        frame = cv2.imread(image_path)
        if frame is not None:
            result = model(frame, iou=0.4, conf=0.6)
            count = len(result[0].boxes)
            detections_count.append(count)

    return detections_count

# Макет приложения
app.layout = html.Div([
    html.H1("Общее количество детекций"),
    dcc.Graph(id='live-update-graph'),
    html.Button('Обработать изображения', id='process-button', n_clicks=0),
    html.Div(id='output-container')
])

# Обновление графиков в Dash
@app.callback(
    Output('live-update-graph', 'figure'),
    Output('output-container', 'children'),
    Input('process-button', 'n_clicks')
)
def update_graph(n_clicks):
    if n_clicks > 0:
        detection_results = process_images()
        total_detections = sum(detection_results) if detection_results else 0
        detection_history.append(total_detections)

        live_update_figure = {
            'data': [{
                'x': list(range(len(detection_history))),
                'y': list(detection_history),
                'type': 'bar',
                'name': 'Общее количество детекций'
            }],
            'layout': {
                'title': 'Мониторинг изображений',
                'xaxis': {'title': 'Время', 'visible': True},
                'yaxis': {'title': 'Количество детекций', 'visible': True},
            }
        }

        return live_update_figure, f'Общее количество детекций: {total_detections}'

    return {}, 'Нажмите кнопку для обработки изображений'

if __name__ == '__main__':
    app.run_server(debug=True)
