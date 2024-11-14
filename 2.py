import cv2
from ultralytics import YOLO
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import os

# Инициализация модели YOLO
model = YOLO('yolov8m-seg.pt')

# Инициализация Dash приложения
app = dash.Dash(__name__)

image_dir = 'images'

# Обработка изображений один раз
def process_images():
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    details = []

    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        frame = cv2.imread(image_path)
        if frame is not None:
            result = model(frame, iou=0.4, conf=0.6)
            count = len(result[0].boxes)
            details.append({'filename': image_file, 'detections': count})

    return details

# Макет приложения
app.layout = html.Div([
    html.H1("Гистограмма количества детекций по изображениям"),
    dcc.Graph(id='detection-histogram'),
    html.Button('Обработать изображения', id='process-button', n_clicks=0),
    html.Div(id='output-container')
])

# Обновление графиков в Dash
@app.callback(
    Output('detection-histogram', 'figure'),
    Output('output-container', 'children'),
    Input('process-button', 'n_clicks')
)
def update_graph(n_clicks):
    if n_clicks > 0:
        details = process_images()
        detection_histogram = {
            'data': [{
                'x': [d['filename'] for d in details],
                'y': [d['detections'] for d in details],
                'type': 'bar',
                'name': 'Количество детекций по изображениям'
            }],
            'layout': {
                'title': 'Количество детекций по изображениям',
                'xaxis': {'title': 'Имя файла', 'tickangle': -45},
                'yaxis': {'title': 'Количество детекций'},
            }
        }

        return detection_histogram, f'Обработано изображений: {len(details)}'

    return {}, 'Нажмите кнопку для обработки изображений'

if __name__ == '__main__':
    app.run_server(debug=True)
