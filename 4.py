import cv2
from ultralytics import YOLO
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from collections import Counter
import os

# Инициализация модели YOLO
model = YOLO('yolov8m-seg.pt')

# Инициализация Dash приложения
app = dash.Dash(__name__)

# Список классов объектов
class_names = model.names
image_dir = 'images'

# Обработка изображений один раз
def process_images():
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    object_types = []

    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        frame = cv2.imread(image_path)
        if frame is not None:
            result = model(frame, iou=0.4, conf=0.6)

            for box in result[0].boxes:
                object_types.append(class_names[int(box.cls.item())])

    return object_types

# Макет приложения
app.layout = html.Div([
    html.H1("Распределение типов объектов"),
    dcc.Graph(id='object-distribution-pie'),
    html.Button('Обработать изображения', id='process-button', n_clicks=0),
    html.Div(id='output-container')
])

# Обновление графиков в Dash
@app.callback(
    Output('object-distribution-pie', 'figure'),
    Output('output-container', 'children'),
    Input('process-button', 'n_clicks')
)
def update_graph(n_clicks):
    if n_clicks > 0:
        object_types = process_images()
        object_count = Counter(object_types)
        labels = list(object_count.keys())
        values = list(object_count.values())

        object_distribution_pie = {
            'data': [{
                'labels': labels,
                'values': values,
                'type': 'pie',
                'name': 'Распределение типов объектов'
            }],
            'layout': {
                'title': 'Распределение типов объектов',
            }
        }

        return object_distribution_pie, f'Обработано объектов: {sum(values)}'

    return {}, 'Нажмите кнопку для обработки изображений'

if __name__ == '__main__':
    app.run_server(debug=True)
