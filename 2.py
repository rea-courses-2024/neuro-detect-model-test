import argparse
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description='Система понимания для YOLO')

    # Скалярные параметры
    parser.add_argument('--epochs', type=int, default=350, help='Количество эпох обучения')
    parser.add_argument('--batch', type=int, default=8, help='Размер пакета для обучения')
    parser.add_argument('--imgsz', type=int, default=640, help='Размер изображения')
    parser.add_argument('--patience', type=int, default=200, help='Параметр терпения для ранней остановки')

    # Непосредственно пользовательские параметры
    parser.add_argument('--data', type=str, required=True, help='Путь к файлу данных (data.yaml)')
    parser.add_argument('--model', type=str, default='yolov8m-seg.pt', help='Путь к модели YOLO')

    # Логические параметры
    parser.add_argument('--verbose', action='store_true', help='Выводить подробную информацию')
    parser.add_argument('--use_augmentation', action='store_true', help='Использовать аугментацию данных')

    # Параметры с несколькими значениями
    parser.add_argument('--optimizers', nargs='+', default=['adam'], help='Список оптимизаторов для использования')
    parser.add_argument('--metrics', nargs='+', default=['loss', 'mAP'], help='Список метрик для оценки')

    # Строковый параметр
    parser.add_argument('--output_dir', type=str, default='./output', help='Директория для сохранения результатов')

    args = parser.parse_args()

    # Создание и обучение модели YOLO
    model = YOLO(args.model)

    # Параметры для обучения
    train_params = {
        'data': args.data,
        'epochs': args.epochs,
        'imgsz': args.imgsz,
        'patience': args.patience,
        'batch': args.batch,
        'verbose': args.verbose,
        'augment': args.use_augmentation,
        'optimizer': args.optimizers[0],  # Используем первый оптимизатор из списка
        'metrics': args.metrics
    }

    # Обучение модели
    model.train(**train_params)

    print(f'Обучение модели {args.model} завершено. Результаты сохранены в {args.output_dir}')


if __name__ == '__main__':
    main()
