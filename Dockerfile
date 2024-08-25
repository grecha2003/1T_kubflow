# Используем базовый образ Python
FROM python:3.8

# Устанавливаем необходимые пакеты
RUN pip install tensorflow

# Копируем скрипт в контейнер
COPY mnist_train.py /mnist_train.py

# Запускаем скрипт
CMD ["python", "/mnist_train.py"]
