# Проект: ABI

## Описание

Этот проект предназначен для обработки данных из CSV-файла, генерации текстовых промптов, получения их эмбеддингов с помощью модели BERT и сохранения этих данных в базе данных Milvus.

## Структура проекта

- `main.py`: Основной скрипт для обработки данных и взаимодействия с Milvus.
- `input.csv`: Входной CSV-файл с данными.
- `Dockerfile`: Dockerfile для создания образа проекта.
- `docker-compose.yml`: Конфигурационный файл Docker Compose для автоматизации запуска проекта и Milvus.
- `requirements.txt`: Файл с зависимостями проекта.

## Установка и запуск

### Требования

- Docker
- Docker Compose

### Шаги по установке и запуску

1. Клонируйте репозиторий:
`git clone https://github.com/DudkevichD/abi.git`
2. Перейдите в папку abi: `cd abi`
3. Запустите Docker Compose: `docker-compose up --build`
  
