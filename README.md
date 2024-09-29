## README
Этот веб-сервис предоставляет API для загрузки видео, обработки задач и получения результатов. Он может быть развернут с использованием Docker. Ниже приведена документация по его использованию.

▎Запуск с помощью Docker

   
   docker build -t my-video-service .
   

2. Запустите контейнер:
   
   docker run -p 5000:5000 my-video-service
   

▎API Эндпоинты

▎1. /set_video_download

- Метод: POST
- Описание: Загружает видео по указанной ссылке и добавляет задачу на обработку.
- Параметры запроса (JSON):
  - download_url (string): URL для загрузки видео.
  - filename (string): Имя файла для сохранения.
  - md5 (string, optional): MD5 хэш для проверки целостности файла.
- Ответ:
  - Успешно: {"task_id": "<task_id>"}
  - Ошибка: {"error": "<описание ошибки>"}

▎2. /task_status

- Метод: POST
- Описание: Проверяет статус задачи обработки видео.
- Параметры запроса (JSON):
  - task_id (string): Идентификатор задачи.
- Ответ:
  - Успешно: JSON объект с результатами задачи.
  - Ошибка: {"result": "error", "error_str": "<описание ошибки>"}

▎3. /download/<filename>

- Метод: GET
- Описание: Позволяет скачать обработанный файл по его имени.
- Параметры URL:
  - <filename> (string): Имя файла для скачивания.
- Ответ: Файл для скачивания.

▎Пример использования

▎Загрузка видео

curl -X POST http://localhost:5000/set_video_download \
-H "Content-Type: application/json" \
-d '{"download_url": "http://example.com/video.mp4", "filename": "video.mp4", "md5": "abcd1234"}'


▎Проверка статуса задачи

curl -X POST http://localhost:5000/task_status \
-H "Content-Type: application/json" \
-d '{"task_id": "your_task_id_here"}'


▎Скачивание файла

curl -O http://localhost:5000/download/video.mp4
