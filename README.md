# RAG Ассистент

RAG-система для поиска по внутренней документации компании. В проекте есть:

- FastAPI backend
- Streamlit frontend
- Weaviate для хранения чанков, эмбеддингов и метаданных
- PostgreSQL для хранения пользователей и сессий

## Установка

```bash
uv sync
```

## Конфигурация

Приложение теперь читает настройки напрямую из файла `/.env`.

Для быстрого старта:

```bash
cp .env.example .env
```

После изменения `/.env` backend, frontend и скрипт индексации нужно перезапустить.

Для быстрых экспериментов можно переключать профиль RAG одной переменной:

```bash
export RAG_PROFILE="balanced"  # balanced | fast | deep
```

Модельные backend-ы теперь тоже конфигурируются отдельно:

```bash
export RAG_EMBEDDING_PROVIDER="huggingface"
export RAG_GENERATION_PROVIDER="huggingface"
```

## Переменные окружения

### PostgreSQL

```bash
export RAG_DATABASE_URL="postgresql://postgres:postgres@127.0.0.1:5432/rag_system"
```

### Bootstrap первого администратора

Если база пустая, backend создаст первого админа автоматически:

```bash
export RAG_BOOTSTRAP_ADMIN_USERNAME="admin"
export RAG_BOOTSTRAP_ADMIN_PASSWORD="admin12345"
export RAG_BOOTSTRAP_ADMIN_FIRST_NAME="System"
export RAG_BOOTSTRAP_ADMIN_LAST_NAME="Admin"
export RAG_BOOTSTRAP_ADMIN_EMAIL="admin@example.com"
```

## Запуск

1. Поднять Postgres и создать базу `rag_system`
2. Поднять Weaviate:

```bash
docker compose -f docker-compose.weaviate.yml up -d
```

3. Задать переменные окружения
4. Подготовить CSV с текстом и загрузить чанки в Weaviate:

```bash
python scripts/preparing_uploading.py
python scripts/download_index.py --recreate-collection
```

Если нужно читать не из SMB, а из локальной директории:

```bash
python scripts/preparing_uploading.py --source-type local --root-path /path/to/documents
```

5. Запустить backend:

```bash
uv run fastapi run backend/app.py --port 8006
```

6. Запустить frontend:

```bash
uv run streamlit run frontend/app.py
```

## Preview одного файла

Чтобы посмотреть, как конкретный локальный файл парсится и режется на чанки тем же pipeline:

```bash
python scripts/preview_document_chunks.py /path/to/file.pdf
```

Скрипт сохранит рядом с исходным файлом:

- `<имя>.extracted.txt` - извлеченный текст
- `<имя>.chunks.json` - чанки с `page_start`, `page_end` и `source_locator`

## Как войти

Если база была пустая и заданы bootstrap-переменные:

- логин: значение `RAG_BOOTSTRAP_ADMIN_USERNAME`
- пароль: значение `RAG_BOOTSTRAP_ADMIN_PASSWORD`

После входа под админом в интерфейсе Streamlit можно:

- посмотреть пользователей из БД
- создать новых пользователей

## Полезные API

- `POST /auth/login`
- `POST /auth/logout`
- `GET /auth/me`
- `GET /auth/users`
- `POST /auth/users`
- `GET /health`
- `POST /chat`

## Важно

- backend ожидает доступный Weaviate с открытыми портами `8080` и `50051`
- если коллекция пуста, сначала нужно выполнить `python scripts/download_index.py`
- если Postgres недоступен, backend не поднимется, потому что на старте создается схема БД
