# RAG Ассистент

RAG-система для поиска по внутренней документации компании. В проекте есть:

- FastAPI backend
- Streamlit frontend
- Weaviate для хранения чанков, эмбеддингов и метаданных
- PostgreSQL для хранения пользователей и сессий

## Что изменилось в авторизации

Логины и пароли больше не берутся из `frontend/config.yaml`.

Теперь схема такая:

- пользователи хранятся в таблице `users` в Postgres
- сессии хранятся в таблице `auth_sessions`
- вход выполняется через `POST /auth/login`
- список пользователей доступен через `GET /auth/users`
- создание пользователей доступно через `POST /auth/users`

При первом запуске backend может автоматически создать bootstrap-админа, если заданы переменные окружения.

## Структура проекта

- `backend/app.py` - FastAPI-приложение и инициализация схемы БД
- `backend/db.py` - работа с Postgres, пользователями и сессиями
- `backend/endpoints/auth.py` - login/logout/users/me
- `backend/endpoints/chat.py` - retrieval и генерация ответа
- `backend/schemas/schema_auth.py` - схемы авторизации
- `frontend/app.py` - Streamlit UI с логином через backend
- `source_readers.py` - слой чтения документов из SMB или локальной директории
- `document_processing.py` - слой парсинга и чанкинга документов
- `embed_uploader.py` - слой расчета эмбеддингов и загрузки чанков в Weaviate
- `scripts/preparing_uploading.py` - чтение документов из источника и сохранение извлеченного текста в CSV
- `scripts/download_index.py` - разбиение на чанки, расчет эмбеддингов и загрузка в Weaviate
- `scripts/preview_document_chunks.py` - локальный preview парсинга и чанкинга одного файла
- `scripts/create_hash.py` - генерация bcrypt-хеша
- `docker-compose.weaviate.yml` - локальный запуск Weaviate

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

### Дополнительные настройки

```bash
export RAG_BACKEND_URL="http://127.0.0.1:8006"
export RAG_CHAT_REQUEST_TIMEOUT_SECONDS="300"
export RAG_SESSION_TTL_DAYS="7"
export RAG_EMBEDDING_PROVIDER="huggingface"
export RAG_GENERATION_PROVIDER="huggingface"
export RAG_EMBEDDING_MODEL="BAAI/bge-m3"
export RAG_GENERATION_MODEL="Qwen/Qwen2.5-7B-Instruct"
export RAG_GENERATION_DTYPE="auto"
export RAG_GENERATION_DEVICE_MAP="auto"
export RAG_EVAL_DATASET_PATH=""
export RAG_SOURCE_CSV_PATH="scripts/documents_data.csv"
export RAG_WEAVIATE_HTTP_HOST="127.0.0.1"
export RAG_WEAVIATE_HTTP_PORT="8080"
export RAG_WEAVIATE_HTTP_SECURE="false"
export RAG_WEAVIATE_GRPC_HOST="127.0.0.1"
export RAG_WEAVIATE_GRPC_PORT="50051"
export RAG_WEAVIATE_GRPC_SECURE="false"
export RAG_WEAVIATE_API_KEY=""
export RAG_WEAVIATE_COLLECTION="DocumentChunk"
export RAG_WEAVIATE_SKIP_INIT_CHECKS="false"
export RAG_DENSE_TOP_K="24"
export RAG_BM25_TOP_K="24"
export RAG_FINAL_TOP_K="6"
export RAG_DOC_LOOKUP_FINAL_TOP_K="4"
export RAG_MAX_CONTEXT_TOKENS="1200"
export RAG_MAX_CHUNKS_PER_DOCUMENT="2"
export RAG_DOC_LOOKUP_MAX_CHUNKS_PER_DOCUMENT="3"
export RAG_BM25_TITLE_WEIGHT="8.0"
export RAG_BM25_PATH_WEIGHT="3.0"
export RAG_BM25_TEXT_WEIGHT="1.0"
export RAG_RERANK_DENSE_WEIGHT="0.45"
export RAG_RERANK_BM25_WEIGHT="0.30"
export RAG_RERANK_TITLE_WEIGHT="0.15"
export RAG_RERANK_PATH_WEIGHT="0.05"
export RAG_RERANK_COVERAGE_WEIGHT="0.05"
export RAG_DOC_LOOKUP_EXACT_BOOST="0.25"
export RAG_DOC_LOOKUP_TITLE_BOOST="0.18"
export RAG_DOC_LOOKUP_PATH_BOOST="0.10"
export RAG_GPU_CONTEXT_BUDGETS="1200,800,512,320"
export RAG_GPU_MAX_NEW_TOKENS="250,160,96,64"
export RAG_CPU_CONTEXT_BUDGETS="1200,900"
export RAG_CPU_MAX_NEW_TOKENS="220,128"
export RAG_GENERATION_TEMPERATURE="0.2"
export RAG_GENERATION_TOP_P="0.9"
export RAG_GENERATION_DO_SAMPLE="true"
export RAG_PROFILE="balanced"
export RAG_PRELOAD_MODELS_ON_STARTUP="false"
export RAG_CHUNK_TOKENS="512"
export RAG_CHUNK_OVERLAP="100"
export RAG_EMBEDDING_BATCH_SIZE="8"
export RAG_EMBEDDING_DEVICE="cuda"
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

## Если локально не хватает GPU-памяти

Для видеокарт уровня 10-12 GB VRAM лучше сразу запускать backend с более консервативными лимитами:

```bash
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export RAG_PROFILE="fast"
export RAG_GENERATION_MODEL="Qwen/Qwen2.5-3B-Instruct"
export RAG_MAX_CONTEXT_TOKENS="512"
export RAG_GPU_CONTEXT_BUDGETS="512,384,256"
export RAG_GPU_MAX_NEW_TOKENS="128,96,64"
```

После изменения переменных backend нужно перезапустить.

## Оффлайн evaluation

Можно завести JSON или JSONL датасет с полями:

```json
[
  {
    "query": "Где находится регламент отпусков?",
    "expected_source_contains": ["регламент", "отпуск"],
    "expected_answer_contains": ["отпуск"]
  }
]
```

И запускать оценку так:

```bash
export RAG_EVAL_DATASET_PATH="eval/dataset.json"
.venv/bin/python scripts/run_eval.py
```
