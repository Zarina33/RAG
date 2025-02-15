-- Включаем расширение vector
CREATE EXTENSION IF NOT EXISTS vector;

-- Создание таблицы для файлов
CREATE TABLE files (
    id SERIAL PRIMARY KEY,
    filename VARCHAR(255) NOT NULL,
    file_type VARCHAR(50) NOT NULL,
    upload_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Создание таблицы для содержимого файлов
CREATE TABLE file_chunks (
    id SERIAL PRIMARY KEY,
    file_id INTEGER REFERENCES files(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    chunk_order INTEGER NOT NULL,
    embeddings vector(384),
    UNIQUE (file_id, chunk_order)
);

-- Создание таблицы для истории запросов
CREATE TABLE queries (
    id SERIAL PRIMARY KEY,
    file_id INTEGER REFERENCES files(id),
    query_text TEXT NOT NULL,
    response TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Создание таблицы для связи запросов и использованных чанков
CREATE TABLE query_chunks (
    query_id INTEGER REFERENCES queries(id),
    chunk_id INTEGER REFERENCES file_chunks(id),
    similarity FLOAT,
    PRIMARY KEY (query_id, chunk_id)
);

-- Индексы
CREATE INDEX idx_file_chunks_file_id ON file_chunks(file_id);
CREATE INDEX idx_queries_file_id ON queries(file_id);