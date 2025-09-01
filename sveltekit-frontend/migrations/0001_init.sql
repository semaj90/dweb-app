CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS users (
  id SERIAL PRIMARY KEY,
  email TEXT NOT NULL UNIQUE,
  password_hash TEXT NOT NULL,
  created_at TIMESTAMP DEFAULT now()
);

CREATE TABLE IF NOT EXISTS items (
  id TEXT PRIMARY KEY,
  title TEXT,
  content TEXT,
  embedding VECTOR(1536)
);

CREATE TABLE IF NOT EXISTS documents (
  id TEXT PRIMARY KEY,
  filename TEXT NOT NULL,
  content TEXT,
  summary TEXT,
  embedding VECTOR(1536),
  created_at TIMESTAMP DEFAULT now(),
  user_id INTEGER REFERENCES users(id)
);