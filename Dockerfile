FROM python:3.13-slim

WORKDIR /app

# 先装依赖（利用层缓存：改业务代码无需重装依赖）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY data ./data
COPY seed_corpus.py .

EXPOSE 8000

# 生产应使用 Alembic 迁移；此处演示用 init_db 兜底
CMD ["sh", "-c", "python seed_corpus.py && uvicorn app.main:app --host 0.0.0.0 --port 8000"]
