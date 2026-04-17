FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV ONNX_MODEL_DIR=/app/artifacts/models
ENV ONNX_MODEL_FILENAME=model.onnx
ENV ONNX_MODEL_AUTO_DOWNLOAD=true
ENV ONNX_MODEL_DRIVE_URL=https://drive.google.com/file/d/1l594HdeuFiWKug3Dqew-GBUzq6FH9inH/view?usp=sharing

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY streamlit_app.py ./
COPY artifacts/preprocessing ./artifacts/preprocessing

RUN python -m app.bootstrap

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD ["python", "-m", "app.healthcheck", "--quiet"]

CMD ["sh", "-c", "python -m app.bootstrap && python -m app.healthcheck --quiet && streamlit run streamlit_app.py --server.address=0.0.0.0 --server.port=8501"]
