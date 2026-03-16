FROM python:3.10-slim
WORKDIR /app
ENV TZ=Asia/Shanghai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# System dependencies for pyqlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && rm -rf /var/lib/apt/lists/*

COPY inference/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY converter/ ./converter/
COPY strategy/ ./strategy/
COPY inference/ ./inference/

CMD ["python", "-m", "inference.run_daily"]
