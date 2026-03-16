FROM python:3.10-slim
WORKDIR /app
ENV TZ=Asia/Shanghai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ openssh-client && rm -rf /var/lib/apt/lists/*

COPY trainer/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .
COPY converter/ ./converter/
COPY strategy/ ./strategy/
COPY trainer/ ./trainer/

CMD ["python", "-m", "trainer.weekly_train"]
