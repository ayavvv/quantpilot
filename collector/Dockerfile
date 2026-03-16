FROM python:3.12-slim
WORKDIR /app
ENV TZ=Asia/Shanghai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
COPY collector/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY collector/ ./collector/
COPY converter/ ./converter/
COPY scripts/ ./scripts/
CMD ["python", "-m", "collector.main"]
