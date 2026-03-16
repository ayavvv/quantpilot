FROM python:3.12-slim
WORKDIR /app
ENV TZ=Asia/Shanghai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
COPY reporter/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY reporter/ ./reporter/
CMD ["python", "-m", "reporter.send_report"]
