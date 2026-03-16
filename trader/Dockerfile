FROM python:3.12-slim
WORKDIR /app
ENV TZ=Asia/Shanghai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
COPY trader/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY trader/ ./trader/
COPY converter/ ./converter/
CMD ["python", "-m", "trader.trade_daily"]
