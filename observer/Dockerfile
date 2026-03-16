FROM python:3.12-slim
WORKDIR /app
ENV TZ=Asia/Shanghai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
COPY observer/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY observer/ ./observer/
EXPOSE 8501
CMD ["streamlit", "run", "observer/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
