FROM unsloth/unsloth:latest-cuda-12.1

RUN apt-get update && apt-get install -y libpoppler-cpp-dev


WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir  
COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
