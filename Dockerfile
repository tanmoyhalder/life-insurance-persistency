FROM python:3.8.8-slim
WORKDIR /app
RUN pip install --upgrade pip
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY webservice /app
EXPOSE 8501
ENTRYPOINT ["streamlit", "run"]
CMD ["Welcome.py"]