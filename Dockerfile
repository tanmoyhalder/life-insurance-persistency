# FROM python:3.8.8-slim
# WORKDIR /app
# COPY Pipfile Pipfile.lock ./
# RUN python -m pip install --upgrade pip
# RUN pip install pipenv && pipenv install --dev --system --deploy
# EXPOSE 8501
# COPY webservice /app
# ENTRYPOINT ["streamlit", "run"]
# CMD ["webservice/app.py"]

FROM python:3.8.8-slim
WORKDIR /app
RUN pip install --upgrade pip
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY webservice /app
EXPOSE 8501
ENTRYPOINT ["streamlit", "run"]
CMD ["app.py"]