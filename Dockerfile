FROM python:3.9

RUN apt update -y && apt install -y libopenblas-dev liblapack3

WORKDIR /app

COPY . .
RUN pip install --upgrade pip poetry
RUN poetry install
EXPOSE 12300
# Run your app
COPY . /app
CMD [ "poetry", "run", "python", "src/main.py" ]
