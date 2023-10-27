FROM python:3.9

RUN apt update -y && apt install -y libopenblas-dev liblapack3 python3-venv

WORKDIR /app

COPY . .
RUN pip install --upgrade pip poetry
RUN poetry export --without-hashes --format=requirements.txt -o requirements.txt
RUN python3 -m venv env
RUN ./env/bin/pip3 install -r requirements.txt


EXPOSE 12300
# Run your app
COPY . /app
CMD [ "./env/bin/python3", "src/main.py" ]
