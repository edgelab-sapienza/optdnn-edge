FROM python:3.9

RUN apt update -y && apt install -y libopenblas-dev liblapack3

WORKDIR /app

COPY . .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 12300
# Run your app
COPY . /app
CMD [ "python3", "src/main.py" ]