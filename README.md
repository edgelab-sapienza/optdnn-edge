# OptDNN: Benchmarker for edge devices

This software runs on edge device and listen for models to be evaluated.
The models are the one sent by the [optimizer](https://gitlab.com/kernelmachine/optdnn) that will be evaluated by this software, the evaluation results are sent back to optimizer-main.

## Build & Run (With docker)
To build the image:
```
docker build -t optimizer-edge .
```
The software expose the port 12300 for the incoming connection, to boot the software, binding the port of the container 12300 to the host port 9999, you can run:
```
 docker run -dp 127.0.0.1:9999:12300 optimizer-edge
```

## Build & Run (Without docker)
The used dependency manager is [Poetry](https://python-poetry.org/), which can be installed with pip using this command: 

    pip install poetry

Now, from the project directory run:

    poetry install

Then to run the code enter:

    poetry shell
    python src/main.py

By default, the port used is 12300, but you can change this parameter with the `--port` argument, for example to listen on port 1234:

    python src/main.py --port 1234
