# Optimizer Edge

This software runs on edge device and listen for models to be evaluated.
The models are the one sent by the [optimizer-main](https://gitlab.com/edgeseamless/optimizer-main) that will be evaluated by this software, the evaluation results are sent back to optimizer-main.

## Build & Run
To build the image:
```
docker build -t optimizer-edge .
```
The software expose the port 12300 for the incoming connection, to boot the software, binding the port of the container 12300 to the host port 9999, you can run:
```
 docker run -dp 127.0.0.1:9999:12300 optimizer-edge
```