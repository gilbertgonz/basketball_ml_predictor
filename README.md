# Basketball ML tracking and trajectory estimation

Install [docker](https://docs.docker.com/engine/install/)

To build:
```
$ docker build -t sports_ml_tracking .
```

To run:
```
$ xhost +local:docker
$ docker run --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix basketball_ml_tracking
