# Basketball ML tracking and trajectory estimation

This project aims to predict successful shots from basketball videos. It uses YOLOv8 to detect the ball/rim and kalman filtering + polynomial regression to track the ball and predict its trajectory towards the rim. The probability of a successful shot is computed using a combined probability of both predictions. 

![](assets/results.gif)

## To run:
1. Install [docker](https://docs.docker.com/engine/install/)

2. Build:
```
$ docker build -t basketball_ml_tracking .
```

3. Run:
```
$ xhost +local:docker
$ docker run --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix basketball_ml_tracking

# To show uncertainty from Kalman filter prediction, run:
$ docker run --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix basketball_ml_tracking ./run.py --show-uncertainty
```

### Sources:
https://drive.google.com/file/d/1CNRmlmaoT-_PZBlRO9ZpJfQuvus8Oknk/view

https://campar.in.tum.de/Chair/KalmanFilter