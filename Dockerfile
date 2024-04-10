FROM ubuntu:jammy as build

RUN apt update && apt install -y \ 
    python3-pip libgl1 libglib2.0-0 x11-apps \
    && pip install ultralytics

COPY test_assets/ /app/test_assets
COPY kalman_filter.py /app/kalman_filter.py
COPY predict.py /app/predict.py
COPY best.pt /app/best.pt

### Final stage build
FROM scratch

COPY --from=build / /

WORKDIR /app

CMD ["./predict.py"]