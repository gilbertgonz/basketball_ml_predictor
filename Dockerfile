FROM ubuntu:jammy as build

RUN apt update && apt install -y \ 
    python3-pip libgl1 libglib2.0-0 x11-apps \
    && pip install ultralytics

COPY assets/ /app/assets
COPY utils.py /app/utils.py
COPY run.py /app/run.py
COPY best.pt /app/best.pt

# Extract videos
RUN cd /app/assets && tar -xzf video.tar.gz

### Final stage build
FROM scratch

COPY --from=build / /

WORKDIR /app

CMD ["./run.py"]