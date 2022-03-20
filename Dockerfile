# This is id of the already pulled image tensorflow/tensorflow
FROM tensorflow/tensorflow:latest-gpu-jupyter

RUN apt update -y && apt install -y \
libsm6 \
libxext6 \
libxrender-dev \
libgl1-mesa-glx \
python3-opencv \
libgl1 \
libavcodec-dev \
libavformat-dev \
libswscale-dev \
libv4l-dev \
libxvidcore-dev \
libx264-dev \
libgtk-3-dev -y

RUN pip install \
opencv-python \
matplotlib
