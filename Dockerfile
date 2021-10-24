ARG PYTHON_VERSION
FROM python:$PYTHON_VERSION
# FROM python:3.9

RUN apt-get -y update && \
  apt-get install -y --no-install-recommends \
  build-essential \
  git \
  wget \
  ca-certificates \
  libblas-dev \
  liblapack-dev \
  ffmpeg \
  dvipng \
  cm-super \
  texlive-xetex \
  texlive-fonts-recommended \
  texlive-plain-generic && \
  apt-get clean && rm -rf /var/lib/apt/lists/*

COPY . /usr/local/gym_socks/
WORKDIR /usr/local/gym_socks/

RUN pip install -U \
  black \
  gym \
  numpy \
  pyglet \
  pytest \
  sacred \
  scipy \
  scikit-learn \
  matplotlib \
  tqdm

RUN pip install .
