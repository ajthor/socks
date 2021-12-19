FROM python:3.9

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
  notebook \
  jupyterlab \
  tqdm

# The following is to enable Binder compatibility using a Dockerfile.
ARG NB_USER=root
ARG NB_UID=0
ENV USER ${NB_USER}
ENV HOME /home

RUN if [[ "$arg" != "root" ]] ; then adduser --disabled-password \
    --gecos "Default user" \
    --uid ${NB_UID} \
    ${NB_USER} ; fi

COPY . ${HOME}

USER root
RUN chown -R ${NB_UID} ${HOME}

WORKDIR ${HOME}
RUN pip install .

USER ${USER}
