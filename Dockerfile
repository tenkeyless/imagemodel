FROM tensorflow/tensorflow:2.4.0-gpu

# ensure local python is preferred over distribution python
ENV PATH /usr/local/bin:$PATH

# http://bugs.python.org/issue19846
# > At the moment, setting "LANG=C" on a Linux system *fundamentally breaks Python 3*, and that's not OK.
ENV LANG C.UTF-8

# Install python tools and dev packages
RUN apt-get update \
    && apt-get install -q -y --no-install-recommends python3.7-dev python3-pip python3-setuptools python3-wheel gcc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
RUN update-alternatives --install /usr/bin/python3 python /usr/bin/python3.7 1
RUN easy_install pip
RUN pip install --upgrade pip

# Base settings
RUN pip install tensorflow-gpu==2.4.0 Keras==2.4.0
RUN apt-get update
RUN apt-get install 'ffmpeg'\
    'libsm6'\ 
    'libxext6' -y
RUN pip install opencv-python
RUN pip install image-keras paramiko Pillow

RUN pip install tensorflow_datasets
RUN pip install tensorflow-metadata
RUN apt-get -y install git
RUN pip install -q git+https://github.com/tensorflow/examples.git

# Google Storage
RUN pip install google-cloud-storage
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
RUN apt-get install apt-transport-https ca-certificates gnupg curl -y
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -
RUN apt-get update && apt-get install google-cloud-sdk -y
RUN pip install cloud-tpu-client

RUN apt -y install graphviz
RUN pip install pydot
RUN pip install tf_clahe
