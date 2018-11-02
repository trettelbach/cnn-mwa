FROM pytorch/pytorch:latest

COPY /histograms.py .
COPY /labels_train.csv .
COPY /labels_test.csv .
COPY /cnn_pytorch_emc_v2.py .

RUN apt-get update \
     && apt-get install -y \
        libgl1-mesa-glx \
        libx11-xcb1 \
     && apt-get clean all \
     && rm -r /var/lib/apt/lists/*

RUN /opt/conda/bin/conda install \
	astropy \
	matplotlib \
	pandas \ 
        PIL

