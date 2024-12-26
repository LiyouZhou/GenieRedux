FROM continuumio/miniconda3

COPY genie_redux_env.yaml /tmp/genie_redux_env.yaml

RUN conda env create -f /tmp/genie_redux_env.yaml && \
    echo "source activate genie_redux" >> ~/.bashrc

RUN apt-get update && apt-get install -y \
    libgl1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*