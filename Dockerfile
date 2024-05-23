
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

## Add System Dependencies
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
        build-essential \
        git \
        wget \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get autoremove -y \
    && apt-get clean

## Change working directory
WORKDIR /app/alphafold

## Clone and install the LucidRains package + requirements
RUN git clone https://github.com/lucidrains/alphafold3-pytorch . \
    && git checkout main \
    && python -m pip install .
