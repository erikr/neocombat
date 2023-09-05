# Setup

This repo contains the code for running the demo described at https://dev.to/irubtsov/object-tracking-and-video-cropping-with-computer-vision-and-machine-learning-3ge2.

1. Install miniconda:

    ```bash
    mkdir -p ~/miniconda3
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
    rm -rf ~/miniconda3/miniconda.sh
    ~/miniconda3/bin/conda init bash
    ```
    
    See https://docs.conda.io/projects/miniconda/en/latest/ for more information.

1. Create environment:

    ```bash
    cd neocombat
    conda env create -f environment.yml
    ```

1. Install other depencies, assuming you are using Ubuntu Linux:

    ```bash
    sudo apt-get update && sudo apt-get install ffmpeg libsm6 libxext6  -y
    ```
