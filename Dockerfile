FROM nvidia/cuda:12.1.0-devel-ubuntu20.04
ENV TZ=America/New_York
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda-12.1
ENV PATH=$PATH:$CUDA_HOME/bin
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64

WORKDIR /opt/ml/model

COPY Chunking/ ./Chunking/
COPY Diagrams/ ./Diagrams/
COPY Diagrams/mermaid/ ./Diagrams/mermaid/
COPY Keywords/ ./Keywords/
COPY STT/ ./STT/
COPY inference.py ./
COPY requirements.txt ./requirements.txt
COPY run.py ./

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && \
    apt-get install -y libxkbcommon0 libatk1.0-0 libcups2 libxcomposite1 libxrandr2 libxdamage1 libxext6 libxfixes3 libxrender1 libasound2 libpangocairo-1.0-0 libpangoft2-1.0-0 fonts-noto-cjk libgbm1  libatk-bridge2.0-0  libxshmfence1 libnss3 libnspr4 libgconf-2-4 libdrm2 python3  python3-distutils  python3-pip wget curl sudo git gcc g++ cmake make tar xz-utils libsndfile1-dev chromium-browser build-essential && \
    rm -rf /var/lib/apt/lists/*


RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - && \
    apt-get install -y nodejs


RUN npm install -g @mermaid-js/mermaid-cli puppeteer@latest


RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install -r ./requirements.txt && \
    python3 -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121


RUN curl -L -o ffmpeg-release-amd64-static.tar.xz https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz && \
    tar -xJf ffmpeg-release-amd64-static.tar.xz —strip-components=1 -C /usr/local/bin && \
    rm ffmpeg-release-amd64-static.tar.xz && \
    chmod +x /usr/local/bin/ffmpeg

ENTRYPOINT ["python3", "run.py"]