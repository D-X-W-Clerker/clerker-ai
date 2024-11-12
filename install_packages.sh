#!/bin/bash

# Command to install packages
# chmod +x install_packages.sh
# ./install_packages.sh


# Install Node.js and Mermaid
wget -nv https://d3rnber7ry90et.cloudfront.net/linux-x86_64/node-v18.17.1.tar.gz

sudo mkdir /usr/local/lib/node
tar -xf node-v18.17.1.tar.gz
sudo mv node-v18.17.1 /usr/local/lib/node/nodejs
echo "export NVM_DIR=''" >> /home/ec2-user/.bashrc
echo "export NODEJS_HOME=/usr/local/lib/node/nodejs" >> /home/ec2-user/.bashrc
echo "export PATH=\$NODEJS_HOME/bin:\$PATH" >> /home/ec2-user/.bashrc
. /home/ec2-user/.bashrc
node -e "console.log('Running Node.js ' + process.version)"


# Python 패키지 업그레이드 및 설치
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
python3 -m pip uninstall numpy -y
python3 -m pip install "numpy<2"
python3 -m pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu


# 필수 패키지 설치 (yum 사용)
sudo yum install -y git gcc gcc-c++ cmake make tar xz
CMAKE_ARGS=-DLLAMA_CUBLAS=on FORCE_CMAKE=1 pip install --force-reinstall --no-cache-dir --upgrade llama-cpp-python==0.2.23

# ffmpeg 설치
curl -L -o ffmpeg-release-amd64-static.tar.xz https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz
sudo tar -xJf ffmpeg-release-amd64-static.tar.xz --strip-components=1 -C /usr/local/bin
rm ffmpeg-release-amd64-static.tar.xz
chmod +x /usr/local/bin/ffmpeg

npm install -g @mermaid-js/mermaid-cli@8.9.2
sudo yum install -y fontconfig
sudo yum install -y google-noto-sans-cjk-fonts
