FROM amazon/aws-lambda-python:3.10
RUN /var/lang/bin/python3.10 -m pip install --upgrade pip
RUN yum install git -y
RUN yum install -y gcc gcc-c++ cmake make
RUN /var/lang/bin/python3.10 -m pip install llama-cpp-python
RUN /var/lang/bin/python3.10 -m pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu
RUN git clone -b server https://github.com/D-X-W-Clerker/clerker-ai.git
RUN /var/lang/bin/python3.10 -m pip install -r clerker-ai/requirements.txt
#COPY lambda_function.py /var/task/
CMD ["lambda_function.handler"]