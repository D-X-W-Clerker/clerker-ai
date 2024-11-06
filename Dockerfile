FROM amazon/aws-lambda-python:3.8

RUN /var/lang/bin/python3.8 -m pip install --upgrade pip
RUN apt-get install git -y
RUN git clone -b server https://github.com/D-X-W-Clerker/clerker-ai.git
RUN pip install -r lambda-container-example/requirements.txt

RUN cp lambda-container-example/lambda_function.py /var/task/
# RUN cp lambda-container-example/imagenet_class_index.json /var/task/

# CMD ["lambda_function.lambda_handler"]