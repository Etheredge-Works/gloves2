FROM python:3.8-buster

COPY deploy.requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
RUN rm /tmp/requirements.txt

COPY app.py /app/app.py
COPY model.onnx /app/model.onnx

WORKDIR /app

CMD ["python", "app.py"]