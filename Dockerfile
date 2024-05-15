FROM python:3.11-slim

RUN pip install torch torchvision

WORKDIR /workspace

COPY ./inference /workspace/inference

RUN pip install -r inference/requirements.txt

EXPOSE 8000

CMD ["uvicorn", "inference.app:app", "--host", "0:0:0:0", "--port", "8000"]
