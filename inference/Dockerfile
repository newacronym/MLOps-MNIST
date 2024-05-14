FROM python:3.11

WORKDIR /workspace

RUN python -m pip install --upgrade pip
RUN apt update && apt upgrade -y

COPY ./ /workspace/

RUN pip install -r inference/requirements.txt

EXPOSE 8000

CMD ["uvicorn", "inference.app:app", "--host", "0:0:0:0", "--port", "8000"]
