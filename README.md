# MNIST



## 1. Training

Install dependencies ``` pip install -r train/requirements.txt ```
train/train.py contains the training code. The code can be simply run by running the command ``` mlflow ui ``` in the terminal and ``` python train/train.py ``` in another terminal.
You can track your experiment, by opening up the URL the ``` mlflow ui ``` command returns.

After training is completed, the trained model will be downloaded locally and in mlflow ui also. It is then registered in the mlflow ui. And then can be later used during deployment.

## 2. For deployment using mlflow

RUN the following command
``` mlflow models serve -m "models:/<model_name>/<model_version>" -h <host> -p <port> ```

It should give you an endpoint from which you can use it for inferencing. 

## 2. Inference

RUN ``` pip install -r inference/requirements.txt ```
For running inference locally, configure the infer.py file by passing the path of the image, and then running ``` python infer.py ```
For running inference as an endpoint RUN ``` uvicorn inference.app:app --relaod ``` this will start the uvicorn server. Use the endpoint it returns in POSTMAN and make a POST request by passing an image that you want to inference in the body section (make sure the "key" is set to file type). 

## 3. Dockerization

To run the project as a docker container.
Build the docker image as ``` docker build . mnist ```
After the build is done, run the docker image as ``` docker run -p 8000:8000 mnist ```

## 4. Kubernetes for Deployment

To deploy it to Kubernetes run the following command ``` kubectl apply -f .deply/dev/mnist-deployment.yaml ```




