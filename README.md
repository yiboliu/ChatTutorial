# RAGCustomizer

## Status Badge

[![CI/CD Pipeline](https://github.com/yiboliu/RAGCustomizer/actions/workflows/ci_cd.yml/badge.svg)](https://github.com/yiboliu/RAGCustomizer/actions/workflows/ci_cd.yml)

## Architecture Diagram
![alt text](https://github.com/yiboliu/RAGCustomizer/blob/main/images/architecture_diagram.png?raw=true)

## Project Purpose

The idea of this project is to allow users to customize RAG based on their needs. 
In the most ready-to-serve LLM agents, the RAG has a fixed set of texts which serves some specific purpose.

In this project, however, we enable users to customize their own RAG system. People can upload files themselves and build the RAG database themselves.
All files uploaded are visible to all users, so people can choose whatever files they like. This way, we grant the customizability to users to the largest extent.

Since we did not include the Llamafile in the docker image, users can choose whichever model they like. 
The Llamafile is a standalone service, independent of RAGCustomizer. 
Therefore, this grants another aspect of customizability.

## Instruction of Setup
I'll provide 2 approaches to set up:
Before going to any of them, remember to start the llamafile. 

Go to the directory of the llamafile and run the following (I'm using TinyLlma-1.1B-Chat-v1.0.F16.llamafile as an example):
``./TinyLlama-1.1B-Chat-v1.0.F16.llamafile --server --port 8081 --nobrowser`` The port number 8081 is set in the code. Feel free to change it as you like.
### Pull docker image from registry
 - pull the docker image from the registry
![alt text](https://github.com/yiboliu/RAGCustomizer/blob/main/images/docker_pull.png?raw=true)
 - run the docker container
![alt text](https://github.com/yiboliu/RAGCustomizer/blob/main/images/docker_run.png?raw=true)

### Download the code and build the app
- Run the below command to download the code:
``git clone https://github.com/yiboliu/RAGCustomizer.git``
- Start your virtual environment (optional but recommended)
- Run ``pip install -r requirements.txt`` in the root dir of this repo
- Go to docker folder ``cd docker``
- Build the docker image ``docker-compose build``
- Run the docker container ``docker-compose up``

After all the service is up, go to ``localhost:5001`` in your browser and enjoy your RAG customization!
## Performance/Evaluation Results

## Unit tests
All tests are located under `test/` dir, so simply running ``python -m unittest <any test file you like>`` can perform testing

## Model Selection
The reason I choose `TinyLlama-1.1B-Chat-v1.0.F16.llamafile` is simply for demo purpose. You can choose whatever model you like. 
For demo purpose, this model is small in size, performant and has a variety of application scenarios. 
Due to the excellent balance between performance and resource efficiency, I chose it. 