# RAGCustomizer

## Status Badge

[![CI/CD Pipeline](https://github.com/yiboliu/RAGCustomizer/actions/workflows/ci_cd.yml/badge.svg)](https://github.com/yiboliu/RAGCustomizer/actions/workflows/ci_cd.yml)

## Architecture Diagram
![alt text](https://github.com/yiboliu/RAGCustomizer/blob/main/architecture_diagram.png?raw=true)

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
### Pull docker image from registry

### Download the code and build the app
## Performance/Evaluation Results

