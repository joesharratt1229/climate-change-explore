# Introduction
This is an implementation of a retrieval augmented generated system (RAG) using GCP native technologies and services. Here a chatbot embed reports relating to Lloyds Climate sustainability and embeds them and stores them in GCS. When a user asks a question the relevant information is retrieved from GCS based on vector similarity and then used to answer the question by using the VERTEX AI gemini endpoint
# Prerequisites
To work the following is needed
* A working GCP account
* GCP service account key with read/write access to GCS and vertex ai endpoint axis (Need to enable these endpoints)
* OpenAI api key
* Storage buckets for embedding and storing files
* Put service account key in root of the directory.

For the app to work you need a pdf file in your file storage bucket.

# Installation
1. To install the necessary libraries install the following (Note that there may be other dependencies that need to be installed to install the appropriate packages)
`pip install requirements.txt`

2. Create an .env file with the appropriate information for api key and GCP account information. Follow the `.env.example` template

3. From root directory run the following
`uvicorn main: app --reload`

4. NOTE: The current application does not have a frontend. To check the application is working you can do the following. Open a terminal and type as follows:
```
import requests
requests.get('http://127.0.0.1:8000/embed', json= {'blob':<FILE NAME>).json()
requests.post('http://127.0.0.1:8000/retrieve-index', json={'query':'I am an SME help me be sustainable'}).json()
```
