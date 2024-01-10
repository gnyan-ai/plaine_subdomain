import threading
from langchain.chains import LLMChain
import os
import atexit
import re
import uuid
import functools
import itertools
import ast
import json
import io
import sys
import glob
import pandas as pd
import numpy as np
from datetime import datetime

import os
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import (
    ConversationalRetrievalChain,
)
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import SystemMessage
from langchain.prompts import PromptTemplate
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.docstore.document import Document
from langchain.memory import ChatMessageHistory, ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda


def get_llm(use_openai=False):
    ollama_endpoint = "http://192.168.27.10:11434"
    from langchain.chains import LLMChain
    from langchain.prompts import PromptTemplate

    # Create and load LLM
    def return_ollama_llm(
        base_url=ollama_endpoint, repeat_penalty=1.25, temperature=0.01
    ):
        from langchain.callbacks.manager import CallbackManager
        from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
        from langchain.llms import Ollama

        if not hasattr(return_ollama_llm, "ollama"):
            return_ollama_llm.ollama = Ollama(
                model="neural-chat:latest",
                base_url=ollama_endpoint,
                repeat_penalty=repeat_penalty,
                temperature=temperature,
                repeat_last_n=-1,
                callback_manager=CallbackManager(
                    [StreamingStdOutCallbackHandler()]),
                stop=["```"],
                timeout=30,
                num_ctx=4096,
            )
        return return_ollama_llm.ollama

    return return_ollama_llm()


llm = get_llm()

file_string = """You are an EXPERT developer who will design, structure, and write great software application code.
Given a brief description of a story, you will do two things.
First, you will -- if provided -- choose the programming language requested in the question; or determine a programming language that best fits the question: BUT YOU WILL NOT MIX PROGRAMMING LANGUAGES.
Second, you will create a project file structure of the necessary files and folders.
For example, a python hello world program will require main.py, requirements.txt, README.md files: your response will therefore be --
```files
[("main.py", "", "file"), ("requirements.txt", "", "file"), ("docs", "", "dir"), ("README.md", "docs", "file")]
```
OR
```files
[("main.pl", "", "file"), ("docs", "", "dir"), ("README.md", "docs", "file")]
```
Take a deep breath and reason step-by-step. Assume Linux filesystem with / file separator.

You will present a SINGLE manifest of files: You are NOT allowed to split subfolder listings in your response.
For example, ```
Inside the 'build/src' directory: [("main.py", "", "file"), ("utils.py", "", "file")]``` is NOT ALLOWED.
Instead, ```files [("build", "", "dir"), ("src", "build", "dir"), ("main.py", "build/src", "file"), ("utils.py", "build/src", "file")]``` is ALLOWED.
Each tuple in the response will be a triple of (filename, directory path, type). Filename MUST be a filename ONLY, not a path. directory path MUST be a path ONLY, not a filename.

You are NOT allowed to offer multiple options like sbt or gradlew or maven archetypes. Choose ONE best option.
You want to be very diligent in the directory structure. Your response MUST be in the form of a single list of tuples. Each tuple is a (file|directory name, its base directory path, and its type {{file|dir}}). 
Make the directory path relative to the top level directory. You ARE NOT ALLOWED to create directories outside of the top level directory.
ALWAYS quote filenames and directories that have whitespaces. Surround the response in ``` backticks at the start and end
Description: '''{question}'''
Response: 
```files
"""
file_prompt = PromptTemplate.from_template(file_string)

content_string = """You are an EXPERT cloud and devops engineer who will shrinkwrap applications and services into containerized pods to deploy to the cloud.
Given a description and the structural file layout of the project, you will generate ONE (and ONLY ONE) configuration file at a time (from the following list) -- 

1) you will create a Dockerfile to package the application or service
2) you will create a docker-compose.yml configuration file
3) you will create a Kubernetes deployment configuration file
4) you will create a Kubernetes service definition configuration file
5) you will create a kubernetes ingress definition configuration file
6) you will create a kubernetes configmap and secrets configuration file
7) you will create a kubernetes persistent volume and persistent volume claim configuration file
8) you will create a helm chart to deploy the application or service
9) you will create a terraform file to deploy the application or service

YOU WILL ONLY GENERATE yaml, configuration, and terraform FILE. YOU WILL NOT GENERATE ANY OTHER FILE TYPES.
YOU CAN ONLY GENERATE ONE FILE AT A TIME. YOU ARE NOT ALLOWED TO GENERATE MULTIPLE FILES AT A TIME.
You will be asked to generate a file at a particular step. You will be asked to generate the files in the order of the steps. Do not skip steps; limit yourself to the requested file at a particular step.
YOU MUST NOT INCLUDE ANY keys, tokens, or license numbers in the configuration files. Instead, you will use placeholders like {{KEY}} or {{TOKEN}} or {{LICENSE}}.
YOU CAN ASSUME that the requested file at a particular step presumes you have already generated the files from the previous steps.
Assume Linux host with / file separator. 
Assume google cloud platform as the cloud provider.
Take a deep breath and reason step-by-step. Please DO NOT be repetitive in your response: be EXTREMELY succint and brief.
You will respond with the contents of the file, nothing extra. Surround response in backticks ```.
YOU ARE NOT ALLOWED to produce any extraneous text besides the contents of the requested file: ABSOLUTELY NO EXCEPTIONS.
YOU ARE NOT ALLOWED to produce syntax errors. 
YOU ARE NOT ALLOWED to produce any binary or non-text content. YOUR RESPONSE MUST BE PLAIN TEXT and IT CANNOT EXCEED 10240 CHARACTERS.
Present the contents of the file in the form of a string, preferably in a single line.
Description: '''{question}'''
List of file layout in context: '''{file_layout}'''
Code for `{file_type} configuration`:```{file_name}
"""
content_prompt = PromptTemplate.from_template(content_string)

file_llm_chain = LLMChain(
    llm=llm,
    prompt=file_prompt,
    verbose=True,
)

content_llm_chain = LLMChain(
    llm=llm,
    prompt=content_prompt,
    verbose=True,
)


def generate_repeatable_id(text):
    import hashlib
    return f"plaineartifacts/{hashlib.md5(text.encode('utf-8')).hexdigest().upper()[:8]}"


def normalize_path(path):
    # Replace multiple backslashes or forward slashes with a single os.sep
    normalized_path = re.sub(r'[/\\]+', "/", os.path.normpath(path))
    return os.path.normpath(normalized_path.lstrip("/"))

def path_split_to_dir_file(root_dir, directory, filename):
    head, tail = os.path.split(normalize_path(os.path.join(directory, normalize_path(filename.lstrip(directory)))))
    return os.path.split(os.path.join(root_dir, head, tail))

def create_files(response):
    root_dir = normalize_path(generate_repeatable_id(response['question']))
    # Scan for text that looks like a list of tuples
    text = response['text']
    import ast
    tuples = re.findall(r'\(.*?\)', text)

    def safe_parse(s):
        try:
            return ast.literal_eval(s)
        except:
            return None

    tuples = [y for y in [safe_parse(f"({x})") for x in tuples if x] if y]
    list_of_tuples = [item for item in tuples if len(item) == 3]
    # Create a pandas frame with the list of tuples
    df = pd.DataFrame(list_of_tuples, columns=[
                      'filename', 'directory', 'type'])
    df['directory'] = df['directory'].fillna('').apply(lambda x: str(x).lstrip('.')).apply(normalize_path)
    df['filename'] = df['filename'].fillna('').apply(lambda x: str(x).lstrip('.')).apply(normalize_path)
    # Create files
    print(df)
    file_layout = []
    for row in df[df['type'] == 'file'].itertuples():
        head, tail = os.path.split(row.filename)
        dir_name, file_name = path_split_to_dir_file(root_dir, row.directory, row.filename)
        file_layout.append(os.path.join(dir_name, file_name))
    response.update({'file_layout': file_layout})
    return response


def create_config(response):
    root_dir = generate_repeatable_id(response['question'])
    import markdown
    import re
    subs = {
        '`': r'\`',
        '*': r'\*',
        '#': r'\#',
        '[': r'\[',
        ']': r'\]',
        '(': r'\(',
        ')': r'\)',
        '_': r'\_',
        '{': r'\{\{',
        '}': r'\}\}',
        '\n': '<br>',
    }

    pattern = re.compile('|'.join([re.escape(k) for k in subs.keys()]))
    def escaped_md(md): return pattern.sub(lambda x: subs[x.group(0)], md)
    file_layout = response['file_layout']
    file_keys = {1:'Dockerfile', 2:'docker-compose.yml', 3:'kubernetes-deployment.yml', 4:'kubernetes-service.yml', 5:'kubernetes-ingress.yml', 6:'kubernetes-configmap.yml', 7:'kubernetes-pvc.yml', 8:'helm-chart.yml', 9:'main.tf'}
    for tgt_file, file_name in file_keys.items():
        file_type = str(tgt_file)
        content = ''
        try:
            def run_with_timeout(timeout):
                result = ['']
                def target():
                    result[0] = content_llm_chain.invoke({'question': response['question'], 'file_layout': str(file_layout), 'file_type': file_type, 'file_name': file_name})
                thread = threading.Thread(target=target)
                thread.start()
                thread.join(30)
                if thread.is_alive():
                    # Thread is still running, we consider this a timeout
                    print("Function call to LLMChain timed out")
                    return ''
                else:
                    return result[0]
            code_response = run_with_timeout(30)
            content = code_response['text'].strip() if code_response else ''
            with open(os.path.join(root_dir, f"{file_name}"), 'w') as f:
                f.write(content)
        except Exception as e:
            print(f"Exception creating file {file_type}: {e}")

    with open(os.path.join(root_dir, '.success'), 'w') as f:
        f.write('')
    return response


full_chain = file_llm_chain | create_files | create_config
preseeded_questions_list = {
    'Rental Car Heatmap':'Create a streamlit application that shows a geo folium heatmap of a rental car companies in the US that shows rentals, reservations, and returns in the US broken down by the state. The state and date filters (start and end time ranges) are to be presented in the sidebar.',
    'Video recoloring':'Develop a streamlit video coloring application using GFPGAN and OpenCV. The application should browse for a grayscale video, process, and emit a colored video file. The application should also have a sidebar to browse for the input grascale videe. It must use main streamlit window to show a preview of the colored video (every 30th colored frame).',
    'REST Service':'Create a Springboot echo REST service that echoes the request body as a response. The service should be able to handle a POST request with a JSON body. The service should be able to handle a GET request with a query parameter. The service should be able to handle a GET request with a path parameter.',
    'Salesforce Integration':'Build a databricks pyspark script to ingest SFDC Accounts into a delta table. The script should be able to handle incremental loads and full loads by maintaining a last updated waterline.',
    'dbt Star schema':'Author a dbt pipeline for a star schema that has a fact table and 4 dimension tables. The fact table is a table of sales transactions. The dimension tables are: product, customer, date, and store. The fact table has the following columns: product_id, customer_id, date_id, store_id, and sales_amount. The product dimension table has the following columns: product_id, product_name, product_category, and product_subcategory. The customer dimension table has the following columns: customer_id, customer_name, customer_city, and customer_state. The date dimension table has the following columns: date_id, date, day_of_week, month, and year. The store dimension table has the following columns: store_id, store_name, store_city, and store_state.',
    'Sales Trends':'Given a dataset of sales transactions across various regions and products, author Spark SQL to compute weekly average sales across N weeks and generate percentage differential across each week.'
}

def invoke_chain(question):
    return full_chain.invoke({'question': question})
