# Course Module

## Table of Contents
- [Introduction to LLM and RAG](#lecture-1)
- [Configuring Your Environment](#lecture-2)
- [Retrieval and Search](#lecture-3)
- [Generating Answers with OpenAI GPT 4-o](#lecture-4)

---

<details>
  <summary id="lecture-1">Introduction to LLM and RAG</summary>


## Introduction
- Focus: Practical applications of LLMs with an emphasis on **RAG (Retrieval-Augmented Generation)**.

## Course Overview
- **Problem Statement**:
  - Goal: Create a Q&A system using LLMs to simplify finding answers in FAQ documents.

## Objective
- **Task**:
  - Use data from existing FAQs to build a Q&A system.
  - The system will take user questions and search FAQ documents to generate answers.
- **Components**:
  - A form where users input questions and receive answers.

## Key Concepts
### LLM (Large Language Models)
- **Definition**: LLMs predict the next word/token in a sequence.
- **Examples**: Basic phone text suggestions, ChatGPT.
- **Functionality**:
  - Simple models predict the next word.
  - Large models with billions of parameters provide contextually rich responses.

### RAG (Retrieval-Augmented Generation)
- **Definition**: Combining retrieval of information with LLM text generation.
- **Components**:
  - **Retrieval**: Searching a knowledge base (e.g., FAQ documents).
  - **Generation**: Using LLM to generate responses based on retrieved context.
  - 
![image](https://github.com/tankudo/LLM-ZoomCamp/assets/58089872/51fc3f18-7563-4df5-8604-6a8ee1ea8168)

## Practical Implementation
- **Process**:
  1. **Question** from the user.
  2. **Retrieve** relevant documents from the knowledge base.
  3. **Generate** a response using the LLM, augmented by retrieved context.
- **Example**:
  - User asks about course enrollment.
  - System searches FAQ documents for relevant information.
  - LLM generates a comprehensive answer based on the retrieved data.

## Detailed Steps
1. **Input**: User's text or question (Prompt).
2. **LLM Output**: Answer based on the prompt.
3. **Retrieval Process**:
   - Search FAQ documents for related entries.
   - Use retrieved documents as context for LLM.
4. **Augmented Generation**:
   - Combine question and context.
   - Generate an answer using LLM.
5. **Return** the answer to the user.

## Course Structure
- **Modules**:
  - Introduction to simple search engines.
  - Implementing ElasticSearch.
  - Exploring advanced search techniques like vector search.

## Conclusion
- The course aims to teach building a robust Q&A system using LLMs and retrieval techniques.
- Students will learn to implement and refine search mechanisms to enhance LLM responses.



</details>

<details>
  <summary id="lecture-2">Configuring Your Environment</summary>
  
  
  ## Introduction
  
Configuration of the environment for a machine learning project, demonstration of GitHub Codespaces usage.

## Tools and Setup
- **Docker**: Not covered in detail, Codespaces has docker.
- **Notebook Providers**: You can use Google Colab, Saturn Cloud, SageMaker, or run locally.

## Preparing for the Second Module
- **GPU Requirement**: Needed for the second module. Ensure you have access to a GPU machine.

## Setting Up GitHub Codespaces
1. **Create a Repository**: Make it public to share notebooks and homeworks.
2. **Launch Codespace**: Click on 'Create Codespace on Main' from the 'Code' tab.
3. **Install Extensions**: Ensure the Codespaces extension is installed in Visual Studio Desktop.

## Check your enviroment
  - `docker run hello-word`
  - `python -V`
    
## Installing Libraries
- Use `pip install` to set up the required libraries:
  - `tqdm`
  - `jupyter notebook==7.1.2`
  - `openai`
  - `elasticsearch`
  - `scikit-learn`
  - `pandas`

## Using OpenAI
1. **Register at (platphormopenai.com)**
   - go to API keys
   - press "create new API key"
   - give it a name and create secret key
2. **Set Environment Variable**: `export OPENAI_API_KEY="your_key"`
3. **Start Jupyter Notebook**: Use `jupyter notebook` to launch the environment.
4. **Access Notebook**: Use the forwarded port (e.g., `localhost:8888`) to open Jupyter in the browser.
5. Copy your tocken from the terminal

## Example Code for OpenAI API
```python import openai
from openai import OpenAI

# Create client
client = openai.Client(api_key='your_api_key')

# if you need to check your key
import os
os.environ

# Create a chat request
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "user", "content": "Is it too late to join the course?"}
    ]
)

# Print the response
print(response.choices[0].message['content'])
```
## Alternative Environment Setup with Anaconda

Download Anaconda:

`wget https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Linux-x86_64.sh`

Miniconda Installation:

`wget https://repo.anaconda.com/miniconda/Miniconda3-py310_24.4.0.0-Linux-x86_64.sh`

Initialize and Check:

`source ~/.bashrc
which python
python --version`

Install Required Libraries:

`pip install tqdm jupyter notebook openai elasticsearch scikit-learn pandas`
## Conclusion

By the end of this setup, you should have a fully functional environment ready for machine learning projects using either GitHub Codespaces or Anaconda. Ensure you keep your OpenAI API key secure and never expose it publicly.

</details>

<details>
  <summary id="lecture-3">Retrieval and Search</summary>

  This lecture is about the concept of retrival. The search engine **Minserch** created in introduction videos will be used.
  
  ### The rag Framework
- The framework consists of two components: the database and LLM.
- For the database, we will use a simple search engine implemented in one of the pre-course workshops.
- In the course repository, you can find a workshop on implementing a search engine, including a video and GitHub repo.
- The search engine is an in-memory search engine for illustration purposes, not production-ready.
- Later in the module, we will replace it with Elastic Search.

### Implementing a Search Engine
- We'll use a simple search engine from the workshop, populate it with FAQ documents, perform a search, and use the results in an LLM to get answers to questions.
- There is a Python file, `minsearch.py`, which implements the search functionality.

### Setting Up the Environment
- Start a new Jupyter notebook named "rag-intro.ipynb".
- Download the search engine implementation using the `wget` command and import it as a package.

### Loading and Processing Data
- The FAQ documents are in JSON format, with each course containing a JSON object that includes the question, section, and text (answer).
- To use these documents, load them into the search engine by:
  1. Importing the JSON library.
  2. Opening the JSON file.
  3. Converting the nested structure into a flat list of dictionaries.
  
### Indexing Documents
- Use the `minsearch` library to index the documents.
- Specify which fields are text fields and which are keyword fields.
- Keyword fields allow for exact filtering, similar to SQL queries.
- Text fields are used for performing the search.

### Performing a Search
- Create an index with the specified text and keyword fields.
- Example query: "The course has already started, can I still enroll?"
- Use boosting to give more importance to certain fields (e.g., question field over the text field).

### Search Implementation
- Fit the index to the documents.
- Execute the query to retrieve relevant documents.
- Filter the results to restrict them to the relevant course (e.g., Data Engineering Zoom Camp).
- Use boost if you need to set one field more then another.

### Retrieving and Using Results
- Retrieve the most relevant documents for the query.
- Use the documents as context for the LLM.
- Next steps involve building a prompt using these documents as context for the LLM.

### Conclusion
- We have implemented the first step: indexing the knowledge base and retrieving context for queries.
- The next video will cover using these documents in an LLM.  

</details>
<details>
  <summary id="lecture-4"> Generating Answers with OpenAI GPT 4-o</summary>

## Generation

### Overview 
- Performing a search using a user query.

### Demonstration
- Example query: "The course has already started, can I still enroll?"
- Retrieve relevant answers from our knowledge base.
- The goal is to use these documents as context in an LLM to answer user queries.

### Using LLMs
- Use OpenAI's GPT-4o for demonstration.
- The LLM will use the retrieved documents as context for generating answers.

### Setting Up the Environment 
- Import OpenAI and set up the API key.
- Configure the environment.
  ```
  from openai import OpenAI
  client = OpenAI()
  ```
### Building the Prompt
- Form a prompt and send it to OpenAI or another LLM.
- Use GPT-4o, which is fast and cost-effective compared to GPT-3.5.
- Prepare the API client and define the user query.
```
response = client.chat.completions.create(
    model='gpt-4o',
    messages=[{"role": "user", "content": q}]
)
```
### Crafting the Prompt Template
- Assign a role to the LLM, e.g., "course teaching assistant."
- Structure the prompt to include the user's question and context from the knowledge base.
- Specify that the LLM should use only the provided context for answers.
```
prompt_template = """
You're a course teaching assistant. Answer the QUESTION based on the CONTEXT from the FAQ database.
Use only the facts from the CONTEXT when answering the QUESTION.

QUESTION: {question}

CONTEXT: 
{context}
""".strip()
```

### Generating the Answer
- Build the context by iterating over the retrieved documents.
  ```
  context = ""
    
    for doc in search_results:
        context = context + f"section: {doc['section']}\nquestion: {doc['question']}\nanswer: {doc['text']}\n\n"
  ```
- Format the prompt with the user's question and the context.
  ```
  prompt = prompt_template.format(question=query, context=context).strip()
  ```
- Send the prompt to GPT-4 and retrieve the generated answer.
  ```
  response = client.chat.completions.create(
        model='gpt-4o',
        messages=[{"role": "user", "content": prompt}]
    )```
### Conclusion
- Accomplished the goal of generating an answer based on retrieved context.
- Next steps: modularize the code, improve logic, and prepare for easy replacement of the search engine or LLM.
- See you in the next video where we will clean and modularize the code.

</details>
