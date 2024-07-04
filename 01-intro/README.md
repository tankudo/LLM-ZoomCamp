# Course Module

## Table of Contents
- [Introduction to LLM and RAG](#lecture-1)
- [Configuring Your Environment](#lecture-2)
- [Retrieval and Search](#lecture-3)

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
![image](https://github.com/tankudo/LLM-ZoomCamp/assets/58089872/3304c12a-07e9-4eb2-b220-ef9d0ce1fa13)

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

  

</details>
