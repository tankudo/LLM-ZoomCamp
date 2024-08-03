## Table of Contents
- [Introduction to monitoring answer quality](#lecture-1)
- [Evaluation and Monitoring in LLMs](#lecture-2)
- [Offline RAG Evaluation](#lecture-3)
- [Offline RAG Evaluation: Cosine Similarity](#lecture-4)
- [Offline RAG Evaluation: LLM as a Judge](#lecture-5)
- [Capturing User Feedback](#lecture-6)
- [Capturing User Feedback, Part 2](#lecture-7)
- [Monitoring the System](#lecture-8)
- [Extra Grafana video](#lecture-9)
---

<details>
  
  <summary id="lecture-1"> Introduction to monitoring answer quality</summary>
  # Lecture Notes on Monitoring LLM Systems

## Introduction
- This week will be all about monitoring.
- 
## Focus of This Week
- Observing and monitoring the quality of LLM answers.
- Discussing methods to monitor and ensure the quality of LLM outputs.

## Key Topics
- Methods for monitoring LLM systems.

## Importance of Monitoring
- Monitoring is crucial because deploying and forgetting is not enough.
- Continuous monitoring is essential to track LLM performance.

## Quality Monitoring
- Monitoring the quality of LLM outputs.
- ompute different quality metrics.
- Use Grafana to visualize metrics over time.
- Utilize user feedback to assess LLM performance.
- Collect chat sessions and user feedback, visualize in Grafana.

## Detailed Monitoring Topics
-  Reasons for monitoring LLM systems.
  - LLMs generate creative and diverse answers, requiring monitoring.
  - Example: AI chatbot becoming racist, a reminder of the need for monitoring.

## Metrics for Quality Assessment
1. **Mathematical Approach**
   - Vector similarity metric: Compare LLM-generated answers with expected answers using vector embeddings.
2. **LLM as a Judge**
   - Use LLMs to detect toxicity in answers.
3. **Prompt Evaluation**
   -  Ask LLMs to evaluate the coherence of generated answers against expected answers.

## Implementation
- Store computed metrics in a relational database (PostgreSQL).
- Use Docker and Docker Compose for easy setup and connection with Grafana.
- Collect and visualize user feedback and chat sessions in Grafana.

## Advanced Monitoring Topics
- Additional monitoring aspects such as bias and fairness.
- Understand customer interactions using topic clustering.
- Track structured feedback (thumbs up/down) and unstructured textual feedback.
- Analyze negative feedback and corresponding chat sessions.
- Monitor indirect feedback like copy-pasting of responses.

## DevOps Perspective
- Monitor system metrics such as latency, traffic, errors, and saturation (the four golden signals).

## Conclusion
- Continuous monitoring and improvement of LLM systems are necessary for maintaining high-quality performance and customer satisfaction.

</details>

<details>
  
  <summary id="lecture-2"> Evaluation and Monitoring in LLMs</summary>
  
## Evaluation and monitoring are closely related. We will start with offline evaluation.

### Offline Evaluation
- **Specific Focus:** Evaluating RAG (Retrieval-Augmented Generation), but applicable to other LLMs.
- **Goal:** Evaluate the quality of LLM applications, including a recap of previous modules.

### Recap of Previous Modules

- **Overview:** 
  - **First Module:** Defined the RAG flow:
    - Query
    - Search results
    - Prompt creation based on query and search results
    - Using an LLM to generate the answer
 ```python
  def rag(q):
    search_results = search(q)
    promt = build_promt(Q, search_results)
    answer = llm(promt)
    return(answer)   
 ```

- **Second Module:** Replacing OpenAI with other LLMs.

- **Third Module:** 
  - **Focus:** Vector search and evaluating retrieval.
  - **Metrics:** Hit rate, Mean Reciprocal Rank (MRR).
  - **Evaluation:** Various ways to implement and evaluate the search function.
    - We know how to evaluate retrival and now we need to know how to evaluate the prompt and LLM.

### Evaluating the Entire System

- **Approaches:** Offline and Online evaluation.
  - **Offline Evaluation:** Metrics like hit rate to evaluate search results before deployment.
      

- **Online Evaluation:** 
  - **Methods:** A/B tests, user feedback (thumbs up/down), and monitoring overall system health.
  - **Metrics:** Performance metrics like CPU usage, user feedback, and answer quality.

## Offline Evaluation in Detail

- **Focus:** Offline evaluation including cosine similarity and LLM as a judge.

### Cosine Similarity

- **Definition:** Measure how close the generated answer is to the expected answer.
- **Process:**
  - Create a test dataset with Q&A pairs.
  - Use LLM to generate answers for the questions.
  - Compute cosine similarity between original and generated answers.

### LLM as a Judge

- **Process:**
  - Ask the LLM to judge the similarity between the original and generated answers.
  - Alternatively, ask the LLM to judge how well the generated answer addresses the question directly.

## Conclusion

- **Next Steps:** In the next video, we will delve deeper into the offline evaluation of RAG systems and compute these metrics.
  
</details>

<details>
  
  <summary id="lecture-3">Offline RAG Evaluation </summary>
 
 ## Recap and Introduction to ROC Function
- **Recap**: Summary of previous course content.
- **Evaluation**: Discussing evaluation methods.
- **Objective**: Evaluating the ROC function, which consists of three components.

## Evaluation of the Entire Function
- **Previous Evaluation**: Evaluated only part of the function.
- **Current Evaluation**: Evaluating the entire function using the same dataset from the previous module and the synthetically generated dataset.
- **Process**: Generate a question, produce an answer, and compute the similarity between the original and generated answers.

## Preparation
- **Preparation**: Initial setup in the notebook, including loading the ground truth dataset and creating an index for documents.
```python
import requests 

base_url = 'https://github.com/DataTalksClub/llm-zoomcamp/blob/main'
relative_url = '03-vector-search/eval/documents-with-ids.json'
docs_url = f'{base_url}/{relative_url}?raw=1'
docs_response = requests.get(docs_url)
documents = docs_response.json()
```

## Implementation Details
- **Data Setup**: 
  - Loaded documents with IDs.
  - Created a question-answer pair and assigned IDs.
  - Loaded ground truth data from the previous module.
  - Created an index for quick retrieval of documents.
```python
import pandas as pd

base_url = 'https://github.com/DataTalksClub/llm-zoomcamp/blob/main'
relative_url = '03-vector-search/eval/ground-truth-data.csv'
ground_truth_url = f'{base_url}/{relative_url}?raw=1'

df_ground_truth = pd.read_csv(ground_truth_url)
df_ground_truth = df_ground_truth[df_ground_truth.course == 'machine-learning-zoomcamp']
ground_truth = df_ground_truth.to_dict(orient='records')
```
```python
doc_idx = {d['id']: d for d in documents}
doc_idx['5170565b']['text']
```
- **Functionality**:
  - Used a vector search model for evaluating question and text pairs.
  - Indexed questions and answers.
  - Employed an elastic search function for retrieving results.
  - Modified the query format to a dictionary for better handling.
```python    
from sentence_transformers import SentenceTransformer

model_name = 'multi-qa-MiniLM-L6-cos-v1'
model = SentenceTransformer(model_name)
```
```python
from elasticsearch import Elasticsearch
```
```python
es_client = Elasticsearch('http://localhost:9200') 

index_settings = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0
    },
    "mappings": {
        "properties": {
            "text": {"type": "text"},
            "section": {"type": "text"},
            "question": {"type": "text"},
            "course": {"type": "keyword"},
            "id": {"type": "keyword"},
            "question_text_vector": {
                "type": "dense_vector",
                "dims": 384,
                "index": True,
                "similarity": "cosine"
            },
        }
    }
}

index_name = "course-questions"

es_client.indices.delete(index=index_name, ignore_unavailable=True)
es_client.indices.create(index=index_name, body=index_settings)
```
```python
from tqdm.auto import tqdm

for doc in tqdm(documents):
    question = doc['question']
    text = doc['text']
    doc['question_text_vector'] = model.encode(question + ' ' + text)

    es_client.index(index=index_name, document=doc)
```
### Retrieval
```python
def elastic_search_knn(field, vector, course):
    knn = {
        "field": field,
        "query_vector": vector,
        "k": 5,
        "num_candidates": 10000,
        "filter": {
            "term": {
                "course": course
            }
        }
    }

    search_query = {
        "knn": knn,
        "_source": ["text", "section", "question", "course", "id"]
    }

    es_results = es_client.search(
        index=index_name,
        body=search_query
    )
    
    result_docs = []
    
    for hit in es_results['hits']['hits']:
        result_docs.append(hit['_source'])

    return result_docs

def question_text_vector_knn(q):
    question = q['question']
    course = q['course']

    v_q = model.encode(question)

    return elastic_search_knn('question_text_vector', v_q, course)
```
```python
question_text_vector_knn(dict(
    question='Are sessions recorded if I miss one?',
    course='machine-learning-zoomcamp'
))
```
### The RAG flow
```python
def build_prompt(query, search_results):
    prompt_template = """
You're a course teaching assistant. Answer the QUESTION based on the CONTEXT from the FAQ database.
Use only the facts from the CONTEXT when answering the QUESTION.

QUESTION: {question}

CONTEXT: 
{context}
""".strip()

    context = ""
    
    for doc in search_results:
        context = context + f"section: {doc['section']}\nquestion: {doc['question']}\nanswer: {doc['text']}\n\n"
    
    prompt = prompt_template.format(question=query, context=context).strip()
    return prompt
```
```python
from openai import OpenAI

client = OpenAI()

def llm(prompt, model='gpt-4o'):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.choices[0].message.content
```
## Evaluating Similarity
- **Approach**:
  - Generated an answer for each question.
  - Computed cosine similarity between original and generated answers.
  - Cosine similarity ranges from 0 (not similar) to 1 (identical).

#### Cosine similarity metric
```python
answer_orig = 'Yes, sessions are recorded if you miss one. Everything is recorded, allowing you to catch up on any missed content. Additionally, you can ask questions in advance for office hours and have them addressed during the live stream. You can also ask questions in Slack.'
answer_llm = 'Everything is recorded, so you won’t miss anything. You will be able to ask your questions for office hours in advance and we will cover them during the live stream. Also, you can always ask questions in Slack.'

v_llm = model.encode(answer_llm)
v_orig = model.encode(answer_orig)

v_llm.dot(v_orig)
```
- **Process**:
  - Created vectors for both original and generated answers.
  - Calculated cosine similarity.
  - Compared the results for performance evaluation.

## Loop and Data Handling
- **Loop**:
  - Iterate over the ground truth dataset.
  - Generate and store answers in a dictionary.
  - Use GPT-4 for generating answers (can be expensive).
  - Consider using GPT-3.5 for cost efficiency.
 
```python
answers = {}
```
```python
for i, rec in enumerate(tqdm(ground_truth)):
    if i in answers:
        continue

    answer_llm = rag(rec)
    doc_id = rec['document']
    original_doc = doc_idx[doc_id]
    answer_orig = original_doc['text']

    answers[i] = {
        'answer_llm': answer_llm,
        'answer_orig': answer_orig,
        'document': doc_id,
        'question': rec['question'],
        'course': rec['course'],
    }
```
## Saving and Analyzing Results
- **Saving Results**:
  - Saved answers in a separate cell to avoid losing data on errors.
  - Considered saving results as JSON or CSV files.
  - Chose CSV for simplicity and used pandas for data handling.

- **Sample Data**:
  - Displayed a sample of five records to check the format and content.

## Final Remarks
- **Execution Time**:
  - The entire process took approximately 2-3 hours.
  - Prepared to share the results to save others from re-execution costs.
 
### Evaluating GPT 3.5
```python
rag(ground_truth[10], model='gpt-3.5-turbo')
```
```python
from tqdm.auto import tqdm

from concurrent.futures import ThreadPoolExecutor

pool = ThreadPoolExecutor(max_workers=6)

def map_progress(pool, seq, f):
    results = []

    with tqdm(total=len(seq)) as progress:
        futures = []

        for el in seq:
            future = pool.submit(f, el)
            future.add_done_callback(lambda p: progress.update())
            futures.append(future)

        for future in futures:
            result = future.result()
            results.append(result)

    return results
```
```python
def process_record(rec):
    model = 'gpt-3.5-turbo'
    answer_llm = rag(rec, model=model)
    
    doc_id = rec['document']
    original_doc = doc_idx[doc_id]
    answer_orig = original_doc['text']

    return {
        'answer_llm': answer_llm,
        'answer_orig': answer_orig,
        'document': doc_id,
        'question': rec['question'],
        'course': rec['course'],
    }
```
```python
results_gpt35 = map_progress(pool, ground_truth, process_record)
```
```python
df_gpt35 = pd.DataFrame(results_gpt35)
df_gpt35.to_csv('data/results-gpt35.csv', index=False)
```
- **Next Steps**:
  - Continue evaluating the similarity metrics.
  - Prepare for further offline evaluation before production roll-out.
  - Ensure robustness by comparing different LLM prompts and models.

## Summary
- **Offline Evaluation**:
  - A critical step before deploying models into production.
  - Helps in comparing different prompts and models.
  - Provides a structured approach to measure the effectiveness of the entire function.
 
</details>
<details>
  
  <summary id="lecture-4">Offline RAG Evaluation: Cosine Similarity </summary>
  
## Offline Evaluation
   - In this camp, we talk about offline evaluation of LLMs.
   - By "offline," I mean metrics we calculate offline before deployment.

## Importance of Offline Metrics
   - Offline metrics are calculated in our test or development environment.
   - These metrics are assessed before deploying to production.
   - Calculations are done in a controlled setting.
   - Metrics are computed in our test environment.
   - Alternatively, they can be computed in the development environment.
   - If the metrics look good, we push to production.
   - Then, we proceed with online evaluation.

## Previous Work and Current Focus
  - Continuing from our previous lecture, where we created a notebook.
  - Notebook loaded with ground truth data.
  - Each question had an original answer and an LLM-generated answer.
  - Answers were generated using our rag function.
  - We used two models: 3.5 and 4.0.
  - Results stored in a CSV file.

## Cosine Similarity Metric
**Cosine Similarity Explanation**
  - Cosine similarity is used to compare the generated answer to the original.
  - Original answer (A) -> Synthetic question -> LLM answer -> Compute cosine similarity between A and LLM answer.
  - Calculated for each record and stored.

## Implementation Details
  - Loaded data into dictionaries for easier manipulation.
  - Computed similarity scores for each record.
  - Results stored and mean cosine similarity computed using pandas.
```python
def compute_similarity(record):
    answer_orig = record['answer_orig']
    answer_llm = record['answer_llm']
    
    v_llm = model.encode(answer_llm)
    v_orig = model.encode(answer_orig)
    
    return v_llm.dot(v_orig)
```
```python
for record in tqdm(results_gpt4o):
    sim = compute_similarity(record)
    similarity.append(sim)
```
```python
df_gpt4o['cosine'] = similarity
df_gpt4o['cosine'].describe()
```
```python
results_gpt35 = df_gpt35.to_dict(orient='records')

similarity_35 = []

for record in tqdm(results_gpt35):
    sim = compute_similarity(record)
    similarity_35.append(sim)
```
```python
df_gpt35['cosine'] = similarity_35
df_gpt35['cosine'].describe()
```
## Model Comparison
  - Both models showed similar distributions in cosine similarity scores.
  - Minor differences, with GPT-3.5 being slightly worse but significantly cheaper and faster.
```python
import matplotlib.pyplot as plt

sns.displot(df_dpt4o['cosine'], label='4o')
sns.displot(df_dpt35['cosine'], label='3.5')
```
![image](https://github.com/user-attachments/assets/f40a2de9-3548-4e49-a1c2-cc58dc2a2859)

## New Model Evaluation
7. **Testing GPT-4 Mini**
  - New model "GPT-4 Mini" claimed to be cheaper and faster.
  - Initial tests faced rate limits, suggesting a more careful approach for evaluation.
  - Plan to complete evaluation later due to rate limits.
![image](https://github.com/user-attachments/assets/2677e4b7-95d0-4d38-a9ad-f743dc804e68)

## Summary
8. **Conclusion**
    - Established a method for offline LLM evaluation using cosine similarity.
    - Compared multiple models to assess performance and cost-efficiency.
    - Future work includes comprehensive evaluation of new models under rate limits.
</details>
<details>
  
  <summary id="lecture-5">Offline RAG Evaluation: LLM as a Judge</summary>

### Introduction
The goal is to have a comprehensive method for evaluating all three steps in the RAG flow: Search, Prompt, and LLM.

### Evaluation Metrics
We use different evaluation metrics for this purpose. We've already discussed cosine similarity, which was used to evaluate three models: GPT-3.5 Turbo, GPT-4.0, and GPT-4.0 Mini. The conclusion was that GPT-4.0 Mini provided the best results in terms of cosine similarity.

### Alternative Evaluation Approach
Instead of relying solely on cosine similarity, we can also use an LLM to judge the quality of the outputs. For example, we can ask the LLM to evaluate if the generated content is good enough.

### Using LLM as a Judge
We can compare original questions and answers with generated questions and answers. This method helps us assess the relevance and quality of the generated responses.

### Prompt Engineering
To facilitate this evaluation, let's prepared a prompt for GPT-4, asking it to act as a prompt engineering expert. The prompt was designed to generate questions and assess the ability to recover original answers.

#### Prompt Details
1. **First Case**: 
   - Input: Original answer, generated question, and generated answer.
   - Purpose: For offline evaluation where we have access to the original answer.

2. **Second Case**: 
   - Input: Generated question and generated answer.
   - Purpose: For online evaluation where the original answer is not available.

### LLM-as-a-Judge
```python
prompt1_template = """
You are an expert evaluator for a Retrieval-Augmented Generation (RAG) system.
Your task is to analyze the relevance of the generated answer compared to the original answer provided.
Based on the relevance and similarity of the generated answer to the original answer, you will classify
it as "NON_RELEVANT", "PARTLY_RELEVANT", or "RELEVANT".

Here is the data for evaluation:

Original Answer: {answer_orig}
Generated Question: {question}
Generated Answer: {answer_llm}

Please analyze the content and context of the generated answer in relation to the original
answer and provide your evaluation in parsable JSON without using code blocks:

{{
  "Relevance": "NON_RELEVANT" | "PARTLY_RELEVANT" | "RELEVANT",
  "Explanation": "[Provide a brief explanation for your evaluation]"
}}
""".strip()
```
```python
prompt2_template = """
You are an expert evaluator for a Retrieval-Augmented Generation (RAG) system.
Your task is to analyze the relevance of the generated answer to the given question.
Based on the relevance of the generated answer, you will classify it
as "NON_RELEVANT", "PARTLY_RELEVANT", or "RELEVANT".

Here is the data for evaluation:

Question: {question}
Generated Answer: {answer_llm}

Please analyze the content and context of the generated answer in relation to the question
and provide your evaluation in parsable JSON without using code blocks:

{{
  "Relevance": "NON_RELEVANT" | "PARTLY_RELEVANT" | "RELEVANT",
  "Explanation": "[Provide a brief explanation for your evaluation]"
}}
""".strip()
```
### Execution of the Evaluation
We executed the evaluation for a sample of 100 to 200 records to ensure the results are reproducible. Here’s a step-by-step overview of the process:

1. **Sampling Data**: 
   - Set a seed for reproducibility.
   - Sample a subset of the data.
```python  
df_sample = df_gpt4o_mini.sample(n=150, random_state=1)
```
```python
samples = df_sample.to_dict(orient='records')
```
2. **Creating Prompts**: 
   - Generate prompts based on the sampled data.
   - Use the LLM to evaluate the prompts.
```python
prompt = prompt1_template.format(**record)
print(prompt)
```
```python
answer = llm(prompt, model='gpt-4o-mini')
```
```python
import json
```
```python
evaluations = []

for record in tqdm(samples):
    prompt = prompt1_template.format(**record)
    evaluation = llm(prompt, model='gpt-4o-mini')
    evaluations.append(evaluation)
```
3. **Parsing Results**: 
   - Ensure the output is in JSON format for easy parsing.
   - Handle potential issues with non-parseable outputs by refining prompts.

### Handling Errors
Errors can occur when the LLM returns code blocks or non-JSON outputs. We adjusted the prompts to avoid such issues and ensure consistent, parseable JSON outputs.

### Results and Analysis
After executing the prompts and parsing the results, we analyzed the distribution of relevant and non-relevant answers. This involved:

- Counting the occurrences of relevant and non-relevant evaluations.
- Identifying examples of non-relevant evaluations to understand common issues.
  ```python
  df_evaluations[df_evaluations.Relevance == 'NON_RELEVANT']
  ```

### Conclusion
The approach of using LLMs to judge the quality of generated answers provides valuable insights into the effectiveness of our ROCK system. By refining prompts and ensuring JSON outputs, we can systematically evaluate and improve our models.

### Next Steps
1. Further refine the evaluation prompts.
2. Execute the evaluation on a larger dataset.
3. Analyze and address common issues in non-relevant evaluations.

</details>

<details>
  
  <summary id="lecture-6">Capturing User Feedback </summary>

### Introduction
In this segment of the lecture, the focus is on how to capture and monitor user feedback within an LLM (Large Language Model) application. The goal is to integrate a user interface that allows users to provide feedback on the responses they receive from the model.

## Capturing User Feedback
The primary aim is to capture user feedback effectively. This can be achieved by integrating features like thumbs up and thumbs down buttons, which allow users to express their satisfaction or dissatisfaction with the responses provided by the LLM.

## Implementation Overview
To implement this, we need:
- A user interface for feedback (thumbs up/down).
- Logic to handle the feedback data.
- Storage to save user feedback along with the question asked and the response received from the LLM.

### Experiment with Claude
An experiment is introduced using **Claude**, which is an alternative to ChatGPT, built with a focus on LLM applications. The idea is to create the required user feedback mechanism using Clot and explore how it performs compared to ChatGPT. Use stream application from the second module.

### Plan
1. Add thumbs up and thumbs down buttons to the application.
2. Store the user feedback along with the question and the LLM’s response in a PostgreSQL database.
3. Use Docker to containerize the application, making the setup and deployment easier.

## Coding and Implementation
- **Streamlit Application Setup:** 
  The existing Streamlit application is enhanced by adding thumbs up and thumbs down buttons. The plan includes moving the core logic out of the main application into separate modules for better code management.

- **Ollama Integration:** 
  The Ollama model is integrated for LLM responses. The decision is made to continue using Ollama rather than switching to OpenAI’s GPT, to maintain the course's focus on using open-source LLMs.

- **Database Setup:**
  PostgreSQL is selected to store conversations, user feedback, and associated data. The database model will capture the question, the response, and whether the feedback was positive or negative.

- **Dockerization:**
  The application is containerized using Docker. The Docker Compose file is updated to include:
  - A persistent data storage for Elasticsearch to avoid re-indexing every time the container is restarted.
  - A PostgreSQL container for the database.
  - The Streamlit application container.

## Execution and Testing
- **Initial Setup and Docker Compose:**
  The Docker Compose setup is executed, which includes setting up Elasticsearch and PostgreSQL. Additional scripts are written to automate the database initialization and indexing processes.

- **Code Refactoring:**
  The code is refactored to improve readability and maintainability by:
  - Moving imports to the top.
  - Creating separate functions for each logical block.
  - Adding print statements for better logging.

- **Testing and Debugging:**
  After running the application, an error regarding missing Python modules (`psycopg2`) is encountered, which needs to be resolved by installing the necessary packages.

## Reindexing with Elasticsearch
- The process of reindexing with Elasticsearch involves executing previously done steps to ensure the data is indexed properly.
- Commands are run incrementally, and errors or issues during execution are troubleshooted by commenting out unnecessary code blocks.

## Troubleshooting Elasticsearch Reindexing
- When encountering issues such as a lack of apparent activity during reindexing, the instructor introduces debugging steps, such as writing the status to logs and ensuring correct installation.

## Connecting to PostgreSQL via PG CLI
- The session demonstrates connecting to a PostgreSQL database using the `pgcli`.
- Key steps include:
  1. Running commands to check if the database connection is successful.
  2. Listing databases and tables using PostgreSQL commands (`\l` and `\dt`).
  3. Ensuring the database setup is correct by verifying the existence of expected tables.

## Docker Compose and Environment Configuration
- The Docker Compose setup is reviewed to ensure it correctly loads data and interacts with services like PostgreSQL and Elasticsearch.
- Important points include:
  - The use of environment variables like `ELASTIC_URL_LOCAL=http://localhost::9201` and `ELASTIC_URL=http://elacticsearch:9200` to manage service connections.
  - Adjusting Docker configuration to ensure services pick up the correct files and run as expected.

## Testing and Adjusting Streamlit for Vector Search
- Streamlit is tested with text-based search initially, and then adjustments are planned for integrating vector search.
- The transition involves modifying the query handling to switch from text search to vector search.

## Rebuilding Docker Services
- The process of stopping, rebuilding, and restarting Docker services is discussed to address issues such as outdated or cached configurations.
- Commands like `docker-compose stop`, `docker-compose build`, and `docker-compose up` are used iteratively to ensure the services are running correctly.

## Downloading Models and Ensuring Service Stability
- For the Ollama model, a Docker container is used to download and set up the model.
- Testing involves checking the logs for errors and ensuring the model integration works with the application.

## Debugging and Logging
- Emphasis is placed on using logs to debug services within Docker.
- The discussion includes:
  - Adding logging or print statements in the code to understand what is happening during execution.
  - Using Docker commands to follow logs (`docker logs -f`) and observe the application's behavior in real-time.

## Final Adjustments and Service Testing
- Final steps involve ensuring that all services (PostgreSQL, Elasticsearch, Ollama) are correctly initialized and running.
- The instructor points out the importance of verifying that each service interacts correctly with others before final deployment.

---

By the end of this section, the services are running, but further checks are necessary to confirm everything is functioning as expected, particularly in ensuring that Docker logs and Streamlit's behavior reflect the desired outcomes.
  
</details>

<details>
  
  <summary id="lecture-7">Capturing User Feedback, Part 2 </summary>

### Overview
In this part of the lecture, we focus on enhancing our assistant by adding the ability to use text search or semantic search and allowing the selection of different models (Ollama or OpenAI GPT-4). We will also integrate the Sentence Transformer library to facilitate vector search.

### Key Points

1. **Objective**:
    - Enable text search or semantic search in the assistant.
    - Integrate a model selection feature to choose between Ollama and GPT-4.
    - Add the Sentence Transformer library for vector search capability.

2. **Initial Setup**:
    - Discuss the need to add features for text and semantic search.
    - Introduce the requirement to integrate the Sentence Transformer library.

3. **Model Selection**:
    - Plan to replace Ollama with OpenAI GPT-4.
    - Provide a toggle feature to switch between Ollama and GPT-4.

4. **Implementing Search Options**:
    - Add a dropdown in Streamlit to select between text search and vector search.
    - Copy necessary code for vector search from a notebook.

5. **Code Adjustments**:
    - Modify the assistant to include the updated model and search options.
    - Adjust the code to handle the new functionalities, ensuring compatibility and performance improvements.

### Detailed Steps

1. **Update Requirements**:
    - Ensure all necessary libraries are included, such as OpenAI, Sentence Transformers, and others.
    - Verify the correct versions of dependencies, like PyTorch and Sentence Transformers.

2. **Modify the Assistant Code**:
    - Implement model selection logic.
    - Add search type logic to toggle between text and vector search.
    - Format and clean up the code for better readability and maintenance.

3. **Handling Dependencies**:
    - Address issues with dependencies like CUDA for GPU usage.
    - Ensure the assistant works efficiently with CPU-only dependencies to avoid unnecessary GPU downloads.

4. **Testing and Debugging**:
   - Faced challenges with unnecessary dependencies (e.g., pandas, Cuda).
   - Resolved issues by:
    - Restarting the assistant to apply changes.
    - Ensuring that the correct versions of all required libraries were installed and used.
  
### 6. Final Verification and Commit
  - Verified that the assistant now correctly:
    - Allows model selection.
    - Switches between text search and vector search.
  - Observed a minor issue where text input doesn't clear after submitting, but decided to address it later.
  - Committed the final version of the code.

### 7. Next Steps: Monitoring Setup
  - **Upcoming Tasks**: Set up Grafana for monitoring the assistant's performance.
  - Preparing to move on to the next phase of the project.

</details>

<details>
  
  <summary id="lecture-8">Monitoring the System </summary>
  
## Overview
In this series, we're discussing evaluation and monitoring. We have covered offline evaluation extensively and recently focused on online evaluation. In the last few videos, we explored capturing user feedback and integrating it into our system.

## Topics Covered

### Offline and Online Evaluation
- Extensive discussion on offline evaluation.
- Recent videos focused on online evaluation.
- Capturing user feedback using thumbs up/down.

### Integrating Feedback Mechanisms
- Combining all previous components into a unified system.
- Storing data in a PostgreSQL database.
- Capturing user feedback (thumbs up/down) to evaluate response relevancy.

### Using LLMs as Judges
- Different methods of using Large Language Models (LLMs) to judge response relevancy.
- QA relevancy: Asking LLMs to evaluate the relevance of answers to questions.

## Today's Objective

### Monitoring System Setup
- Integrating all components into a monitoring system.
- Building a dashboard to display various metrics:
  - Number of requests.
  - Relevance of responses.
  - Thumbs up/down counts.
  - Last 5 conversations with ability to filter by relevance.
  - Model used.
  - Prompt and completion token.
  - Cost of operations.

### Tools and Technologies
- **Grafana**: For creating real-time dashboards.
- **PostgreSQL**: As the backend database.

## Implementation Steps

### Preparing the Prompt
- Customize the prompt to include monitoring with Grafana.
- Specify the metrics to display (response time, relevance, token count, cost).
- Use PostgreSQL as the backend database.

### Modifying the Code
- Update the LLM invocation to return additional information (response time, relevancy, token count).
- Implement functions to log each LLM invocation.
- Adjust the cost calculation to be more accurate and moved to a separate function.

### Streamlit App Updates
- Display the monitoring information in the Streamlit app for immediate feedback.
- Add new fields to display logs and statistics.

### Setting Up Grafana
- Configure Grafana to connect to the PostgreSQL database.
- Create dashboards to display the metrics in real-time.

## Demonstration

### Running the Docker Compose
- Use `docker-compose up` to build and start the Streamlit app and Grafana.
- Initialize the database and ensure connectivity.

### Generating Synthetic Data
- Create a script to generate synthetic data for testing purposes.
- Use prompts to guide the creation and insertion of this data into the database.

### Accessing and Configuring Grafana
- Set up Grafana with the default login (admin/admin).
- Add PostgreSQL as a data source.
- Create and configure the first dashboard with SQL queries to display metrics.

## Conclusion
- Summary of integrating monitoring with Grafana.
- Explanation of displaying various metrics using the dashboard.
- Importance of real-time monitoring in evaluating LLM performance.

## Next Steps
- Continuously improve and adjust the monitoring system based on feedback.
- Explore advanced features in Grafana for more sophisticated monitoring.

</details>

<details>
  
  <summary id="lecture-9">Extra Grafana video </summary>
## Addressing Previous Issues

- In the previous video, there were issues with saving at the end. 
  - **Solution**: Adding a timestamp within SQL query, which is optional, and setting it to the current time when it's none. 
  - **Outcome**: It works fine now, and in Grafana, the latest questions can be seen.

## Course Questions

- Common questions about the course:
  - Can I take the course if I don't know math?
  - Can I take the course if I don't know Docker?

## Time Intervals in Grafana

- Discussing intervals in Grafana:
  - Changing the interval to the last 5 minutes updates the time series panels.
  - Some elements don't change based on intervals and always show data from the entire database.
  - **Solution**: Use Grafana variables to filter data. For example, adding a `WHERE` statement to filter conversations within a specific time range.

## Example Queries in Grafana

- Example of using Grafana variables:
  - Prepared queries in `grafon.md`.
  - Added variables to filter feedback and relevance statistics based on timestamps.
  - Running the queries to see updates in the specified interval.

## Saving and Loading Grafana Dashboards

- How to save and load Grafana dashboards:
  - Go to settings and click on the JSON model to copy and paste the dashboard JSON to Git.
  - Share and export the dashboard to a file.
  - Import the saved file to reload the dashboard.
  - Programmatically save and load dashboards using API instead of manual clicking.

## Conclusion

- Wrapping up the module:
  - The module has been long, but we've covered all necessary topics.
  - Time to do homework. See you soon!

</details>


