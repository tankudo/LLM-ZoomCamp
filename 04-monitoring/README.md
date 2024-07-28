## Table of Contents
- [Introduction to monitoring answer quality](#lecture-1)
- [Evaluation and Monitoring in LLMs](#lecture-2)
- [Offline RAG Evaluation](#lecture-3)
- [](#lecture-4)
- [](#lecture-5)
- [](#lecture-6)
- [](#lecture-7)
- [](#lecture-8)
- [](#lecture-9)
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
  
  <summary id="lecture-4"> </summary>
  
</details>
<details>
  
  <summary id="lecture-5"> </summary>
  
</details>

<details>
  
  <summary id="lecture-6"> </summary>
  
</details>

<details>
  
  <summary id="lecture-7"> </summary>
  
</details>

<details>
  
  <summary id="lecture-8"> </summary>
  
</details>

<details>
  
  <summary id="lecture-9"> </summary>
  
</details>


