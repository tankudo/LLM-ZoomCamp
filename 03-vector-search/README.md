## Table of Contents
- [Introduction to Vector Search](#lecture-1)
- [Semantic Search with Elasticsearc](#lecture-2)
- [Advanced Semantic Search](#lecture-3)
- [Evaluation Metrics for Retrieval](#lecture-4)
- [Ground Truth Dataset Generation for Retrieval Evaluation](#lecture-5)
- [Evaluation of Text Retrieval Techniques for RAG ou](#lecture-6)
- [Evaluating Vector Retrival](#lecture-7)
---

<details>
  
  <summary id="lecture-1"> Introduction to Vector Search</summary>
  # Basics of Vector Database
  
- **Intuition**: Understanding the basics and intuition behind vector databases.

## Popular Use Case
- **Semantic Search**:
  - **Example**: Google searches for "how many employees does Apple have" vs. "where can I pick apples near me".
  - **Understanding Context**: Google uses semantic search to understand context and provide relevant results.

## Why Vector Databases are Popular
- **Data Explosion**: Increase in diversified data formats (text, image, audio).
- **Limitations of Classical Databases**: Not suited for handling unstructured data.
- **Large Language Models**: Lack long-term memory; vector databases help store and retrieve relevant information for better results.

## Vector Embeddings
- **Definition**: Embeddings are a bunch of numbers capturing associations of text.
- **Closeness in Meaning**: Data points closer in meaning are placed near each other in vector space.
- **Types of Embeddings**: Words, sentences, and large documents (legal, marketing) can all be converted into embeddings.

## Creating Vector Embeddings
- **Process**:
  - Gather and preprocess data.
  - Train a machine learning model to generate embeddings.
  - Iterative process to refine model for better embeddings.
- **Pre-trained Models**: Can use models from sources like Hugging Face.

## Vector Database
- **Storage and Retrieval**: Efficiently store and retrieve vast amounts of vector data.
- **Indexing**: Key for efficient retrieval; various indexing methods available.
- **Workflow**:
  - Convert data (image, document, audio) into embeddings using a pre-trained model.
  - Index and store embeddings in a vector database.
  - Query converted to vector, searched in the database, and relevant results retrieved based on similarity scores.

## Applications of Vector Database
- **Use Cases**:
  - Remember past data for large language models.
  - Semantic search, personalized recommendations, and text generation.

## Next Session
- **Lab**: Setup a semantic search engine using Elasticsearch.
- **Practical Application**: Demonstration of the discussed concepts.

</details>
<details>
  
  <summary id="lecture-2"> Semantic Search with Elasticsearc</summary>

- In this section, we will build a semantic search engine using Elasticsearch.
- We will use the same Python environment and Docker container as Module 1.

### Docker Setup

- Ensure Docker is running. If not, start it using the command from Module 1.
- Once Docker is up, we'll connect to Elasticsearch locally.

### Environment Setup

- Use the Python environment from Module 1.
- Install necessary packages, including additional ones if required.

### Architecture Overview

- We'll use a documents.json file, creating 'documents' for Elasticsearch.
- Pre-trained models from Hugging Face will generate embeddings.
- Embeddings are indexed into Elasticsearch.
- Queries will retrieve semantic results.
  ![image](https://github.com/user-attachments/assets/8a448a25-373e-4b42-b668-58b634b0acde)


### Elastic Search Concepts

- Understand 'documents' and 'index' concepts.
  - Documents: Collections of fields with associated values.
  - Index: Optimized collection of documents for efficient search.

### Data Preparation

- Convert documents.json into Elasticsearch-friendly format.
- Flatten hierarchy to ensure all data is on the same level.
```python
import json

with open('documents.json', 'rt') as f_in:
    docs_raw = json.load(f_in)

documents = []

for course_dict in docs_raw:
    for doc in course_dict['documents']:
        doc['course'] = course_dict['course']
        documents.append(doc)
```

### Package Installation

- Install Sentence Transformers package for embedding generation.
```python
# This is a new library compared to the previous modules. 
# Please perform "pip install sentence_transformers==2.7.0"
from sentence_transformers import SentenceTransformer

# if you get an error do the following:
# 1. Uninstall numpy 
# 2. Uninstall torch
# 3. pip install numpy==1.26.4
# 4. pip install torch
# run the above cell, it should work
model = SentenceTransformer("all-mpnet-base-v2")
```
### Sentence Transformers Overview

- Utilize pre-trained models to generate embeddings efficiently.
- Example usage and inference with Sentence Transformers.
```python
model = SentenceTransformer("all-mpnet-base-v2")

# different models have different length
len(model.encode("This is a simple sentence"))
```
### Embedding Generation

- Create embeddings for documents using selected models.
- Store embeddings alongside original data for indexing.
```python
#created the dense vector using the pre-trained model
operations = []
for doc in documents:
    # Transforming the title into an embedding using the model
    doc["text_vector"] = model.encode(doc["text"]).tolist()
    operations.append(doc)
``` 

### Elasticsearch Setup

- Ensure Elasticsearch Docker container is running locally.
- Connect to Elasticsearch instance from Python environment.
```python
from elasticsearch import Elasticsearch
es_client = Elasticsearch('http://localhost:9200') 
```

### Indexing

- Define mappings to structure data in Elasticsearch.
- Map fields and data types for efficient storage and retrieval.

### Mapping Creation

- Define schema-like mapping for Elasticsearch indexing.
- Specify fields, data types, and embedding dimensions.
```python
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
            "course": {"type": "keyword"} ,
            "text_vector": {"type": "dense_vector", "dims": 768, "index": True, "similarity": "cosine"},
        }
    }
}
index_name = "course-questions"

for doc in operations:
    try:
        es_client.index(index=index_name, document=doc)
    except Exception as e:
        print(e)
```
- Create end user query
```python
search_term = "windows or mac?"
vector_search_term = model.encode(search_term)

query = {
    "field": "text_vector",
    "query_vector": vector_search_term,
    "k": 5,
    "num_candidates": 10000, 
}
```
### Connection Verification

- Test Elasticsearch connection to ensure successful setup.
```python
res = es_client.search(index=index_name, knn=query, source=["text", "section", "question", "course"])
res["hits"]["hits"]
```
- Perform Keyword search with Semantic Search (Hybrid/Advanced Search)
```python
# Note: I made a minor modification to the query shown in the notebook here
# (compare to the one shown in the video)
# Included "knn" in the search query (to perform a semantic search) along with the filter  
response = es_client.search(
    index=index_name,
    query={
        "bool": {
          "multi_match": {"query" : "windows or python?,
                        "fields": ["text", "question", "course", "title"],
                        "type": "best_fields"
                          },
                },
          "filter" : {
              "term" : {
                        "course": "data-engineering-zoomcamp"
          }
        }
        }
    }
}

response["hits"]["hits"]
```
### Conclusion

- Architecture setup and initial data preparation complete.
- Ready to proceed with indexing and semantic search implementation.
</details>


<details>
  
  <summary id="lecture-3"> Advanced Semantic Search</summary>

### Understanding Semantic vs. Keyword Search
- The advanced semantic search is not truly semantic; it's more like a keyword search.
- Semantic search requires converting user input into vector embedding.

### Correct Implementation
- In this session, we correct the implementation for advanced semantic search.

### Code Example
- Here is the corrected code for advanced semantic search.
- Main difference: passing vector embedding from user input.
```python
knn_query = {
    "field": "text_vector",
    "query_vector": vector_search_term,
    "k": 5,
    "num_candidates": 10000
}

response = es_client.search(
    index=index_name,
    query={
        "match": {"section": "General course-related questions"},
    },
    knn=knn_query,
    size=5
)

response["hits"]["hits"]
```
### Results Analysis
- Scores in advanced search may exceed 1, indicating different scoring scales.
- Elastic search handles core functionality; scores can vary based on search complexity.

### Custom Scoring
- Use `explain=true` to understand score calculation.
- Custom scoring can align with business or user needs.


</details>
<details>
  
  <summary id="lecture-4"> Evaluation Metrics for Retrieval</summary>

- Let's discuss evaluation (Evola) in our search results.
- This is crucial in the retrieval (R) part where we retrieve data from our knowledge base.
- There are various methods of storing and retrieving data.
- We've seen Min search, Elasticsearch for text retrieval, and Vector search using Elasticsearch.
- What's the best method? It depends on your data and requirements.
- Evaluation techniques help determine the best approach.

### Understanding Evaluation Metrics

- Different evaluation metrics measure search effectiveness.
- Ground truth or gold standard data sets are essential for evaluation.
- Each query should have known relevant documents for evaluation.
- Evaluating system performance involves comparing retrieved documents to expected results.

### Practical Applications

- Use evaluation metrics to assess system performance objectively.
- Different metrics like precision, recall, and F1-score provide insights into retrieval quality.
- Experiment with various retrieval methods and parameters to optimize search results.
- Evaluate which method retrieves relevant documents most effectively.
- Understand the uniqueness of each data set and tailor your evaluation metrics accordingly.

### Generating Gold Standard Data

- Use human-labeled data or automated methods to create gold standard data sets.
- Assess and rank retrieval methods based on their ability to retrieve relevant documents.
- Discuss different evaluation metrics and their relevance in ranking systems.
- Explore techniques for generating and using gold standard data effectively.

### Conclusion and Next Steps

- Next, we'll delve into generating gold standard data sets using machine learning.
- In production, user feedback and annotators help refine evaluation metrics.
- Stay tuned for practical examples and applications in upcoming videos.

### Additional Resources

- Explore various evaluation metrics and their roles in assessing retrieval systems.
- Learn more about ranking metrics and their practical applications in search systems.

### Next Video Preview

- In the next video, we'll explore how to create a gold standard data set using NLP techniques.

</details>
<details>
  
  <summary id="lecture-5"> Ground Truth Dataset Generation for Retrieval Evaluation</summary>

### Ground Truth Dataset
- A ground truth dataset is crucial for evaluating the performance of retrieval systems.
- This dataset typically includes thousands of queries (1,000, 2,000, 10,000, or more) with known relevant documents.

### Creating Query-Document Pairs
- For each query in the dataset, the relevant documents from our knowledge base are identified.
- Often, multiple relevant documents exist for a single query.
- Simplification for this exercise: each query will have one known relevant document.

### Generating the Dataset
- Plan to generate a dataset with:
  - A large number of queries (e.g., 5,000 queries for 1,000 FAQ records).
  - For each query, identifying the relevant document.

### Using Human Annotators and Domain Experts
- Human annotators and domain experts can manually evaluate the relevance of documents to queries.
- This method, though time-consuming, produces high-quality, gold-standard datasets.

### Observing User Queries and System Responses
- Another method involves observing real user queries and system responses, and then having humans evaluate the results.

### Simplification for Experimental Purposes
- For our experiments, we will simplify the process:
  - Use an existing dataset.
  - Assign unique IDs to documents.
  - Use a hashing method to ensure document IDs remain consistent.

### Generating Unique Document IDs
- Importance of having unique IDs for each document to maintain integrity in the dataset.
- Challenges with assigning IDs and ensuring they remain consistent despite updates.
- Using a combination of document content and MD5 hashing to generate stable IDs.
```phyton
n = len(documents)
for i in range(n):
    documents[i] = i
```
- Generate id based on content
```phyton
import hashlib

def generate_document_id(doc):
    # combined = f"{doc['course']}-{doc['question']}"
    combined = f"{doc['course']}-{doc['question']}-{doc['text'][:10]}"
    hash_object = hashlib.md5(combined.encode())
    hash_hex = hash_object.hexdigest()
    document_id = hash_hex[:8]
    return document_id
```
```python
for doc in documents:
    doc['id'] = generate_document_id(doc)
```

### Saving the Dataset
- Exporting the dataset to JSON format for further use.
- Ensuring readability by adding indentation to the JSON file.

### Generating User Queries with LLMs
- Using a language model to generate user queries based on FAQ records.
- Crafting a prompt to simulate a student asking questions based on the provided FAQ record.
```python
prompt_template = """
You emulate a student who's taking our course.
Formulate 5 questions this student might ask based on a FAQ record. The record
should contain the answer to the questions, and the questions should be complete and not too short.
If possible, use as fewer words as possible from the record. 

The record:

section: {section}
question: {question}
answer: {text}

Provide the output in parsable JSON without using code blocks:

["question1", "question2", ..., "question5"]
""".strip()
```
- Ensuring the generated questions are complete, relevant, and do not use too many exact words from the record.

### Implementing the LLM for Query Generation
- Using the prompt to generate five user questions for each FAQ record.
- Configuring the language model client and executing the query generation process.
```python
from openai import OpenAI
client = OpenAI()
```
```python
prompt = prompt_template.format(**doc)

response = client.chat.completions.create(
        model='gpt-4o',
        messages=[{"role": "user", "content": prompt}]
    )

json_response = response.choices[0].message.content
json.loads(json_response)
```
Do for all:
```python
def generate_questions(doc):
    prompt = prompt_template.format(**doc)

    response = client.chat.completions.create(
        model='gpt-4o',
        messages=[{"role": "user", "content": prompt}]
    )

    json_response = response.choices[0].message.content
    return json_response
```
```python
from tqdm.auto import tqdm
```
```python
results = {}
```
### Handling Duplicate Questions
- Addressing the issue of duplicate questions in the dataset.
```python
for doc in tqdm(documents): 
    doc_id = doc['id']
    if doc_id in results: # acts like a hash
        continue

    questions = generate_questions(doc)
    results[doc_id] = questions
```
- Merging duplicates or differentiating them by including parts of the answer in the hash.

### Finalizing the Dataset
- Final steps in processing the dataset.
- Ensuring all document IDs are unique and correctly assigned.
- Saving the processed dataset to JSON format for use in retrieval experiments.
```python
parsed_resulst = {}

for doc_id, json_questions in results.items():
    parsed_resulst[doc_id] = json.loads(json_questions)
```
```python
doc_index = {d['id']: d for d in documents}
```
```python
final_results = []

for doc_id, questions in parsed_resulst.items():
    course = doc_index[doc_id]['course']
    for q in questions:
        final_results.append((q, course, doc_id))
```
- Save to csv
```python
import pandas as pd
```
```python
df = pd.DataFrame(final_results, columns=['question', 'course', 'document'])
```
```python
df.to_csv('ground-truth-data.csv', index=False)
```
### Conclusion
- The importance of a well-constructed ground truth dataset for evaluating retrieval systems.
- Simplifications made for the experiment and the methods used to ensure data integrity and relevance.
</details>
<details>
  
  <summary id="lecture-6"> Evaluation of Text Retrieval Techniques for RAG ou</summary>
We are going to use the data we created in the previous lecture to evaluate text search results. In particular, we will take the document, the ground truth dataset, and use it to evaluate our search queries.

## Ground Truth Dataset
We generated this ground truth dataset by taking each record in our FAQ and asking OpenAI's GPT-4o to generate five questions based on this record. This dataset contains:
- A question
- The course for which the question is relevant
- The relevant document

## Evaluation Process
For each query in our ground truth dataset, we will:
1. Execute the query.
2. Check if the relevant document is returned.
3. Based on this, compute different metrics.

## Metrics
Today, we will focus on two key metrics:
1. **Hit Rate**
2. **Mean Reciprocal Rank (MRR)**

### Hit Rate
Hit rate tells us if we are able to retrieve the relevant document among the top five results. Specifically, if the relevant document is in the top five results, the hit rate is considered a success for that query.

### Mean Reciprocal Rank (MRR)
MRR not only tells us if we retrieved the relevant document but also how good the ranking is. The higher the ranking of the relevant document, the better. MRR is calculated as follows:
- For each query, if the relevant document is found at position \( k \), the reciprocal rank is \( \frac{1}{k} \).
- If the document is not found, the reciprocal rank is 0.
- The MRR is the average of these reciprocal ranks across all queries.

## Implementation

### Loading Data
We load the ground truth dataset and convert it into a dictionary for easy manipulation.

```python
# Reading the dataset
df_ground_truth = pd.read_csv('ground-truth-data.csv')  # Load your ground truth dataset here
ground_truth = df_ground_truth.to_dict('records')
```
## Iterating Over Queries
We iterate over each query in the ground truth data and execute it using our search function. We then check if the relevant document is among the results.
```python
# Iterate over each query
for query in ground_truth:
    results = search_function(query['question'], query['course'])
    is_relevant = query['document_id'] in [doc['id'] for doc in results]
    # Store the relevance and rank
```
## Calculating Metrics
We calculate the hit-rate and Mean Reciprocal Rank (MRR) based on the results from our search function.
```python
# Calculate Hit Rate
def hit_rate(relevance_total):
    cnt = 0

    for line in relevance_total:
        if True in line:
            cnt = cnt + 1

    return cnt / len(relevance_total)

# Calculate MRR
def mrr(relevance_total):
    total_score = 0.0

    for line in relevance_total:
        for rank in range(len(line)):
            if line[rank] == True:
                total_score = total_score + 1 / (rank + 1)

    return total_score / len(relevance_total)
```
## Results
Finally, we evaluate the performance of our search system based on these metrics and discuss the results.
```python
def evaluate(ground_truth, search_function):
    relevance_total = []

    for q in tqdm(ground_truth):
        doc_id = q['document']
        results = search_function(q)
        relevance = [d['id'] == doc_id for d in results]
        relevance_total.append(relevance)

    return {
        'hit_rate': hit_rate(relevance_total),
        'mrr': mrr(relevance_total),
    }
```
```python
evaluate(ground_truth, question_vector_knn)
```
## Conclusion
In this session, we successfully evaluated our text search system using the ground truth dataset. We used two metrics, Hit Rate and MRR, to measure the effectiveness and ranking quality of our search results. This process helps us understand how well our system retrieves relevant documents and ensures that the most pertinent information is presented to the user.
</details>
<details>
  
  <summary id="lecture-7"> Evaluating Vector Retrival</summary>

</details>
