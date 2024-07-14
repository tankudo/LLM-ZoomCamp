## Table of Contents
- [Introduction to Vector Search](#lecture-1)
- [](#lecture-2)
- [](#lecture-3)
- [](#lecture-4)
- [](#lecture-5)
- [](#lecture-6)
- [](#lecture-7)
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
```
### Connection Verification

- Test Elasticsearch connection to ensure successful setup.

### Conclusion

- Architecture setup and initial data preparation complete.
- Ready to proceed with indexing and semantic search implementation.




  
</details>
<details>
  
  <summary id="lecture-3"> </summary>
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
