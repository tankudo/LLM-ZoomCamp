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

### Package Installation

- **46.199 - 6.241**: Install Sentence Transformers package for embedding generation.

### Sentence Transformers Overview

- **49.199 - 5.601**: Utilize pre-trained models to generate embeddings efficiently.
- **52.44 - 5.36**: Example usage and inference with Sentence Transformers.

### Embedding Generation

- **54.8 - 6.239**: Create embeddings for documents using selected models.
- **57.8 - 5.88**: Store embeddings alongside original data for indexing.

### Elasticsearch Setup

- **61.039 - 2.641**: Ensure Elasticsearch Docker container is running locally.
- **64.64 - 6.76**: Connect to Elasticsearch instance from Python environment.

### Indexing

- **68.439 - 5.481**: Define mappings to structure data in Elasticsearch.
- **71.4 - 4.16**: Map fields and data types for efficient storage and retrieval.

### Mapping Creation

- **73.92 - 3.8**: Define schema-like mapping for Elasticsearch indexing.
- **75.56 - 4.32**: Specify fields, data types, and embedding dimensions.

### Connection Verification

- **77.72 - 4.68**: Test Elasticsearch connection to ensure successful setup.

### Conclusion

- **79.88 - 4.199**: Architecture setup and initial data preparation complete.
- **82.4 - 4.8**: Ready to proceed with indexing and semantic search implementation.




  
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
