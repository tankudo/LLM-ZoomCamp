
## Basics of Vector Database
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

**Thank you for joining! Happy learning!**

