## Table of Contents
- [Introduction to monitoring answer quality](#lecture-1)
- [Evaluation and Monitoring in LLMs](#lecture-2)
- [](#lecture-3)
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

**Time:** 364.84 - 4.84
- **Focus:** Offline evaluation including cosine similarity and LLM as a judge.

### Cosine Similarity

**Time:** 374.919 - 4.321
- **Definition:** Measure how close the generated answer is to the expected answer.
- **Process:**
  - Create a test dataset with Q&A pairs.
  - Use LLM to generate answers for the questions.
  - Compute cosine similarity between original and generated answers.

### LLM as a Judge

**Time:** 501.8 - 4.839
- **Process:**
  - Ask the LLM to judge the similarity between the original and generated answers.
  - Alternatively, ask the LLM to judge how well the generated answer addresses the question directly.

## Conclusion

**Time:** 555.16 - 7.799
- **Next Steps:** In the next video, we will delve deeper into the offline evaluation of RAG systems and compute these metrics.
- **Sign-off:** See you soon.

  
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

<details>
  
  <summary id="lecture-8"> </summary>
  
</details>

<details>
  
  <summary id="lecture-9"> </summary>
  
</details>


