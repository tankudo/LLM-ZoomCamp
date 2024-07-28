## Table of Contents
- Introduction to monitoring answer quality[](#lecture-1)
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


