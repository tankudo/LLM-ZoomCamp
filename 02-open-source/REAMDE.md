# Course Module

## Table of Contents
- [Open Source LLMs](#lecture-1)
- [Using SaturnCloud for GPU Notebooks](#lecture-2)
- [HuggingFace and Google FLAN T5](#lecture-3)
- [Phi 3 Mini](#lecture-4)
- [Mistral-7B and HuggingFace Hub Authentication](#lecture-5)
- [Exploring Open Source LLMs](#lecture-6)
- [](#lecture-7)
- [](#lecture-8)
- [](#lecture-9)
---

<details>
  <summary id="lecture-1"> Open Source LLMs</summary>

## Introduction
- **Module Focus:** In this module, will explored  open-source LLMs - alternatives to OpenAI.

## Open Source LLMs
- **Control:** These models can be executed on our own hardware, giving us full control.

## Model Options
- **Diverse Models:** There are numerous open-source models available, including well-known ones and many others.
- **Examples:**
  - **LLaMA:** Developed by Meta (Facebook).
  - **Flan-T5 or T5:** Developed by Google.
  - **Mistral:** Another notable model with an accessible API.

## Practical Implementation
- **Running Models:** We will cover how to run these models, typically requiring a GPU.
- **Environment Setup:** Discussion on setting up the proper environment for running these models.

## Workflow Adaptation
- **Replacing Components:** We'll explore how to replace parts of the workflow with different models.
- **Focus:** Setting up environments and utilizing different LLMs effectively.

## Hands-On Practice
- **Flan-T5 Example:** We'll delve into specific models like Flan-T5.
- **Model Access:** How to access models from platforms like Hugging Face Hub.

## Conclusion
- **Exciting Module:** This module promises to be interesting and informative.
- **Next Steps:** In the next module, we will see how to get a notebook with a GPU to run these models.

See you soon!


</details>

<details>
  <summary id="lecture-2"> Using SaturnCloud for GPU Notebooks</summary>

## Introduction
- **Objective:** Ler's use Saturn Cloud to get a notebook with a GPU, essential for running open-source LLMs.

## Why GPU?
- **GPU Requirement:** Most open-source LLMs require a GPU for efficient performance.
- **Saturn Cloud:** One of the possible options for setting up a GPU environment.

## Alternative Options
- **Google Colab**
- **Google Cloud Platform**
- **AWS SageMaker**
- **Other Cloud Services**

## Saturn Cloud Setup
1. **Login:**
   - Log in to Saturn Cloud if you already have an account.
   - If you do not have an account, you can sign up and get extra GPU hours by filling out a form on the course management platform or requesting a technical demo.

2. **Creating a Notebook:**
   - **Log in:** Log in using your credentials.
   - **Create Notebook:** Go to the notebook creation section in Saturn Cloud.
   - **Secrets:** Add necessary tokens like OpenAI or Hugging Face tokens in the secrets section.
   - **SSH Keys:** Set up Git access by adding your public SSH key to GitHub for easy access to repositories.

3. **Configuring the Environment:**
   - **Notebook Server:** Create a new notebook server.
   - **Starter Notebook:** Use the provided starter notebook for the course.
   - **Environment Setup:** Choose the appropriate Python version and environment. Install necessary packages like `transformers`, `accelerate`, and `bitsandbytes`.

4. **Running the Notebook:**
   - **Install Packages:** Install the required libraries and clone the course repository.
   - **Start Notebook:** Launch the Jupyter Notebook or JupyterLab interface.
   - **Execute Code:** Run the starter code provided in the notebook.

5. **Environment Verification:**
   - **Check GPU:** Verify the presence of a GPU using the `nvidia-smi` command.
   - **Ready to Use:** Ensure the environment is correctly set up and ready for running LLMs.

## Next Steps
- **Upcoming Videos:** Learn how to use this environment to run Google's Flan-T5 model, an open-source LLM.
- **Hands-On Practice:** Continue with practical exercises to solidify your understanding.

</details>

<details>
  <summary id="lecture-3">  HuggingFace and Google FLAN T5</summary>

## Introduction  
- The first open-source LLM is from Google, accessed via the Hugging Face library.
- Brief overview of Hugging Face:
  - A repository for models where various companies, including Google, publish their models.
  - Known for the Hugging Face Hub, a central place for finding and downloading models.

## About Hugging Face
- Hugging Face hosts models which can be uploaded and downloaded.
- Trending models can be found on this website.
- Models are not limited to LLMs; various types of models are available.
- Hugging Face Hub is a repository familiar to GitHub but for models.

## Using Hugging Face Hub
- We will use the Google Flan-T5 XL model.
- The Hugging Face library, 'Transformers,' is used to fetch and run models.
- Here is also good demonstration of finding and accessing model documentation and examples on Hugging Face Hub.
```python
from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xl")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xl", device_map="auto") # actual model
```

## Downloading Model Files
- Hugging Face Hub helps fetch necessary model files via the Transformers library.
- Importance of setting the appropriate file download location to manage space.

## Setting up the Environment
- Importance of sufficient space for downloading large model files.
- Execution and testing of the setup using a sample model.
```python
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xl", cache_dir='/run/cache/huggingface')
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xl", cache_dir='/run/cache/huggingface', device_map="auto")
```

## Importing and Using the Model
- Explanation of importing necessary components: the generator and tokenizer.
```python
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
outputs = model.generate(input_ids, )
result = tokenizer.decode(outputs[0])
```

##  Generating Responses
- Using parameters like `max_length` to control the length of generated responses.
```python
outputs = model.generate(
        input_ids,
        max_length=generate_params.get("max_length", 100),
        num_beams=generate_params.get("num_beams", 5),
        do_sample=generate_params.get("do_sample", False),
        temperature=generate_params.get("temperature", 1.0),
        top_k=generate_params.get("top_k", 50),
        top_p=generate_params.get("top_p", 0.95),
    )
```

## Summary
- Transition from using OpenAI to a local GPU instance.
- Benefits of running models locally.
- Introduction to exploring other models, such as Microsoft's T3, in future videos.

---

**Links:**
- [Hugging Face Hub](https://huggingface.co/)
- [Google Flan T5 XL Model Documentation](https://huggingface.co/google/flan-t5-xl)

</details>

<details>
  <summary id="lecture-4"> Phi 3 Mini</summary>

## Using Another Open Source Model on Hugging Face Hub

### Introduction
- **Objective**: Explore another open-source model available on Hugging Face Hub.

### New Model: Microsoft Phi-3
- **Model Name**: Phi-3 from Microsoft.
- **Characteristics**: 
  - Relatively new model.
  - Different sizes available (e.g., Mini, Small).
  - Suitable for GPUs with around 15-16 GB memory.

### Steps to Explore Phi-3 Model
1. **Search and Access**:
   - Look up the model on Hugging Face Hub.
   - Review the model's page and documentation.

2. **GPU Specification Check**:
   - Open a terminal on the machine with GPU.
   - Run `nvidia-smi` to check GPU specs (memory and type).

3. **Prepare the Environment**:
   - Stop any previous model instances to free up GPU memory.
   - Duplicate the existing Jupyter notebook for the new model.

4. **Setup Phi-3 Model**:
   - Rename the new notebook appropriately.
   - Create a pipeline to combine the model and tokenizer.
   - Set the random seed for reproducibility.
```python  
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

torch.random.manual_seed(0)
```

5. **Download and Initialize Model**:
   - Download the Phi-3 model and its tokenizer.
   - Note the size comparison with Flan-T5 (Phi-3 is around 7 GB).
  ```python
  model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-128k-instruct", 
    device_map="cuda", 
    torch_dtype="auto", 
    trust_remote_code=True, 
)
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct")
```

### Model Usage
- **Interface**:
  - Slightly different from Flan-T5.
  - Supports a chatbot-like interface.
- **Execution**:
  - Create and run the pipeline.
```python
  pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,

)
```

  - Pass messages and parameters to the model.
```python
search_results = search(query)
prompt = build_prompt(query, search_results)
messages = [
        {"role": "user", "content": prompt},
    ]

generation_args = {
        "max_new_tokens": 500,
        "return_full_text": False,
        "temperature": 0.0,
        "do_sample": False,
    }
```

  - Observe the output and adjust if necessary.
```python
output = pipe(messages, **generation_args)
```

### Conclusion
- **Model Performance**:
  - Output seems coherent.
  - Adjustments like stripping spaces may be needed.
- **Next Steps**:
  - Explore another model from Mistral in the next video.

</details>

<details>
  <summary id="lecture-5"> Mistral-7B and HuggingFace Hub Authentication</summary>

## Overview
This session focuses on implementing the Mistral 7B model using Hugging Face's platform.

## Steps to Access and Use the Model

1. **Model Discovery**
   - Copy the model name, search on Google, and locate the Hugging Face model card page.

2. **Model Information**
   - Unlike other models, the Mistral 7B model page lacks example code snippets.
   - Possible solutions:
     - Search in Goodle, ask ChatGPT but also, HuggingFace turorial has information about Mistral.

3. **Setting Up the Model**
   - Create a model and tokenizer instance similarly to previous models.
   - Use Hugging Face's API to load the model and tokenizer.

4. **Handling Authorization**
   - This model requires acceptance of an agreement on Hugging Face.
   - Sign in to Hugging Face, go to settings, and create an access token.
   - Use the token in your code to authenticate and access the model.

5. **Code Implementation**
   - Import necessary libraries and use the token for Hugging Face login:
     ```python
     from huggingface_hub import login
     login(token=os.environ['HF_TOKEN'])
     ```

6. **Model and Tokenizer Initialization**
   - Initialize the model and tokenizer:
     ```python
     from transformers import AutoModelForCausalLM, AutoTokenizer

     model = AutoModelForCausalLM.from_pretrained("mistral/Mistral-7b-v0.1", device_map="auto", load_in_4bit=True)
     tokenizer = AutoTokenizer.from_pretrained("mistral/Mistral-7b-v0.1", padding_side="left")
     ```

## Experimentation and Troubleshooting

1. **Memory and Resource Management**
   - Ensure sufficient memory is available by stopping other running models.
   - Monitor GPU memory usage.

2. **Handling Large Prompts**
   - The Mistral 7B model may not handle large prompts efficiently.
   - Experiment with reducing prompt size and adjusting model parameters.

3. **Error Handling**
   - Common issues include unauthorized access and incomplete prompts.
   - Verify token validity and prompt formatting.

4. **Using Pipelines**
   - Utilize Hugging Face pipelines for a more streamlined approach:
     ```python
     from transformers import pipeline

     generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
     ```

## Saving and Serving Models

1. **Local Model Storage**
   - Download and save models locally to avoid repeated authentication:
     ```python
     model.save_pretrained("local_model_directory")
     tokenizer.save_pretrained("local_model_directory")
     ```

2. **Loading Saved Models**
   - Load the saved model and tokenizer from the local directory:
     ```python
     model = AutoModelForCausalLM.from_pretrained("local_model_directory")
     tokenizer = AutoTokenizer.from_pretrained("local_model_directory")
     ```


## Conclusion

- The Mistral 7B model offers significant capabilities but requires careful prompt handling and parameter tuning.
- Effective use involves proper setup, troubleshooting, and efficient resource management.
- Saving models locally can enhance deployment flexibility and efficiency.

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

