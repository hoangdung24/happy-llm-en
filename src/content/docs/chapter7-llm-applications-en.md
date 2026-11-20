---
title: "Chapter 7: LLM Applications"
---

The previous chapters introduced the foundational knowledge of large models, training methods, and fine-tuning techniques. This chapter focuses on the key technologies and frameworks for applying large models in practice. It covers core topics such as large model evaluation, RAG (Retrieval-Augmented Generation), and Agent (intelligent agent). These topics help readers understand the practical application scenarios and implementation methods of large models.

## 7.1 Evaluation of LLMs

In recent years, the field of artificial intelligence has developed fast. Large-scale pre-trained language models (called large models) have become a core force of technological progress. These large models show strong capabilities in natural language processing and other tasks. However, to measure the performance of a large model accurately, one must use scientific and reasonable evaluation.

What is large model evaluation? Large model evaluation quantifies and compares the performance of large models across different tasks. It uses many standardized methods and datasets. These evaluations cover the accuracy of the model on specific tasks. They also cover the generalization ability, inference speed, and resource consumption of the model. Through evaluation, we can understand the actual performance of large models and their application potential in the real world.

The development of large models is costly. It needs substantial computational resources and data, so evaluation is essential to make sure that the model has practical value. First, evaluation shows the performance of the model across various tasks. It helps researchers and enterprises judge the applicability and reliability of the model. Second, evaluation can expose potential weaknesses of the model, such as bias and robustness issues. It provides a basis for further optimization and improvement. In addition, fair and open evaluation provides a common standard for academia and industry. It promotes the exchange and advancement of technology.

### 7.1.1 Evaluation Datasets for LLMs

In the evaluation of large models, standardization of the evaluation sets is essential. Current mainstream evaluation sets for large models assess the following aspects. Each evaluation set has its unique purpose and typical application scenarios:

1. General Evaluation Sets:
   - MMLU (Massive Multitask Language Understanding). MMLU evaluates the understanding ability of the model across a variety of tasks. It covers various disciplines and knowledge domains. The tasks include history, mathematics, physics, biology, and law. The evaluation examines the knowledge reserves and the language comprehension ability of the model across different disciplines.

2. Tool-Use Evaluation Sets:
   - BFCL V2. This set evaluates the performance of the model in complex tool-use tasks. It focuses on correctness and efficiency during multi-step operations. These tasks usually involve interaction with databases or execution of specific instructions. They simulate real-world tool-use scenarios.

3. Mathematics Evaluation Sets:
   - GSM8K. GSM8K is a dataset of elementary school math problems. It tests the mathematical reasoning and logical analysis abilities of the model. Specific tasks include arithmetic operations, simple equation solving, and numerical reasoning. The problems in GSM8K seem simple. However, the model must understand the problem semantics and perform correct mathematical operations. This shows the dual challenge of logical reasoning and language understanding.
   - MATH. The MATH dataset is used to test the model's performance on more complex mathematical problems, including algebra and geometry.

4. Reasoning Evaluation Sets:
   - ARC Challenge. ARC Challenge evaluates the performance of the model on scientific reasoning tasks. It focuses on the answers to commonsense and scientific questions. Typical application scenarios include solving science exam questions and developing encyclopedia Q&A systems.
   - GPQA. This set evaluates the ability of the model to answer open-ended questions in zero-shot conditions. It usually applies to customer service chatbots and knowledge Q&A systems. It helps the model give reasonable answers when specific domain data is lacking.
   - HellaSwag. This set evaluates the ability of the model to choose the most logically consistent answer in complex contexts. It is suitable for scenarios that need high-level understanding and reasoning, such as story continuation and dialogue generation.

5. Long-Text Understanding Evaluation Sets:
   - InfiniteBench/En.MC. This set evaluates the ability of the model to handle long-text reading comprehension. It focuses on the understanding of scientific literature. It is suitable for application scenarios such as automatic summarization of academic literature and long-form report analysis.
   - NIH/Multi-needle. This set tests the understanding and summarization abilities of the model in multi-sample long-document environments. It applies to scenarios that need the processing of massive information, such as interpreting government reports and analyzing long internal corporate documents.

6. Multilingual Evaluation Sets:
   - MGSM. This set assesses the ability of the model to solve mathematical problems in different languages. It examines the multilingual adaptability of the model. It is especially suitable for math education in international environments and for cross-language technical support scenarios.

The diversity of these evaluation sets helps us assess the performance of large models across different tasks and application scenarios. It makes sure that models remain efficient and accurate when they handle diverse tasks. For example, in the MMLU evaluation, some large models perform excellently on subject tasks such as history and physics. This result shows a deep understanding of multi-domain knowledge. In the GSM8K math evaluation, the latest large models perform close to or even above some human baselines in arithmetic and equation solving. This result shows potential in complex mathematical reasoning tasks. These actual evaluation results show the progress and application potential of the models in various complex tasks.


### 7.1.2 Mainstream Evaluation Leaderboards

The evaluation of large models is not limited to specific datasets. Many institutions also publish model leaderboards based on evaluation results. These leaderboards provide an important reference for academia and industry. They help these groups understand the current cutting-edge technologies and models. The following are some mainstream evaluation leaderboards:

#### Open LLM Leaderboard

The Open LLM Leaderboard is an open leaderboard from Hugging Face. It aggregates the evaluation results of multiple open-source large models. It helps users understand the performance of different models on various tasks. The leaderboard evaluates model performance through multiple standardized test sets. Continuous updates reflect the latest technological progress. The result is a high-value comparison reference for researchers and developers, as shown in Figure 7.1.

<div align='center'>
    <img src="/images/7-images/7-1-Open-LLM-Leaderboard.webp" alt="Figure 7.1 Open LLM Leaderboard" width="1119" height="1071" loading="lazy" decoding="async" />
    <p>Figure 7.1 Open LLM Leaderboard</p>
</div>

#### Lmsys Chatbot Arena Leaderboard

The chatbot evaluation leaderboard from LMSYS shows the abilities of various large models in dialogue tasks through multi-dimensional evaluation. The leaderboard evaluates dialogue quality through real user-model interactions. It focuses on the natural language generation ability of the model, its contextual understanding ability, and user satisfaction. It is currently an important tool for evaluating chatbot performance, as shown in Figure 7.2.

<div align='center'>
    <img src="/images/7-images/7-1-lmsys-Chatbot-Arena-Leaderboard.webp" alt="Figure 7.2 Lmsys Chatbot Arena Leaderboard" width="1119" height="1266" loading="lazy" decoding="async" />
    <p>Figure 7.2 Lmsys Chatbot Arena Leaderboard</p>
</div>

#### OpenCompass

OpenCompass is a domestic (Chinese) evaluation leaderboard. It evaluates the performance of large models across multiple languages and tasks. It provides a reference for applications specific to the Chinese market. The leaderboard combines tests of Chinese language understanding and multilingual capabilities to meet localization needs. It gives particular attention to the accuracy, robustness, and adaptability of large models in Chinese contexts. Domestic enterprises and researchers use it as an important reference for model selection.

<div align='center'>
    <img src="/images/7-images/7-1-opencompass.webp" alt="Figure 7.3 OpenCompass" width="923" height="984" loading="lazy" decoding="async" />
    <p>Figure 7.3 OpenCompass</p>
</div>

### 7.1.3 Domain-Specific Evaluation Leaderboards

In addition, there are large model evaluation leaderboards for specific tasks in different domains, as shown in Figure 7.4. These leaderboards focus on specific application domains. They help users understand the capabilities of large models in a particular vertical field:

- Finance Leaderboard: This leaderboard uses the CFBenchmark evaluation set. It assesses the capabilities of large models in multiple fundamental tasks. These tasks include financial natural language processing, financial predictive computation, financial analysis, and safety inspection. Tongji University provides it together with the Shanghai Artificial Intelligence Laboratory and Dongfang Finance.

- Safety Leaderboard: This leaderboard uses the Flames evaluation set. It assesses the resistance of large models across five major dimensions. These dimensions include fairness, safety, data protection, and legality. It helps users gain deep insight into the performance of the model in terms of safety. The Shanghai Artificial Intelligence Laboratory and Fudan University provide it.

- General Knowledge Leaderboard: This leaderboard uses the BotChat evaluation set. It assesses how well large language models generate everyday multi-turn dialogues. It judges whether the model has a human-like level in conversation. The Shanghai Artificial Intelligence Laboratory provides it.

- Legal Leaderboard: This leaderboard uses the LawBench evaluation set. It assesses the understanding, reasoning, and application abilities of the model in the legal domain. It covers tasks such as answering legal questions, text generation, and legal case analysis. Nanjing University provides it.

- Medical Leaderboard: This leaderboard uses the MedBench evaluation set. It assesses the performance of large language models in medical knowledge Q&A and in the understanding of safety and ethics. The Shanghai Artificial Intelligence Laboratory provides it.

<div align='center'>
    <img src="/images/7-images/7-1-vertical-domain-leaderboard.webp" alt="Figure 7.4 Vertical Domain Leaderboards" width="1070" height="813" loading="lazy" decoding="async" />
    <p>Figure 7.4 Vertical Domain Leaderboards</p>
</div>


## 7.2 RAG 

### 7.2.1 Basic Principles of RAG

Large language models (LLMs) have strong language understanding and generation capabilities. However, they face several challenges when they generate content. For example, LLMs sometimes generate inaccurate or misleading content. This is known as the large model "hallucination". In addition, the training data that the model relies on can be outdated. When it faces the latest information, it is difficult to guarantee the accuracy and timeliness of the generated results. LLMs also process specialized domain knowledge inefficiently. They cannot deeply understand complex domain knowledge. Therefore, how to improve the generation quality and efficiency of large models has become an important research direction.

Against this background, Retrieval-Augmented Generation (RAG) technology emerged. It became a major innovation trend in the AI field. Before it generates an answer, RAG retrieves relevant information from an external large-scale document database. It integrates this information into the generation process. This guides and optimizes the output of the language model. This process greatly improves the accuracy and relevance of content generation. It also makes the generated content better meet real-time requirements.

The core principle of RAG combines "retrieval" with "generation". A user poses a query. The system then uses the retrieval module to find text fragments related to the question. It passes these fragments to the language model as additional information. The model generates a more precise and reliable answer based on them. In this way, RAG effectively alleviates the "hallucination" problem of large language models. The generated content is grounded in real documents, so the answers are more traceable and credible. Meanwhile, up-to-date information sources are introduced. RAG technology therefore greatly accelerates knowledge updates. The system can absorb and reflect the latest domain developments in time.

### 7.2.2 Building a RAG Framework

Next, I will guide you step by step to implement a simple RAG model. This model is a simplified version based on RAG. We call it Tiny-RAG. Tiny-RAG only keeps the core functions of RAG, namely retrieval and generation. The goal is to help everyone better understand the principles and implementation of the RAG model.

#### Step 1: Introduction to the RAG Pipeline

RAG first retrieves relevant information from a broad document database. Then the language model generates an answer. RAG uses this information to guide the generation process. This greatly improves the accuracy and relevance of the content. RAG effectively alleviates the hallucination problem, speeds up knowledge updates, and enhances the traceability of content generation. It makes large language models more practical and trustworthy in real-world applications.

What are the basic components of RAG?

- Vectorization module: used to vectorize document chunks.
- Document loading and splitting module: used to load documents and split them into document chunks.
- Database: stores the document chunks and their corresponding vector representations.
- Retrieval module: retrieves relevant document chunks according to the Query (question).
- Large model module: answers the user's question based on the retrieved documents.

These are all the module contents of TinyRAG, as shown in Figure 7.5.

<div align='center'>
    <img src="/images/7-images/7-2-tinyrag.webp" alt="Figure 7.5 TinyRAG Project Structure" width="1600" height="739" loading="lazy" decoding="async" />
    <p>Figure 7.5 TinyRAG Project Structure</p>
</div>

Next, let us review what the RAG pipeline looks like:

- Indexing: split the document library into shorter chunks and build a vector index using an encoder.
- Retrieval: retrieve relevant document chunks according to the similarity between the question and the chunks.
- Generation: generate the answer to the question conditioned on the retrieved context.

The flowchart is shown in Figure 7.6 below. Image source: [Retrieval-Augmented Generation for Large Language Models: A Survey](https://arxiv.org/pdf/2312.10997.pdf)

<div align='center'>
    <img src="/images/7-images/7-2-rag.webp" alt="Figure 7.6 RAG Flowchart" width="1561" height="910" loading="lazy" decoding="async" />
    <p>Figure 7.6 RAG Flowchart</p>
</div>

#### Step 2: Document Loading and Splitting

Next, we implement a class for document loading and splitting. This class mainly loads documents and splits them into document chunks.

Documents can be text content such as articles, books, conversations, or code. Examples are pdf files, md files, and txt files. The complete code is in the [RAG/utils.py](./RAG/utils.py) file. This code supports loading files of types such as pdf, md, and txt. You only need to write the corresponding functions.

```python
def read_file_content(cls, file_path: str):
    # Choose the reading method based on the file extension
    if file_path.endswith('.pdf'):
        return cls.read_pdf(file_path)
    elif file_path.endswith('.md'):
        return cls.read_markdown(file_path)
    elif file_path.endswith('.txt'):
        return cls.read_text(file_path)
    else:
        raise ValueError("Unsupported file type")
```

After the code reads the document, it must split the document. We can set a maximum Token length and split the document according to this maximum length. When you split the document, use sentences as the unit. Split roughly by `\n`. Make sure that chunks have some overlapping content. Overlap improves retrieval accuracy.

```python
def get_chunk(cls, text: str, max_token_len: int = 600, cover_content: int = 150):
    chunk_text = []

    curr_len = 0
    curr_chunk = ''

    token_len = max_token_len - cover_content
    lines = text.splitlines()  # Assume the text is split into lines by newline characters

    for line in lines:
        # Keep spaces, only remove leading and trailing whitespace
        line = line.strip()
        line_len = len(enc.encode(line))
        
        if line_len > max_token_len:
            # If a single line exceeds the limit, split it into multiple chunks
            # First save the current chunk (if it has content)
            if curr_chunk:
                chunk_text.append(curr_chunk)
                curr_chunk = ''
                curr_len = 0
            
            # Split the long line by token length
            line_tokens = enc.encode(line)
            num_chunks = (len(line_tokens) + token_len - 1) // token_len
            
            for i in range(num_chunks):
                start_token = i * token_len
                end_token = min(start_token + token_len, len(line_tokens))
                
                # Decode the token fragments back to text
                chunk_tokens = line_tokens[start_token:end_token]
                chunk_part = enc.decode(chunk_tokens)
                
                # Add coverage content (except for the first chunk)
                if i > 0 and chunk_text:
                    prev_chunk = chunk_text[-1]
                    cover_part = prev_chunk[-cover_content:] if len(prev_chunk) > cover_content else prev_chunk
                    chunk_part = cover_part + chunk_part
                
                chunk_text.append(chunk_part)
            
            # Reset the current chunk state
            curr_chunk = ''
            curr_len = 0
            
        elif curr_len + line_len + 1 <= token_len:  # +1 for newline
            # The current line can be added to the current chunk
            if curr_chunk:
                curr_chunk += '\n'
                curr_len += 1
            curr_chunk += line
            curr_len += line_len
        else:
            # The current line cannot be added to the current chunk, start a new chunk
            if curr_chunk:
                chunk_text.append(curr_chunk)
            
            # Start a new chunk, add coverage content
            if chunk_text:
                prev_chunk = chunk_text[-1]
                cover_part = prev_chunk[-cover_content:] if len(prev_chunk) > cover_content else prev_chunk
                curr_chunk = cover_part + '\n' + line
                curr_len = len(enc.encode(cover_part)) + 1 + line_len
            else:
                curr_chunk = line
                curr_len = line_len

    # Add the last chunk (if it has content)
    if curr_chunk:
        chunk_text.append(curr_chunk)

    return chunk_text

```

#### Step 3: Vectorization

First, let us implement a vectorization class by ourselves. This is the foundation of the RAG architecture. The vectorization class mainly vectorizes document chunks. It maps a piece of text into a vector.

First, we set up a `BaseEmbeddings` base class. When we use other models, we only need to inherit from this base class and then make modifications on top of it. This approach facilitates code extension.

```python
class BaseEmbeddings:
    """
    Base class for embeddings
    """
    def __init__(self, path: str, is_api: bool) -> None:
        """
        Initialize the embeddings base class
        Args:
            path (str): Path to the model or data
            is_api (bool): Whether to use the API mode. True means using an online API service, False means using a local model
        """
        self.path = path
        self.is_api = is_api
    
    def get_embedding(self, text: str, model: str) -> List[float]:
        """
        Get the embedding vector representation of the text
        Args:
            text (str): Input text
            model (str): Name of the model to use
        Returns:
            List[float]: The embedding vector of the text
        Raises:
            NotImplementedError: This method needs to be implemented in subclasses
        """
        raise NotImplementedError
    
    @classmethod
    def cosine_similarity(cls, vector1: List[float], vector2: List[float]) -> float:
        """
        Compute the cosine similarity between two vectors
        Args:
            vector1 (List[float]): The first vector
            vector2 (List[float]): The second vector
        Returns:
            float: The cosine similarity of the two vectors, in the range [-1,1]
        """
        # Convert the input lists to numpy arrays with dtype float32
        v1 = np.array(vector1, dtype=np.float32)
        v2 = np.array(vector2, dtype=np.float32)

        # Check whether the vectors contain infinity or NaN values
        if not np.all(np.isfinite(v1)) or not np.all(np.isfinite(v2)):
            return 0.0

        # Compute the dot product of the vectors
        dot_product = np.dot(v1, v2)
        # Compute the norms (lengths) of the vectors
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        
        # Compute the denominator (the product of the two vector norms)
        magnitude = norm_v1 * norm_v2
        # Handle the special case where the denominator is 0
        if magnitude == 0:
            return 0.0
            
        # Return the cosine similarity
        return dot_product / magnitude
```

The `BaseEmbeddings` base class has two main methods: `get_embedding` and `cosine_similarity`. `get_embedding` obtains the vector representation of the text. `cosine_similarity` computes the cosine similarity between two vectors. When the class initializes, it sets the model path and whether the model is an API model. For example, OpenAI's Embedding API needs `self.is_api=True`.

Inheriting from the `BaseEmbeddings` class only requires implementation of the `get_embedding` method. The `cosine_similarity` method is inherited directly. This is the benefit of writing a base class.

```python
class OpenAIEmbedding(BaseEmbeddings):
    """
    class for OpenAI embeddings
    """
    def __init__(self, path: str = '', is_api: bool = True) -> None:
        super().__init__(path, is_api)
        if self.is_api:
            self.client = OpenAI()
            # Get the SiliconFlow API key from environment variables
            self.client.api_key = os.getenv("OPENAI_API_KEY")
            # Get the SiliconFlow base URL from environment variables
            self.client.base_url = os.getenv("OPENAI_BASE_URL")
    
    def get_embedding(self, text: str, model: str = "BAAI/bge-m3") -> List[float]:
        """
        Here the free embedding model BAAI/bge-m3 from SiliconFlow is used by default
        """
        if self.is_api:
            text = text.replace("\n", " ")
            return self.client.embeddings.create(input=[text], model=model).data[0].embedding
        else:
            raise NotImplementedError
```

> Note: Here we use by default the [SiliconFlow large model API service platform](https://cloud.siliconflow.cn/i/ybUFvmqK), which is accessible to users in China.


#### Step 4: Database and Vector Retrieval

After document splitting and loading of the Embedding model, we must design a vector database. The database stores the document chunks and their corresponding vector representations. We must also design a retrieval module. The module retrieves relevant document chunks based on the Query.

The functions of the vector database are:

- `persist`: persist and save the database.
- `load_vector`: load the database from local storage.
- `get_vector`: obtain the vector representation of documents.
- `query`: retrieve relevant document chunks according to the question.

The complete code is in the [/VectorBase.py](./RAG/VectorBase.py) file.

```python
class VectorStore:
    def __init__(self, document: List[str] = ['']) -> None:
        self.document = document

    def get_vector(self, EmbeddingModel: BaseEmbeddings) -> List[List[float]]:
        # Obtain the vector representation of documents
        pass

    def persist(self, path: str = 'storage'):
        # Persist and save the database
        pass

    def load_vector(self, path: str = 'storage'):
        # Load the database from local storage
        pass

    def query(self, query: str, EmbeddingModel: BaseEmbeddings, k: int = 1) -> List[str]:
        # Retrieve relevant document chunks according to the question
        pass
```

The `query` method vectorizes the question of the user. It then retrieves relevant document chunks in the database and returns the results.

```python
def query(self, query: str, EmbeddingModel: BaseEmbeddings, k: int = 1) -> List[str]:
    query_vector = EmbeddingModel.get_embedding(query)
    result = np.array([self.get_similarity(query_vector, vector) for vector in self.vectors])
    return np.array(self.document)[result.argsort()[-k:][::-1]].tolist()
```

#### Step 5: Large Model Module

Next is the large model module. It answers the question of the user based on the retrieved documents.

First, implement a base class. The base class makes extension to other models convenient.

```python
class BaseModel:
    def __init__(self, path: str = '') -> None:
        self.path = path

    def chat(self, prompt: str, history: List[dict], content: str) -> str:
        pass

    def load_model(self):
        pass
```

`BaseModel` contains two methods: `chat` and `load_model`. For open-source models that run locally, you must implement `load_model`. API models do not need it. Here we still use the SiliconFlow large model API service platform, which is accessible to users in China. An API service has a benefit: users do not need local computing resources. This greatly lowers the learning threshold for learners.

```python
from openai import OpenAI

class OpenAIChat(BaseModel):
    def __init__(self, model: str = "Qwen/Qwen2.5-32B-Instruct") -> None:
        self.model = model

    def chat(self, prompt: str, history: List[dict], content: str) -> str:
        client = OpenAI()
        client.api_key = os.getenv("OPENAI_API_KEY")   
        client.base_url = os.getenv("OPENAI_BASE_URL")
        history.append({'role': 'user', 'content': RAG_PROMPT_TEMPLATE.format(question=prompt, context=content)})
        response = client.chat.completions.create(
                model=self.model,
                messages=history,
                max_tokens=2048,
                temperature=0.1
            )
        return response.choices[0].message.content

```

Design a large model prompt dedicated to RAG, as follows:

```python
RAG_PROMPT_TEMPLATE="""
使用以上下文来回答用户的问题。如果你不知道答案，就说你不知道。总是使用中文回答。
问题: {question}
可参考的上下文：
···
{context}
···
如果给定的上下文无法让你做出回答，请回答数据库中没有这个内容，你不知道。
有用的回答:
"""
# (Gloss: "Use the above context to answer the user's question. If you don't know the answer, just say you don't know. Always answer in Chinese.
# Question: {question}
# Context that can be referenced:
# ...
# {context}
# ...
# If the given context does not allow you to answer, please reply that this content is not in the database and you don't know.
# Helpful answer:")
```

Now we can use the InternLM2 model to do RAG!

#### Step 6: Tiny-RAG Demo

Next, let us look at the Tiny-RAG Demo!

```python
from VectorBase import VectorStore
from utils import ReadFiles
from LLM import OpenAIChat
from Embeddings import OpenAIEmbedding

# Database has not been saved yet
docs = ReadFiles('./data').get_content(max_token_len=600, cover_content=150) # Get the content of all files under the data directory and split it
vector = VectorStore(docs)
embedding = OpenAIEmbedding() # Create the EmbeddingModel
vector.get_vector(EmbeddingModel=embedding)
vector.persist(path='storage') # Save the vectors and document content to the storage directory; next time you can directly load the local database

# vector.load_vector('./storage') # Load the local database

question = 'RAG的原理是什么？'  # (Gloss: "What is the principle of RAG?")

content = vector.query(question, EmbeddingModel=embedding, k=1)[0]
chat = OpenAIChat(model='Qwen/Qwen2.5-32B-Instruct')
print(chat.chat(question, [], content))
```

You can also load an already-processed database from local storage:

```python
from VectorBase import VectorStore
from utils import ReadFiles
from LLM import OpenAIChat
from Embeddings import OpenAIEmbedding

# After saving the database
vector = VectorStore()

vector.load_vector('./storage') # Load the local database

question = 'RAG的原理是什么？'  # (Gloss: "What is the principle of RAG?")

embedding = ZhipuEmbedding() # Create the EmbeddingModel

content = vector.query(question, EmbeddingModel=embedding, k=1)[0]
chat = OpenAIChat(model='Qwen/Qwen2.5-32B-Instruct')
print(chat.chat(question, [], content))
```

> Note: All the code in Section 7.2 can be found in [Happy-LLM Chapter7 RAG](https://github.com/datawhalechina/happy-llm/tree/main/docs/chapter7/RAG).
## 7.3 Agent

### 7.3.1 What is an LLM Agent?

Simply put, a large model Agent is a system with an LLM as its core "brain". It has the abilities of autonomous planning, memory, and tool use. It no longer only responds passively to user prompts (Prompt). Instead, it can do the following:

1. Goal Understanding: receive a relatively complex or high-level goal (e.g., "Help me plan a weekend trip itinerary to Beijing and book flights and hotels").
2. Planning: break the big goal down into a series of executable small steps. Examples are "search Beijing attractions", "check the weather", "compare flight prices", "find suitable hotels", and "call the booking API".
3. Memory: have short-term memory and long-term memory. Short-term memory remembers the context of the current task. Long-term memory learns from and retrieves information from past interactions or external knowledge bases.
4. Tool Use: call external APIs, plugins, or code execution environments. These calls obtain information (such as search engines and databases), perform operations (such as sending emails and booking services), or carry out computations.
5. Reflection & Iteration: (in more advanced Agents) able to evaluate its own actions and results, learn from them, and adjust subsequent plans.

A traditional LLM is like a knowledgeable librarian who can only talk on paper. An LLM Agent is more like an all-around personal assistant. This assistant knows a lot, can run errands and get things done, and can even think of the best solution proactively.

<div align='center'>
    <img src="/images/7-images/7-3-agent-how-it-works.webp" alt="Figure 7.7 How an Agent Works" width="1600" height="1375" loading="lazy" decoding="async" />
    <p>Figure 7.7 How an Agent Works</p>
</div>

The LLM Agent combines the language understanding and generation capabilities of large language models with key modules such as planning, memory, and tool use. It achieves autonomy and complex task processing capabilities beyond traditional large models. This capability gives LLM Agents broad application potential in many vertical domains (such as law, healthcare, and finance). Figure 7.7 shows how an Agent works.

### 7.3.2 Types of LLM Agents

The concept of LLM Agents is still evolving rapidly. However, we can roughly classify them into several categories based on their design philosophy and capability focus:

Task-Oriented Agents:
- Characteristics: focused on completion of well-defined tasks in specific domains, such as customer service, code generation, and data analysis.
- Working style: usually have pre-set workflows and a specific set of callable tools. The LLM mainly understands user intent, fills task slots, generates responses, or calls appropriate tools.
- Examples: chatbots specialized in restaurant reservations, and coding assistants for programming support (e.g., GitHub Copilot embodies Agent characteristics in some advanced features).

Planning & Reasoning Agents:
- Characteristics: emphasize the ability to decompose complex tasks autonomously. They formulate multi-step plans and adjust according to environmental feedback. They usually need stronger reasoning abilities.
- Working style: often adopt specific thinking frameworks, such as ReAct (Reason+Act). The model first "thinks" (Reasoning) to analyze the current situation and the needed actions. It then executes "actions" (Action) by calling tools. After that, it proceeds to the next round of thinking based on the results returned by the tool. Prompt engineering techniques such as Chain-of-Thought (CoT) also form the basis of their reasoning.
- Examples: research-oriented Agents that must integrate multiple tools such as web search, calculators, and database queries to answer complex questions. Other examples are Agents capable of autonomous completion of tasks like "write a report on topic XX, with relevant data charts".

Multi-Agent Systems:
- Characteristics: composed of multiple Agents with different roles or capabilities. They work collaboratively to jointly accomplish a grander goal.
- Working style: Agents can communicate, collaborate, debate, or even compete with each other. For example, one Agent is responsible for planning, one for execution, and one for review.
- Examples: simulation of a software development team (Product Manager Agent, Programmer Agent, Tester Agent) that automatically generates and tests code. Another example is simulation of a corporate organizational structure to complete business planning. Frameworks such as AutoGen and ChatDev support building such systems.

Exploration & Learning Agents:
- Characteristics: these Agents execute tasks. They can also learn new knowledge and new skills, or optimize their own strategies through interaction with the environment. This behavior is similar to the Agent concept in reinforcement learning.
- Working style: these Agents can have more complex memory and reflection mechanisms. They can adjust future planning and actions based on experiences of success or failure.
- Examples: Agents that can explore an unknown software environment autonomously and learn how to operate in it. Other examples are Agents that continuously improve their strategy while they play games.

### 7.3.3 Hands-On: Building a Tiny-Agent

We will build a Tiny-Agent based on the `openai` library and its `tool_calls` functionality. This Agent is a simple task-oriented Agent. It can answer some simple questions based on user input.

The final implementation effect is shown in Figure 7.8:

<div align='center'>
    <img src="/images/7-images/7-3-tinyagent-example.webp" alt="Figure 7.5 Tiny-Agent example output" width="1600" height="368" loading="lazy" decoding="async" />
    <p>Figure 7.8 Effect Illustration</p>
</div>

#### Step 1: Initialize the Client and Model

First, we need a client that can call large models. Here we use the `openai` library and configure it to point to a service endpoint compatible with the OpenAI API, such as [SiliconFlow](https://cloud.siliconflow.cn/i/ybUFvmqK). Meanwhile, specify the model to use, such as `Qwen/Qwen2.5-32B-Instruct`.

```python
from openai import OpenAI

# Initialize the OpenAI client
client = OpenAI(
    api_key="YOUR_API_KEY",  # Replace with your API Key
    base_url="https://api.siliconflow.cn/v1", # Use the SiliconFlow API address
)

# Specify the model name
model_name = "Qwen/Qwen2.5-32B-Instruct"
```

> Note: You need to replace `YOUR_API_KEY` with a valid API Key obtained from [SiliconFlow](https://cloud.siliconflow.cn/i/ybUFvmqK) or another provider.

#### Step 2: Define Tool Functions

We define the tool functions that the Agent can use in the `src/tools.py` file. Each function needs a clear docstring that describes its functionality and parameters. The code uses the docstring to generate the JSON Schema of the tool automatically.

```python
# src/tools.py
from datetime import datetime

# Get the current date and time
def get_current_datetime() -> str:
    """
    Get the current date and time.
    :return: A string representation of the current date and time.
    """
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
    return formatted_datetime

def count_letter_in_string(a: str, b: str):
    """
    Count the number of occurrences of a letter in a string.
    :param a: The string to search in.
    :param b: The letter to count.
    :return: The number of times the letter appears in the string.
    """
    return str(a.count(b))

def search_wikipedia(query: str) -> str:
    """
    Search Wikipedia for the summaries of the top three pages matching the given query.
    :param query: The query string to search for.
    :return: A string containing the summaries of the top three pages.
    """
    page_titles = wikipedia.search(query)
    summaries = []
    for page_title in page_titles[: 3]:  # Take the first three page titles
        try:
            # Use the page function of the wikipedia module to get the Wikipedia page object for the specified title.
            wiki_page = wikipedia.page(title=page_title, auto_suggest=False)
            # Get the page summary
            summaries.append(f"页面: {page_title}\n摘要: {wiki_page.summary}")  # (Gloss: "Page: {page_title}\nSummary: {wiki_page.summary}")
        except (
                wikipedia.exceptions.PageError,
                wikipedia.exceptions.DisambiguationError,
        ):
            pass
    if not summaries:
        return "维基百科没有搜索到合适的结果"  # (Gloss: "Wikipedia did not find any suitable results")
    return "\n\n".join(summaries)
# ... (there may be other tool functions as well)
```

For the OpenAI API to understand these tools, we must convert them into a specific JSON Schema format. The `function_to_json` helper function in `src/utils.py` does this conversion.

```python
# src/utils.py (partial)
import inspect

def function_to_json(func) -> dict:
    # ... (function implementation details)
    # Return a dict conforming to the OpenAI tool schema
    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": inspect.getdoc(func),
            "parameters": {
                "type": "object",
                "properties": parameters,
                "required": required,
            },
        },
    }
```

#### Step 3: Build the Agent Class

We define the `Agent` class in the `src/core.py` file. This class manages the conversation history, calls the OpenAI API, handles tool call requests, and executes tool functions.

```python
# src/core.py (partial)
from openai import OpenAI
import json
from typing import List, Dict, Any
from utils import function_to_json
# Import the defined tool functions
from tools import get_current_datetime, add, compare, count_letter_in_string

SYSTEM_PROMPT = """
你是一个叫不要葱姜蒜的人工智能助手。你的输出应该与用户的语言保持一致。
当用户的问题需要调用工具时，你可以从提供的工具列表中调用适当的工具函数。
"""
# (Gloss: "You are an AI assistant named 'Buyao Congjiang Suan' (literally 'No Scallion, Ginger, or Garlic'). Your output should stay consistent with the user's language.
# When a user's question requires calling a tool, you can call the appropriate tool function from the provided tool list.")

class Agent:
    def __init__(self, client: OpenAI, model: str = "Qwen/Qwen2.5-32B-Instruct", tools: List=[], verbose : bool = True):
        self.client = client
        self.tools = tools
        self.model = model
        self.messages = [
            {"role": "system", "content": SYSREM_PROMPT},
        ]
        self.verbose = verbose

    def get_tool_schema(self) -> List[Dict[str, Any]]:
        # Get the JSON schemas of all tools
        return [function_to_json(tool) for tool in self.tools]

    def handle_tool_call(self, tool_call):
        # Handle a tool call
        function_name = tool_call.function.name
        function_args = tool_call.function.arguments
        function_id = tool_call.id

        function_call_content = eval(f"{function_name}(**{function_args})")

        return {
            "role": "tool",
            "content": function_call_content,
            "tool_call_id": function_id,
        }

    def get_completion(self, prompt) -> str:

        self.messages.append({"role": "user", "content": prompt})

        # Get the model's completion response
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            tools=self.get_tool_schema(),
            stream=False,
        )
        
        # Check whether the model called a tool        
        if response.choices[0].message.tool_calls:
            self.messages.append({"role": "assistant", "content": response.choices[0].message.content})
            # Handle tool calls
            tool_list = []
            for tool_call in response.choices[0].message.tool_calls:
                # Handle the tool call and add the result to the message list
                self.messages.append(self.handle_tool_call(tool_call))
                tool_list.append([tool_call.function.name, tool_call.function.arguments])
            if self.verbose:
                print("调用工具：", response.choices[0].message.content, tool_list)  # (Gloss: "Tool call: ")
            # Get the model's completion response again, this time including the tool call results
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                tools=self.get_tool_schema(),
                stream=False,
            )

        # Add the model's completion response to the message list
        self.messages.append({"role": "assistant", "content": response.choices[0].message.content})
        return response.choices[0].message.content
```

The Agent's workflow is as follows:

1.  Receive user input.
2.  Call the large model (e.g., Qwen). Inform it of the available tools and their Schemas.
3.  If the model decides to call a tool, the Agent parses the request and executes the corresponding Python function.
4.  The Agent returns the execution result of the tool to the model.
5.  The model generates the final reply based on the tool result.
6.  The Agent returns the final reply to the user.

As shown in Figure 7.9, the Agent tool-calling workflow:

<div align='center'>
    <img src="/images/7-images/7-3-Tiny_Agent.webp" alt="Figure 7.9 Agent Workflow" width="1524" height="1088" loading="lazy" decoding="async" />
    <p>Figure 7.9 Agent Workflow</p>
</div>

#### Step 4: Run the Agent

Now we can instantiate and run the Agent. The `if __name__ == "__main__":` section of `demo.py` provides a simple command-line interaction example.

```python
# demo.py (partial)
if __name__ == "__main__":
    client = OpenAI(
        api_key="YOUR_API_KEY", # Replace with your API Key
        base_url="https://api.siliconflow.cn/v1",
    )

    # Create the Agent instance, passing in the client, model name, and tool function list
    agent = Agent(
        client=client,
        model="Qwen/Qwen2.5-32B-Instruct",
        tools=[get_current_datetime, add, compare, count_letter_in_string],
        verbose=True # Set to True to see tool call information
    )

    # Start the interactive conversation loop
    while True:
        # Use colored output to distinguish user input from AI answers
        prompt = input("\033[94mUser: \033[0m")  # Show the user input prompt in blue
        if prompt.lower() == "exit":
            break
        response = agent.get_completion(prompt)
        print("\033[92mAssistant: \033[0m", response)  # Show the AI assistant's answer in green
```

Example interaction:

```bash
User: 你好  # (Gloss: "Hello")
Assistant:  你好！有什么可以帮助你的吗？  # (Gloss: "Hello! Is there anything I can help you with?")
User: 9.12和9 .2哪个更大？  # (Gloss: "Which is bigger, 9.12 or 9.2?")
调用工具： ['compare']  # (Gloss: "Tool call: ['compare']")
Assistant:  9.2 比 9.12 更大。  # (Gloss: "9.2 is bigger than 9.12.")
User: 为什么？  # (Gloss: "Why?")
Assistant:  当我们比较9.12和9.2时，可以将它们看作是9.12和9.20。由于9.20在小数点后第二位是0，而9.12在小数点后第二位是2，所以在小数点后第一位相等的情况下，9.20（即9.2）大于9.12。因此，9.2 比 9.12 更大。  # (Gloss: "When we compare 9.12 and 9.2, we can treat them as 9.12 and 9.20. Since 9.20 has 0 in the second decimal place while 9.12 has 2 in the second decimal place, when the first decimal places are equal, 9.20 (i.e., 9.2) is greater than 9.12. Therefore, 9.2 is bigger than 9.12.")
User: strawberry中有几个r？  # (Gloss: "How many r's are in strawberry?")
调用工具： ['count_letter_in_string']  # (Gloss: "Tool call: ['count_letter_in_string']")
Assistant:  单词 "strawberry" 中有3个字母 'r'。  # (Gloss: "The word \"strawberry\" has 3 letter 'r's.")
User: 你确信嘛？  # (Gloss: "Are you sure?")
调用工具： ['count_letter_in_string']  # (Gloss: "Tool call: ['count_letter_in_string']")
Assistant:  是的，我确定。单词 "strawberry" 中确实有3个字母 'r'。让我们再次确认一下，"strawberry" 中的 'r' 确实出现了3次。  # (Gloss: "Yes, I'm sure. The word \"strawberry\" indeed has 3 letter 'r's. Let's double-check: the 'r' in \"strawberry\" really appears 3 times.")
User: 好的 你很薄，现在几点 了？  # (Gloss: "OK, you are very thin (likely a typo), what time is it now?")
调用工具： ['get_current_datetime']  # (Gloss: "Tool call: ['get_current_datetime']")
Assistant:  当前的时间是2025年4月26日17:01:33。不过，我注意到您提到“你很薄”，这似乎是一个打字错误，如果您有任何其他问题或者需要进一步的帮助，请告诉我！  # (Gloss: "The current time is April 26, 2025, 17:01:33. However, I noticed you mentioned 'you are very thin', which seems to be a typo. If you have any other questions or need further help, please let me know!")
User: exit
```

In addition, we have also prepared a Streamlit app for demonstration. It can run locally to show the capabilities of the Agent. Run it with `streamlit run web_demo.py`. The following shows the running effect of the Agent.

<div align='center'>
    <img src="/images/7-images/7-3-streamlit-demo.webp" alt="Figure 7.10 Streamlit Demo" width="1562" height="1494" loading="lazy" decoding="async" />
    <p>Figure 7.10 Streamlit Demo</p>
</div>


The sources for this chapter are listed below.

[1] Hugging Face. (2023). *Open LLM Leaderboard: 开源大语言模型基准测试平台* [Open LLM Leaderboard: A Benchmark Testing Platform for Open-Source Large Language Models]. https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard  

[2] awacke1. (2023). *LMSYS Chatbot Arena Leaderboard: 大型语言模型竞技场评估平台* [LMSYS Chatbot Arena Leaderboard: An Evaluation Platform for Large Language Model Arena]. https://huggingface.co/spaces/awacke1/lmsys-chatbot-arena-leaderboard  

[3] OpenCompass 团队 [OpenCompass Team]. (2023). *OpenCompass: 大模型统一评测平台* [OpenCompass: A Unified Evaluation Platform for Large Models]. https://rank.opencompass.org.cn/home  

[4] OpenCompass 金融榜团队 [OpenCompass Finance Leaderboard Team]. (2024). *CFBENCHMARK: 金融领域大模型评测榜单* [CFBENCHMARK: A Leaderboard for Evaluating Large Models in the Financial Domain]. https://specialist.opencompass.org.cn/CFBenchmark  

[5] OpenCompass 安全榜团队 [OpenCompass Safety Leaderboard Team]. (2024). *Flames: 大模型安全评测榜单* [Flames: A Safety Evaluation Leaderboard for Large Models]. https://flames.opencompass.org.cn/leaderboard  

[6] OpenCompass 通识榜团队 [OpenCompass General Knowledge Leaderboard Team]. (2024). *BotChat: 大模型通用对话能力评测* [BotChat: Evaluating the General Conversational Ability of Large Models]. https://botchat.opencompass.org.cn/  

[7] OpenCompass 法律榜团队 [OpenCompass Legal Leaderboard Team]. (2024). *LawBench: 法律领域大模型评测* [LawBench: Evaluating Large Models in the Legal Domain]. https://lawbench.opencompass.org.cn/leaderboard  

[8] OpenCompass 医疗榜团队 [OpenCompass Medical Leaderboard Team]. (2024). *MedBench: 医疗领域大模型评测* [MedBench: Evaluating Large Models in the Medical Domain]. https://medbench.opencompass.org.cn/leaderboard  

[9] Zhi Jing, Yongye Su, and Yikun Han. (2024). *When Large Language Models Meet Vector Databases: A Survey.* arXiv preprint arXiv:2402.01763.

[10] Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Meng Wang, and Haofen Wang. (2024). *Retrieval-Augmented Generation for Large Language Models: A Survey.* arXiv preprint arXiv:2312.10997.

[11] Zhiruo Wang, Jun Araki, Zhengbao Jiang, Md Rizwan Parvez, and Graham Neubig. (2023). *Learning to Filter Context for Retrieval-Augmented Generation.* arXiv preprint arXiv:2311.08377.

[12] Ori Ram, Yoav Levine, Itay Dalmedigos, Dor Muhlgay, Amnon Shashua, Kevin Leyton-Brown and Yoav Shoham. (2023). *In-Context Retrieval-Augmented Language Models.* arXiv preprint arXiv:2302.00083.
