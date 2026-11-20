---
title: "Chapter 1: Fundamental Concepts of NLP"
---

Natural Language Processing (NLP) is an important branch of artificial intelligence. NLP helps computers understand and process human language, so that humans and machines can communicate naturally. Information technology develops fast, and text data is now a daily part of our lives. NLP tools help us extract useful information from large amounts of text. They also help us understand the deeper meaning of language. The NLP field passed through several technology changes. It started with rule-based methods, moved to statistical learning, and now uses deep learning in many areas. Text representation is one core technology of NLP. Research and progress in text representation decide how well NLP systems perform.

This chapter introduces the basic concepts of NLP. It helps you understand and review the related knowledge.

## 1.1 What is NLP

NLP is a technology that lets computers understand, interpret, and generate human language. It is a very active and important research area in artificial intelligence. Its main task is to copy human thinking and language use with computer programs. NLP joins knowledge from computer science, artificial intelligence, linguistics, and psychology. It aims to break the wall between human language and computer language. That wall blocks easy talk and interaction between people and machines.

NLP technology lets computers do many language tasks. These tasks include Chinese word segmentation, subword segmentation, part-of-speech tagging, text classification, entity recognition, relation extraction, text summarization, machine translation, and automatic question answering. These tasks need more than the look of language. Computers must also find the deep meaning behind the words. This meaning includes semantics, context, sentiment, and culture.

Deep learning and other modern technologies brought big progress to NLP. Deep learning models train on large amounts of data. From that data, they learn complex language patterns and structures. Their performance on many NLP tasks comes close to human level, or even passes it. But NLP still faces many hard problems. These problems include unclear meaning, abstract ideas, metaphor, and irony. Researchers work on these problems with better algorithms, bigger datasets, and better language models. This work pushes NLP technology forward.

## 1.2 The Development History of NLP

NLP evolved from early rule-based methods, to statistical methods, and then to machine learning and deep learning. Each change moved NLP forward. It brought large gains in machine translation, sentiment analysis, entity recognition, and text summarization. Computing power grows, and algorithms improve. NLP will play an even larger role in more fields.

### Early Exploration (1940s – 1960s)

NLP exploration began after World War II. People saw the value of translating one language into another by machine. In 1950, Alan Turing proposed the Turing Test.

> He said that a machine can take part in a conversation by using a typewriter. If the machine imitates a human with no noticeable difference, then we can consider it capable of thinking.

The test asks whether a machine can show intelligent behavior that matches human behavior. In the same period, Noam Chomsky proposed the theory of generative grammar. This theory shaped how researchers understood machine translation. Machine translation systems of this era were very simple. They relied on dictionary lookups and basic word-order rules. Their results were not good.

### Symbolicism and Statistical Methods (1970s – 1990s)

After the 1970s, NLP researchers explored new areas. These areas included logic-based paradigms and natural language understanding. Researchers split into two camps: symbolicists (or rule-based approaches) and statistical methods. Symbolicists studied formal languages and generative grammar. Statistical researchers used statistical and probabilistic approaches. In the 1980s, computing power grew and machine learning algorithms arrived. Statistical models then began to replace complex "hand-written" rules.

### Machine Learning and Deep Learning (2000s – Present)

After the 2000s, deep learning brought large progress to NLP. Models such as Recurrent Neural Networks (RNN), Long Short-Term Memory networks (LSTM), and attention mechanisms saw wide use in NLP tasks. They gave strong results. In 2013, the Word2Vec model opened a new era of word vector representation. It gave NLP tasks a better method of text representation. In 2018, the BERT model started a new wave of pre-trained language models. It brought new opportunities and new challenges to NLP. In recent years, Transformer-based models such as GPT-3 train with very large numbers of parameters. They can generate high-quality text that in some cases rivals human writing.

## 1.3 NLP Tasks

In the broad research field of NLP, several core tasks form the foundation of the field. These tasks cover everything from basic text processing to complex semantic understanding and generation. These tasks include, but are not limited to, Chinese word segmentation, subword segmentation, part-of-speech tagging, text classification, entity recognition, and relation extraction. They also include text summarization, machine translation, and the development of automatic question answering systems. Each task has its own specific challenges and application scenarios. Together they drive the development of language technology, providing powerful tools for processing and analyzing the ever-growing volume of text data.

### 1.3.1 Chinese Word Segmentation

Chinese Word Segmentation (CWS) is a fundamental task in the NLP field. Chinese text has no obvious separators between words, such as the spaces in English. A reader cannot determine word boundaries directly by spaces. Therefore, Chinese word segmentation is the first step in Chinese text processing. Its purpose is to split continuous Chinese text into meaningful sequences of words.

```
English input: The cat sits on the mat.
English segmentation output: [The | cat | sits | on | the | mat]
Chinese input: 今天天气真好，适合出去游玩. (The weather is really nice today, good for going out.)
Chinese segmentation output: ["今天" (today), "天气" (weather), "真" (really), "好" (good), "，", "适合" (suitable), "出去" (go out), "游玩" (play), "。"]
```

A correct segmentation result is important for later tasks such as part-of-speech tagging, entity recognition, and syntactic analysis. Inaccurate segmentation directly affects the effectiveness of the entire text processing pipeline.

```
Input: 雍和宫的荷花开的很好。 (The lotus flowers at Yonghe Temple are blooming beautifully.)

Correct segmentation: 雍和宫 | 的 | 荷花 | 开 | 的 | 很 | 好 | 。
Incorrect segmentation 1: 雍 | 和 | 宫的 | 荷花 | 开的 | 很好 | 。 (The place name is split apart)
Incorrect segmentation 2: 雍和 | 宫 | 的荷 | 花开 | 的很 | 好。 (Word boundaries are chaotic)
```

A correct segmentation result is important for later tasks such as part-of-speech tagging, entity recognition, and syntactic analysis. Inaccurate segmentation directly affects the effectiveness of the entire text processing pipeline.

### 1.3.2 Subword Segmentation

Subword segmentation is a common text preprocessing technique in the NLP field. It aims to further decompose vocabulary into smaller units, that is, subwords. Subword segmentation is particularly useful for handling vocabulary sparsity. When the model encounters rare words or unseen new words, it can understand or generate them through known subword units. Subword segmentation is especially important when processing languages with complex spelling and many compound words (such as German). It is also important in pre-trained language models (such as the BERT and GPT series).

Many methods exist for subword segmentation. Common ones include Byte Pair Encoding (BPE), WordPiece, Unigram, and SentencePiece. The basic idea of these methods is to decompose words into smaller, frequently occurring fragments. These fragments can be single characters, character combinations, or roots and affixes.

```
Input: unhappiness

Without subword segmentation: the whole word is treated as one unit, output: "unhappiness"
With subword segmentation (assuming the BPE algorithm): the word is split into: "un", "happi", "ness"
```

In this example, subword segmentation decomposes the word "unhappiness" into three parts. The first part is the prefix "un", which indicates negation. The second part is "happi", a root variant of "happy" meaning happiness. The third part is "ness", a noun suffix that indicates a state. The model has never seen the complete word "unhappiness". Even so, it can understand through these known subwords that the general meaning is "the state of not being happy".

### 1.3.3 Part-of-Speech Tagging

Part-of-Speech Tagging (POS Tagging) is a fundamental task in the NLP field. Its goal is to assign a part-of-speech tag to each word in a text, such as noun, verb, or adjective. This process usually uses a predefined tag set, such as common English tags: noun (N), verb (V), and adjective (Adj). POS tagging is important for understanding sentence structure, for syntactic analysis, and for semantic role labeling. POS tagging also supports other advanced NLP tasks. Through POS tagging, computers can better understand the meaning of text. They can then do more complex processing, such as information extraction, sentiment analysis, and machine translation.

Suppose we have the English sentence: She is playing the guitar in the park.

The POS tagging result is as follows:

- She (Pronoun, PRP)
- is (Verb, VBZ)
- playing (present participle of a verb, Verb, VBG)
- the (Determiner, DT)
- guitar (Noun, NN)
- in (Preposition, IN)
- the (Determiner, DT)
- park (Noun, NN)
- . (Punctuation, .)

POS tagging usually relies on machine learning models. These models include Hidden Markov Models (HMM), Conditional Random Fields (CRF), and deep-learning-based Recurrent Neural Networks (RNN) and Long Short-Term Memory networks (LSTM). These models learn to predict the part of speech of each word in new sentences from large amounts of labeled data.

### 1.3.4 Text Classification

Text Classification is a core task in the NLP field. It automatically assigns a given text to one or more predefined categories. This technique is widely used in many scenarios, such as sentiment analysis, spam detection, news classification, and topic identification. The key to text classification lies in understanding the meaning and context of the text. On that basis, the task maps the text to a specific category.

Suppose there is a text classification task. Its purpose is to classify news articles into one of three categories: "Sports", "Politics", or "Technology".

```
Text: "NBA季后赛将于下周开始，湖人和勇士将在首轮对决。" (The NBA playoffs will start next week; the Lakers and the Warriors will face off in the first round.)
Category: "Sports"

Text: "美国总统宣布将提高关税，引发国际贸易争端。" (The US President announced tariff increases, triggering international trade disputes.)
Category: "Politics"

Text: "苹果公司发布了新款 Macbook，配备了最新的m3芯片。" (Apple released a new MacBook equipped with the latest M3 chip.)
Category: "Technology"
```

The key to success in text classification lies in choosing appropriate feature representations and classification algorithms. It also lies in having high-quality training data. Deep learning technology has developed, and neural networks for text classification have become a trend. These networks can capture complex patterns and semantic information in text data. As a result, many tasks show significant performance improvements.

### 1.3.5 Entity Recognition

Named Entity Recognition (NER) is a key task in the NLP field. It automatically identifies entities with specific meanings in text and classifies them into predefined categories. These categories include person names, locations, organizations, dates, and times. Entity recognition is important for applications such as information extraction, knowledge graph construction, question answering systems, and content recommendation. It helps systems understand the key elements in text and their attributes.

Suppose there is an entity recognition task. Its purpose is to identify entities such as person names, place names, and organization names from text.

```
Input: 李雷和韩梅梅是北京市海淀区的居民，他们计划在2024年4月7日去上海旅行。
(Li Lei and Han Meimei are residents of Haidian District, Beijing. They plan to travel to Shanghai on April 7, 2024.)

Output: [("李雷" (Li Lei), "Person"), ("韩梅梅" (Han Meimei), "Person"), ("北京市海淀区" (Haidian District, Beijing), "Location"), ("2024年4月7日" (April 7, 2024), "Date"), ("上海" (Shanghai), "Location")]
```

Through the entity recognition task, we can identify the entities in text and also learn their categories. This work provides important information for a deep understanding of the content and context of text. NLP technology has developed, and the accuracy and efficiency of entity recognition continue to improve. It can provide strong support for many NLP applications.

### 1.3.6 Relation Extraction

Relation Extraction is a key task in the NLP field. Its goal is to identify semantic relations between entities from text. These relations can be causal relations, ownership relations, kinship relations, or geographic location relations. Relation extraction is very important for understanding text content and for constructing knowledge graphs. It also improves the ability of machines to understand language.

Suppose we have the following sentence:

```
Input: 比尔·盖茨是微软公司的创始人。 (Bill Gates is the founder of Microsoft.)

Output: [("比尔·盖茨" (Bill Gates), "Founder", "微软公司" (Microsoft))]
```

In this example, the goal of the relation extraction task is to identify the "founder" relation between "Bill Gates" and "Microsoft" in the text. Through relation extraction, we can extract useful information from text. This step helps computers better understand the text content. It also provides support for later tasks, such as knowledge graph construction and question answering systems.

### 1.3.7 Text Summarization

Text Summarization is an important task in NLP. Its goal is to generate a concise and accurate summary that captures the main content of the original text. Text summarization has two major categories, based on the generation method: Extractive Summarization and Abstractive Summarization.

- Extractive summarization: Extractive summarization composes the summary by directly selecting key sentences or phrases from the original text. Its advantage is that the information in the summary comes entirely from the original text, so accuracy is relatively high. However, it is merely a concatenation of sentences from the original text, so the generated summary may sometimes lack fluency.
- Abstractive summarization: Unlike extractive summarization, abstractive summarization selects text fragments. It also reorganizes and rewrites these fragments and generates new content. Abstractive summarization is more challenging because it requires an understanding of the deeper meaning of the text. It must also express the same information in new ways. Abstractive summarization usually requires more complex models, such as attention-based sequence-to-sequence models (Seq2Seq).

Suppose we have the following news report:

```
2021年5月22日，国家航天局宣布，我国自主研发的火星探测器“天问一号”成功在火星表面着陆。此次任务的成功，标志着我国在深空探测领域迈出了重要一步。“天问一号”搭载了多种科学仪器，将在火星表面进行为期90个火星日的科学探测工作，旨在研究火星地质结构、气候条件以及寻找生命存在的可能性。
(On May 22, 2021, the China National Space Administration announced that China's independently developed Mars probe "Tianwen-1" successfully landed on the surface of Mars. The success of this mission marks an important step forward for China in the field of deep space exploration. "Tianwen-1" carries a variety of scientific instruments and will conduct scientific exploration on the Martian surface for 90 Martian days, aimed at studying Mars' geological structure and climate conditions, and searching for the possibility of life.)
```

Extractive summary:

```
我国自主研发的火星探测器“天问一号”成功在火星表面着陆，标志着我国在深空探测领域迈出了重要一步。
(China's independently developed Mars probe "Tianwen-1" successfully landed on the surface of Mars, marking an important step forward for China in the field of deep space exploration.)
```

Abstractive summary:

```
“天问一号”探测器成功实现火星着陆，代表我国在宇宙探索中取得重大进展。
(The "Tianwen-1" probe successfully achieved a Mars landing, representing major progress for China in space exploration.)
```

Text summarization has a wide range of applications in fields such as information retrieval, news delivery, and report generation. Through automatic summarization, users can quickly grasp the core information of a text. They save reading time and improve information processing efficiency.

### 1.3.8 Machine Translation

Machine Translation (MT) is a core task in the NLP field. It refers to the process of using computer programs to automatically translate one natural language (the source language) into another natural language (the target language). Machine translation involves the direct conversion of vocabulary. More importantly, it conveys the semantics, style, and cultural background of the source text accurately. This work makes the translation natural, accurate, and fluent in the target language. It crosses language barriers and promotes communication and understanding among speakers of different languages.

Suppose we have a Chinese sentence: "今天天气很好。" (The weather is very nice today.), and we want to translate it into English.

```
Source language: 今天天气很好。 (The weather is very nice today.)

Target language: The weather is very nice today.
```

In this simple example, machine translation accurately converts the Chinese sentence into English. It preserves the meaning and structure of the original sentence. However, the challenges facing machine translation increase when it processes longer and more complex texts. To improve the quality of machine translation, researchers continue to explore new methods and techniques, such as neural-network-based Seq2Seq models and Transformer models. These models can learn the complex mapping relationships between the source and target languages. They achieve more accurate and fluent translation.

### 1.3.9 Automatic Question Answering

Automatic Question Answering (QA) is an advanced task in the NLP field. It enables computers to understand questions posed in natural language and to automatically provide accurate answers based on a given data source. The QA task simulates the human ability to understand and answer questions. It covers everything from simple factual queries to complex reasoning and explanation. Building a QA system involves multiple NLP subtasks, such as information retrieval, text understanding, knowledge representation, and reasoning.

Automatic question answering has three rough types: Retrieval-based QA, Knowledge-based QA, and Community-based QA. Retrieval-based QA retrieves answers from large volumes of text by means of search engines. Knowledge-based QA answers questions through structured knowledge bases. Community-based QA relies on user-generated question-and-answer data, such as Q&A communities and forums.

The development and optimization of QA systems is an ongoing process. Technological progress and algorithmic improvements have improved these systems significantly in accuracy, understanding capability, and scope of application. QA systems combine different types of data sources and technical methods. As a result, they become increasingly intelligent and increasingly capable of handling complex and diverse questions.

## 1.4 The Development History of Text Representation

The purpose of text representation is to convert human language from its natural form into a form that computers can process. This conversion digitizes text data so that computers can effectively analyze and process it. Text representation is a foundational and essential piece of work in the NLP field. It directly influences and even determines the quality and performance of NLP systems.

In NLP, text representation converts linguistic units in text (such as characters, words, phrases, and sentences) into forms that computers can understand and manipulate. It also converts their relationships and structural information. These forms include vectors, matrices, or other data structures. Such representations must preserve sufficient semantic information for later NLP tasks, such as text classification, sentiment analysis, and machine translation. They must also take computational and storage efficiency into account.

The development of text representation has gone through multiple stages. It moved from early rule-based methods, to statistical learning approaches, and now to deep learning techniques. Text representation technology continues to evolve and provides strong support for the development of NLP.

### 1.4.1 Word Vectors

The Vector Space Model (VSM) is a fundamental and effective text representation method in the NLP field. Salton of Harvard University first proposed it. The vector space model achieves a mathematical representation of text. It converts text (including words, sentences, paragraphs, or entire documents) into vectors in a high-dimensional space. In this model, each dimension represents a feature term (for example, a character, word, word group, or phrase). Each element value in the vector represents the weight of that feature term in the text. Specific calculation formulas determine this weight (such as term frequency TF and inverse document frequency TF-IDF). The weight reflects the importance of the feature term in the text.

The applications of the vector space model are extremely broad, such as text similarity computation, text classification, and information retrieval. It converts complex text data into a mathematical form that is easy to compute and analyze. This form makes text similarity computation and pattern recognition possible. In addition, matrix operations such as eigenvalue computation and singular value decomposition (SVD) can optimize text vector representations. This optimization further improves processing efficiency and effectiveness.

However, the vector space model also has many problems. The most important ones are data sparsity and the curse of dimensionality. The huge number of feature terms leads to extremely high vector dimensions, and most element values are zero. In addition, the model is based on the assumption of independence between feature terms. It therefore ignores structural information in text, such as word order and context, and this limits the expressiveness of the model. The shortcomings of feature term selection and weight calculation methods are also problems that the vector space model must solve.

Word vectors in the VSM method:

```python
# "雍和宫的荷花很美" (The lotus flowers at Yonghe Temple are beautiful)
# Vocabulary size: 16384; words in the sentence: ["雍和宫" (Yonghe Temple), "的" (de), "荷花" (lotus), "很" (very), "美" (beautiful)] = 5 words

vector = [0, 0, ..., 1, 0, ..., 1, 0, ..., 1, 0, ..., 1, 0, ..., 1, 0, ...]
#                    ↑          ↑          ↑          ↑          ↑
#      Only 5 positions out of 16384 dimensions are 1; the other 16379 positions are 0
# Actual effective dimensions: only 5 (non-zero dimensions)
# Sparsity rate: (16384-5)/16384 ≈ 99.97%
```
> A vocabulary is a set containing all possible words. In the vector space model, each word corresponds to a position in the vocabulary. In this way, words can be converted into vector representations. For example, if the vocabulary size is 16384, then each word is represented as a 16384-dimensional vector. In this vector, only the position corresponding to that word is 1 and all other positions are 0.

To solve these problems, research on the vector space model has focused mainly on two aspects. The first is improving feature representation methods, such as keyword extraction using graph-based or topic-based methods. The second is improving and optimizing the calculation of feature term weights. This work uses fusion computations based on existing methods or proposes new calculation methods.

### 1.4.2 Language Models

The N-gram model is a statistics-based language model in the NLP field. It is widely used in many tasks, such as speech recognition, handwriting recognition, spelling correction, machine translation, and search engines. The core idea of the N-gram model is based on the Markov assumption. The probability of a word appearing depends only on the N-1 words preceding it. Here N represents the number of consecutive words and can be any positive integer. For example, when N=1, the model is called a unigram and considers only the probability of a single word. When N=2, it is called a bigram and considers the preceding word to estimate the probability of the current word. When N=3, it is called a trigram and considers the preceding two words to estimate the probability of the third word. Higher-order N-grams follow the same pattern.

The N-gram model estimates the probability of an entire sentence through the chain rule of conditional probability. Specifically, for a given sentence, the model computes the conditional probability of each N-gram appearing. It multiplies these probabilities to obtain the probability of the whole sentence. For example, for the sentence "The quick brown fox", as a trigram model we compute probabilities such as $P("brown" | "The", "quick")$ and $P("fox" | "quick", "brown")$, and multiply them together.

The advantages of N-grams are simple implementation, easy understanding, and good results in many tasks. However, the data sparsity problem arises when N is large. The parameter space of the model expands sharply, and the probability of the same N-gram sequence appearing becomes very low. This causes the model to fail to learn effectively, and its generalization ability declines. In addition, the N-gram model ignores range dependencies between words. It cannot capture the complex structures and semantic information in sentences.

Despite its limitations, the N-gram model is still widely used in many NLP tasks because of its simplicity and practicality. In some applications, combining the N-gram model with other techniques (such as deep learning models) can yield better performance.

### 1.4.3 Word2Vec

Word2Vec is a popular word embedding technique proposed by Tomas Mikolov et al. in 2013. It is a neural-network-based language model (NNLM). Its aim is to generate dense vector representations of words by learning the contextual relationships between words. The core idea of Word2Vec is to use the contextual information of words in text to capture the semantic relationships between words. Semantically similar or related words are then closer together in vector space.

The Word2Vec model has two main architectures. The Continuous Bag of Words (CBOW) model computes the context of a target word. It then outputs the vector representation of the target word based on the word vectors of the words in that context. The Skip-Gram model is the opposite of CBOW, and it uses the target word's vector representation to compute the word vectors of the context. Practice has verified that CBOW is suitable for small datasets, while Skip-Gram performs better on large corpora.

Compared with traditional high-dimensional sparse representations (such as One-Hot encoding), Word2Vec generates low-dimensional (usually a few hundred dimensions) dense vectors. These vectors help reduce computational complexity and storage requirements. The Word2Vec model can capture semantic relationships between words. For example, "king" and "queen" are positioned relatively close together in vector space, because in large amounts of text they usually appear in similar contexts. The Word2Vec model can also generalize well to unseen words, because it learns from contextual information rather than from a dictionary. However, the CBOW/Skip-Gram models are based on local context. They cannot capture long-distance dependencies and lack overall word-to-word relationships, so they underperform on some complex semantic tasks.

### 1.4.4 ELMo

ELMo (Embeddings from Language Models) achieved a leap from static word vectors to dynamic word vectors that handle polysemy (one word, multiple meanings). First, a language model is trained on a large corpus to obtain a word vector model. Then, for a specific task, the model is fine-tuned to obtain word vectors better suited to that task. ELMo was the first to introduce the pre-training idea into word vector generation. It uses a bidirectional LSTM structure that can capture the contextual information of words and generate richer and more accurate word vector representations.

ELMo adopts a typical two-stage process. The first stage is pre-training using a language model. The second stage occurs when performing a specific task. The model extracts the word vectors of the corresponding words from the pre-trained network as new features to supplement the downstream task. The RNN-based LSTM model has a long training time, and feature extraction is the key to optimizing and improving the ELMo model.

The main advantage of the ELMo model is its ability to capture the polysemy of words and contextual information. It produces richer and more accurate word vectors that are suitable for a variety of NLP tasks. However, the ELMo model also has some problems, such as high model complexity, long training time, and high consumption of computational resources.

## References

[1] Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, Jeffrey Dean. (2013). *Distributed Representations of Words and Phrases and their Compositionality.* arXiv preprint arXiv:1310.4546.

[2] Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. arXiv preprint arXiv:1810.04805.

[3] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin. (2023). *Attention Is All You Need.* arXiv preprint arXiv:1706.03762.

[4] Malek Hajjem, Chiraz Latiri. (2017). *Combining IR and LDA Topic Modeling for Filtering Microblogs.* Procedia Computer Science, 112, 761–770. https://doi.org/10.1016/j.procs.2017.08.166.

[5] Matthew E. Peters, Mark Neumann, Mohit Iyyer, Matt Gardner, Christopher Clark, Kenton Lee, Luke Zettlemoyer. (2018). *Deep contextualized word representations.* arXiv preprint arXiv:1802.05365.

[6] Salton, G., Wong, A., Yang, C. S. (1975). *A vector space model for automatic indexing.* Communications of the ACM, 18(11), 613–620. https://doi.org/10.1145/361219.361220.

[7] 赵京胜,宋梦雪,高祥,等.自然语言处理中的文本表示研究[J].软件学报,2022,33(01):102-128.DOI:10.13328/j.cnki.jos.006304.

[8] 中文信息处理发展报告（2016）前言[C]//中文信息处理发展报告（2016）.中国中文信息学会,2016:2-3.DOI:10.26914/c.cnkihy.2016.003326.
