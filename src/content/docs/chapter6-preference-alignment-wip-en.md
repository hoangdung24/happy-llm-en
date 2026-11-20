---
title: "Chapter 6 Supplement: Preference Alignment (WIP)"
description: "Preference alignment for LLMs: reward modelling, RLHF, and direct preference optimization. Work in progress chapter."
---

Before we get into the details of reinforcement learning, let us first look at its origins. Reinforcement Learning (RL) is nothing new. Its theoretical foundations go back to behavioral psychology in the early 20th century. Edward Thorndike and B.F. Skinner studied animal learning in that period. Thorndike proposed the "Law of Effect". If a behavior brings a positive outcome, the probability that the animal repeats it increases. Skinner developed this idea further. He proposed the theory of operant conditioning, which shapes behavior through rewards and punishments.

Reinforcement learning in computer science grew out of these psychological principles. In the 1980s, computing power improved and mathematical theory developed. People began to apply these bio-psychological learning concepts to machines and computer programs. In this way, they developed reinforcement learning in the modern sense.

## 6.4.1 Basic Principles of Reinforcement Learning

Now we move on to the core part. These are the basic principles of reinforcement learning.

- State: the specific situation of a system at a given moment. In a board game, the state can be the current arrangement of all pieces on the board. For a self-driving car, the state can include the car's speed, position, and the positions of surrounding obstacles, etc.
- Action: an operation the agent can execute in a given state. For a bicycle, actions can include moving forward, stopping, and turning, etc. In a complex system, the action set can be very large.
- Reward: the feedback the agent receives after it executes an action. A reward is usually a numeric value. Rewards can be immediate or delayed. A good action can receive a positive reward, and a bad action can receive a negative reward.
- Policy: a set of rules that guide how the agent selects actions. The policy tells the agent what to do in each state.
- Value Function: an evaluation tool for a policy. It aims to predict the total reward that the agent can obtain in the long run from the current state. The value function helps the agent consider the reward of the current step. It also helps the agent weigh short-term and long-term gains.
- Model: in some reinforcement learning systems, we build a model of the environment. This model helps the agent foresee the outcomes of its actions. This is very useful in many complex computation scenarios.

<img src="/images/7.1-1.webp" alt="Reinforcement Learning" width="798" height="514" loading="lazy" decoding="async" />

These elements work together. They help the agent learn the best action strategy through continuous trial and error in a virtual environment. In reinforcement learning, the agent is the subject of learning and decision-making. It interacts with the environment through the following steps:

1. Observe the state: the agent first observes the current state (State).
2. Select an action: based on the observed state and a pre-determined policy, the agent selects an action (Action).
3. Execute the action: the agent executes the selected action.
4. Receive the reward and the new state: after the agent executes the action, it receives the corresponding reward (Reward). It also receives the updated new state (State) from the environment.
5. Update the policy: the agent uses the received reward information to adjust its policy, so that it obtains better results in the future.

The agent repeats this process over and over. It continuously optimizes its policy through repeated interactions. The goal is to perform better and better on the given task.

## 6.4.2 The Goal of Reinforcement Learning

The goal of reinforcement learning is very clear. Through repeated trial and learning in a given environment, the agent must choose a series of actions that maximize its total cumulative reward. This may sound somewhat abstract, so we can use playing a game as an analogy. In a game, the player's goal is to earn a high score or complete a level. The player does this through a series of operations (such as walking, jumping, fighting monsters). In reinforcement learning, this concept of a high score or a successfully passed level corresponds to "maximizing the reward".

Mathematically, we can express this goal as training a policy $\pi$. In all states $s$, the actions that the agent chooses must maximize the expected value of the return $R(\tau)$. Specifically, we want to maximize the following expectation:

$$
E(R(\tau))_{\tau \sim P_{\theta}(\tau)} = \sum_{\tau} R(\tau) P_{\theta}(\tau)
$$

where:
- $E(R(\tau))_{\tau \sim P_{\theta}(\tau)}$: the expected value of the return $R(\tau)$ of trajectory $\tau$ under the policy $P_{\theta}(\tau)$.
- $R(\tau)$: the return of trajectory $\tau$, i.e., the sum of all rewards obtained from the initial state to the terminal state.
- $\tau$: denotes a trajectory, i.e., the sequence of states and actions of the agent in the environment.
- $P_{\theta}(\tau)$: the probability of generating trajectory $\tau$ under parameters $\theta$, usually determined by the policy or the policy network.
- $\theta$: the parameters of the policy, controlling the behavior of the policy $P_{\theta}$.

To find this policy, we use gradient ascent. We continuously update the policy parameters $\theta$, so that $E(R(\tau))_{\tau \sim P_{\theta}(\tau)}$ keeps increasing.

This way of learning is very effective. It does not rely on large amounts of labeled data. Instead, it learns through direct interaction with the environment and through feedback from the environment. This gives reinforcement learning great potential in many complex tasks that require adaptation and decision-making. Robot control, autonomous driving, financial trading, and even games are examples.

People apply reinforcement learning to large models, such as AlphaGo and AlphaZero. These models have further shown the strong capability of reinforcement learning in complex tasks. These models continuously optimized their policies through reinforcement learning methods. In the end, they defeated top human players in games such as Go and chess. This demonstrates the great potential of reinforcement learning in complex tasks.

Reinforcement learning can also be used for preference alignment problems. For example, it can let a large model learn to imitate the way humans communicate. People also use it in fields such as autonomous driving. The application domains of reinforcement learning are very broad, and there will be more application scenarios in the future.

## 6.4.3 The Reward Model

In the field of natural language processing, large language models (such as the Llama series and the Qwen series) have already demonstrated strong capabilities. They show strong text understanding and generation. However, these pretrained models cannot always directly satisfy specific business requirements and human values. For this reason, people usually need to perform "Instruction Tuning" on pretrained models. They provide the model with specific instructions (prompts) and examples. Then the model performs tasks such as dialogue, question answering, and text generation in a way that better matches human expectations.

After the initial instruction tuning, we also want the model's answers to be correct. We also want them to satisfy human aesthetics, values, and safety standards to the greatest extent. To this end, we introduce the concept of Reinforcement Learning from Human Feedback (RLHF). In RLHF, we first obtain preferences over the model's answers from human annotators. For example, we present multiple model answers and have human annotators rank them. Then we use this feedback to guide the model's learning. In this way, we continuously improve the alignment between the model's generated content and human preferences.

In the RLHF pipeline, we need to build a dedicated Reward Model to "score" (assign rewards to) the model's answers automatically. We train this reward model on human-annotated data. In actual deployment, it independently and automatically scores model outputs. This reduces the cost and latency of continuous human involvement.

## 6.4.4 Dataset Construction

Before we build the Reward Model, we first need to prepare a high-quality human feedback dataset. The core goal of this dataset is to provide multiple candidate answers (completions) for each given prompt. Human annotators then carefully evaluate and rank these answers. Through comparing and filtering the answers, we can provide the machine model with a clear reference standard. This helps the model further learn how to generate outputs that better match human expectations for a given task.

We can collect the data following these steps:

1. Collect initial answers: first, we need an already basically fine-tuned "large model". This is often a pretrained model with a certain level of instruction understanding and generation capability. We generate multiple answers from it for a set of carefully designed prompts. These answers will serve as the basis for the later human annotation work.


2. Human annotation and evaluation: with multiple candidate answers in hand, we invite professional annotators or crowdsourced annotators to evaluate the quality of each answer. These evaluations are usually based on a series of pre-designed evaluation criteria. These criteria include the accuracy, completeness, contextual relevance, and language fluency of the answer. They also include whether the answer adheres to ethical and safety guidelines. Comparing and ranking the different answers helps us identify the best and worst answers. In this way, we form valuable training data.

3. Data formatting and organization: after annotation is complete, we organize and format the data. We usually use structured data formats that are convenient for computer processing, such as JSON, CSV, or others. The dataset must clearly identify each question (prompt), its corresponding multiple answers (completions), and the human annotators' choices among these answers. For example, the best answer is marked "chosen" and the inferior answer is marked "rejected". These labels can directly serve as the supervised signal for the reward model's learning. Then the reward model automatically prefers generating high-quality answers during training.

Below is a simple data example. It shows two questions (question) and their corresponding answers and human evaluation results. By comparing the "chosen" and "rejected" fields, we can see which answer is of higher quality.

```json
[
    {
        "question": "Python中的列表是什么？ (What is a list in Python?)",
        "chosen": "Python中的列表是一种有序的可变容器，允许存储多个元素，并且可以通过索引访问。 (A list in Python is an ordered, mutable container that allows storing multiple elements and supports access by index.)",
        "rejected": "Python中的列表用于存储数据。 (A list in Python is used to store data.)"
    },
    {
        "question": "Python中的元组是什么？ (What is a tuple in Python?)",
        "chosen": "Python中的元组是一种有序的不可变容器，允许存储多个元素，并且一旦创建就不能修改。 (A tuple in Python is an ordered, immutable container that allows storing multiple elements and cannot be modified once created.)",
        "rejected": "Python中的元组用于存储数据。 (A tuple in Python is used to store data.)"
    }
]
```

In the above example, the human annotators judged the answers under the "chosen" field to be of higher quality than the corresponding "rejected" answers. They judged them on description, accuracy, and informativeness. For example, for the definition of a list, the "chosen" answer explains the features of a list more clearly. It names the features as ordered, mutable, and index access. It does not stay at the vague description of "used to store data".


## 7.2.2 Reward Model Training

We can train the reward model with the help of TRL (Transformer Reinforcement Learning). TRL is a reinforcement learning framework for large models. TRL is a training framework based on reinforcement learning. It guides the model, through human feedback, to generate answers that better match human expectations. In TRL, we treat the reward model as an independent component. We use it to evaluate the answers generated by the model. It gives rewards or penalties based on the evaluation results.
