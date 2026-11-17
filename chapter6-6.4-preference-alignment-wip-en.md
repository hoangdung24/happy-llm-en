# 6.4 Preference Alignment via Reinforcement Learning

Before we get into the details of reinforcement learning, let us first look at its origins. Reinforcement Learning (RL) is actually nothing new. Its theoretical foundations can be traced back to behavioral psychology in the early 20th century, especially the studies of animal learning by Edward Thorndike and B.F. Skinner. Thorndike proposed the "Law of Effect": if a behavior brings a positive outcome, the probability of that behavior being repeated increases. Skinner further developed this idea, proposing the theory of operant conditioning, shaping behavior through rewards and punishments.

Reinforcement learning in computer science grew out of these psychological principles. In the 1980s, with the improvement of computing power and the development of mathematical theory, people began trying to apply these bio-psychological learning concepts to machines and computer programs, thereby developing reinforcement learning in the modern sense.

## 6.4.1 Basic Principles of Reinforcement Learning

Now, we move on to the core part—the basic principles of reinforcement learning.

- State: the specific situation of a system at a given moment. For example, in a board game, the state can be represented by the current arrangement of all pieces on the board. For a self-driving car, the state may include the car's speed, position, and the positions of surrounding obstacles, etc.
- Action: an action is an operation the agent can execute in a given state. Taking a bicycle as an example, actions might include moving forward, stopping, turning, etc. In a complex system, the action set can be very large.
- Reward: this is the feedback the agent receives after executing an action, usually a numeric value. Rewards can be immediate or delayed. A good action may receive a positive reward, while a bad action may receive a negative reward.
- Policy: a policy is a set of rules guiding how the agent selects actions. Simply put, the policy tells the agent what to do in each state.
- Value Function: this is an evaluation tool for a policy, aiming to predict the total reward that can be obtained in the long run starting from the current state. The value function helps the agent consider not only the reward of the current step, but also better weigh short-term and long-term gains.
- Model: in some reinforcement learning systems, we build a model of the environment to help the agent foresee the outcomes of its actions. This is very useful in many complex computation scenarios.

![Reinforcement Learning](./images/7.1-1.png)

These elements work together to help the agent learn the best action strategy through continuous trial and error in a virtual environment. In reinforcement learning, the agent is the subject of learning and decision-making. It interacts with the environment through the following steps:

1. Observe the state: the agent first observes the current state (State).
2. Select an action: based on the observed state and a pre-determined policy, the agent selects an action (Action).
3. Execute the action: the agent executes the selected action.
4. Receive the reward and the new state: after executing the action, the agent receives the corresponding reward (Reward) and the updated new state (State) from the environment.
5. Update the policy: the agent uses the received reward information to adjust its policy, in order to obtain better results in the future.

Repeating this process over and over, the agent continuously optimizes its policy through repeated interactions, with the goal of performing better and better on the given task.

## 6.4.2 The Goal of Reinforcement Learning

The goal of reinforcement learning is very clear: ***through repeated trial and learning in a given environment, enable the agent to choose a series of actions that maximize its total cumulative reward.*** This may sound somewhat abstract; we can use playing a game as an analogy. In a game, the player's goal is to earn a high score or complete a level through a series of operations (such as walking, jumping, fighting monsters). In reinforcement learning, this concept of a high score or successfully passing a level corresponds to "maximizing the reward".

Mathematically, this goal can be expressed as training a policy $\pi$ such that, in all states $s$, the actions chosen by the agent maximize the expected value of the return $R(\tau)$. Specifically, we want to maximize the following expectation:

$$
E(R(\tau))_{\tau \sim P_{\theta}(\tau)} = \sum_{\tau} R(\tau) P_{\theta}(\tau)
$$

where:
- $E(R(\tau))_{\tau \sim P_{\theta}(\tau)}$: the expected value of the return $R(\tau)$ of trajectory $\tau$ under the policy $P_{\theta}(\tau)$.
- $R(\tau)$: the return of trajectory $\tau$, i.e., the sum of all rewards obtained from the initial state to the terminal state.
- $\tau$: denotes a trajectory, i.e., the sequence of states and actions of the agent in the environment.
- $P_{\theta}(\tau)$: the probability of generating trajectory $\tau$ under parameters $\theta$, usually determined by the policy or the policy network.
- $\theta$: the parameters of the policy, controlling the behavior of the policy $P_{\theta}$.

To find this policy, we use gradient ascent, continuously updating the policy parameters $\theta$ so that $E(R(\tau))_{\tau \sim P_{\theta}(\tau)}$ keeps increasing.

This way of learning is very effective, because it does not rely on large amounts of labeled data, but instead learns through direct interaction with and feedback from the environment. This gives reinforcement learning great potential in many complex tasks that require adaptation and decision-making, such as robot control, autonomous driving, financial trading, and even games.

The application of reinforcement learning to large models, such as AlphaGo and AlphaZero, has further shown people the powerful capability of reinforcement learning in complex tasks. These models continuously optimized their policies through reinforcement learning methods and ultimately defeated top human players in games such as Go and chess, demonstrating the great potential of reinforcement learning in complex tasks.

Reinforcement learning can also be used for preference alignment problems; for example, it can let a large model learn to imitate the way humans communicate, and it is also used in fields such as autonomous driving. The application domains of reinforcement learning are very broad, and there will be more application scenarios in the future.

## 6.4.3 The Reward Model

In the field of natural language processing, large language models (such as the Llama series, the Qwen series, etc.) have already demonstrated powerful text understanding and generation capabilities. However, these pretrained models cannot always directly satisfy specific business requirements and human values. For this reason, people usually need to perform "Instruction Tuning" on pretrained models, i.e., providing the model with specific instructions (prompts) and examples, so that it performs in tasks such as dialogue, question answering, and text generation in a way that better matches human expectations.

After completing the initial instruction tuning, we also want the model's answers to be not only correct, but also to satisfy human aesthetics, values, and safety standards to the greatest extent. To this end, the concept of Reinforcement Learning from Human Feedback (RLHF) is introduced. In RLHF, we first obtain preferences over the model's answers from human annotators (for example, presenting multiple model answers and having human annotators rank them), and then use this feedback to guide the model's learning, thereby continuously improving the alignment between the model's generated content and human preferences.

In order to automatically "score" (assign rewards to) the model's answers in the RLHF pipeline, we need to build a dedicated Reward Model. This reward model is trained on human-annotated data and, in actual deployment, independently and automatically scores model outputs, thereby reducing the cost and latency of continuous human involvement.

## 6.4.4 Dataset Construction

Before building the Reward Model, we first need to prepare a high-quality human feedback dataset. The core goal of this dataset is to provide multiple candidate answers (completions) for each given prompt, and to have human annotators carefully evaluate and rank these answers. Through comparing and filtering the answers, we can provide the machine model with a clear reference standard, helping it further learn how to generate outputs that better match human expectations for a given task.

The data can be collected following these steps:

1. Collect initial answers: first, we need to generate multiple answers from an already basically fine-tuned "large model" (often a pretrained model with a certain level of instruction understanding and generation capability) for a set of carefully designed prompts. These answers will serve as the basis for the subsequent human annotation work.


2. Human annotation and evaluation: with multiple candidate answers in hand, we invite professional annotators or crowdsourced annotators to evaluate the quality of each answer. These evaluations are usually based on a series of pre-designed evaluation criteria, such as the accuracy, completeness, contextual relevance, and language fluency of the answer, as well as whether it adheres to ethical and safety guidelines. Comparing and ranking the different answers helps us identify the best and worst answers, thereby forming valuable training data.

3. Data formatting and organization: after annotation is complete, we organize and format the data, usually in structured data formats convenient for computer processing such as JSON, CSV, or others. The dataset must clearly identify each question (prompt), its corresponding multiple answers (completions), and the human annotators' choices among these answers (e.g., the best answer marked "chosen" and the inferior answer "rejected"). These labels can directly serve as the supervised signal for the reward model's learning, making it automatically prefer generating high-quality answers during training.

Below is a simple data example, showing two questions (question) and their corresponding answers and human evaluation results. By comparing the "chosen" and "rejected" fields, we can intuitively see which answer is of higher quality.

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

In the above example, the human annotators judged the answers under the "chosen" field to be of higher quality than the corresponding "rejected" answers in terms of description, accuracy, and informativeness. For example, for the definition of a list, the "chosen" answer explains the features of a list more clearly (ordered, mutable, supports index access), rather than staying at the vague description of "used to store data".


## 7.2.2 Reward Model Training

We can train the reward model with the help of TRL (Transformer Reinforcement Learning), a reinforcement learning framework for large models. TRL is a training framework based on reinforcement learning, designed to guide the model, through human feedback, to generate answers that better match human expectations. In TRL, we treat the reward model as an independent component, used to evaluate the answers generated by the model and to give rewards or penalties based on the evaluation results.
