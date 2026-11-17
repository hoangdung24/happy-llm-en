---
title: "Chapter 8: LLM Reinforcement Learning"
---

In Chapter 4, starting from Reinforcement Learning from Human Feedback (RLHF), we introduced the reward model, Proximal Policy Optimization (PPO), and the basic pipeline of instruction alignment. Chapter 6 further laid the engineering foundations of large language model training, and Chapter 7 gave the model the abilities of planning, memory, and tool calling.

Once an Agent genuinely enters environments such as search and code execution, supervised fine-tuning alone can hardly cover all possible interaction trajectories. The model needs to try actions itself, judge whether a trajectory is valid based on the final outcome, and then update successful experience into the policy. Large language model reinforcement learning has thus gradually moved from "making answers better match preferences" toward "letting the model learn to act in an environment."

This chapter is organized around four interconnected topics. First, we introduce Group Relative Policy Optimization (GRPO), which is widely used in current large language model reinforcement learning, building a complete picture of rollout, reward, advantage, and policy updates; next, we introduce On-Policy Distillation (OPD), observing how the Student continuously receives the Teacher's token-by-token guidance on its own state distribution; finally, using Search-R1 and ReTool as examples, we extend the same training framework to search engines and code interpreters, completing two runnable Agentic RL practices.

<div align='center'>
    <img src="https://raw.githubusercontent.com/datawhalechina/happy-llm/main/docs/images/8-images/8-1-chapter-roadmap.png" alt="Chapter 8 content structure" width="90%">
    <p>Figure 8.1 Structure of Chapter 8</p>
</div>

All examples in this chapter use PyTRIO 0.2.6. PyTRIO connects local data processing, environment interaction, and training control with remote model sampling, forward/backward passes, and weight management. Although the four examples differ in task, the core pipeline can always be summarized as:

1. Use the current policy to generate one trajectory or a group of trajectories;
2. Compute the training signal from outcomes or Teacher feedback;
3. Align the prompt, response, old-policy logprobs, and advantage into a `Datum`;
4. Call `forward_backward()` and `optim_step()` to update the policy;
5. Refresh the sampling client so that the next round of rollouts uses the new weights.

The accompanying code is located in `docs/chapter8`. Install the common dependencies from the repository root:

```bash
uv venv --python 3.13
source .venv/bin/activate
uv pip install -r docs/chapter8/requirements.txt
trio login
```

The models supported by the PyTRIO service are updated continuously; before running formal experiments, you can use the following code to view the current list:

```python
import pytrio as trio

service_client = trio.ServiceClient()
print(service_client.get_supported_models())
```

> **Further reading:** If you want to learn more about Agentic RL algorithms and runnable practices, you can read another open-source project maintained by the author, [agentic-rl-lab](https://github.com/KMnO4-zx/agentic-rl-lab). In addition to the content covered in this chapter, that repository also includes principle breakdowns, training code, and experiment records for OPSD, DAPO, GSPO, ALFWorld, and other algorithms and environments.

> Note: The author chose PyTrio rather than Verl for this chapter's examples, with the aim of lowering the engineering barrier and experimental cost of Agentic RL. Following the minimal trial-run configuration in this chapter, the author strives to keep the base experiment budget of the four examples within 200 RMB, so that readers can personally run through the closed loop of rollout, reward, advantage, and policy update. Actual costs vary with the model, sampling length, number of training steps, and evaluation scale.

## 8.1 GRPO

> **Code source:** This section is based on `01-demo-sync.py` and `02-demo-async.py` in the author's open-source repository [agentic-rl-lab/01-grpo](https://github.com/KMnO4-zx/agentic-rl-lab/tree/main/01-grpo). The main text explains the synchronous version `01-demo-sync.py` section by section; the asynchronous version is preserved as a complete companion implementation.

GRPO was first systematically proposed in DeepSeekMath. It retains PPO's on-policy update idea while using a group of sampled results for the same question to estimate relative advantages, thereby eliminating the need to train a separate Value Model. For math, code, and logic tasks with verifiable answers, this method can directly use outcome rewards and has become one of the most common foundational algorithms in large language model reinforcement learning.

### 8.1.1 From Language Model to Reinforcement Learning Policy

An autoregressive language model receives a prompt $x$ and sequentially generates a token sequence $y=(y_1,y_2,\ldots,y_T)$. From the perspective of probabilistic modeling, the probability of a complete response can be written as:

$$\pi_\theta(y\mid x) = \prod_{t=1}^{T} \pi_\theta(y_t\mid x,y_{\lt t}),$$

where $\pi_\theta$ denotes the language model policy parameterized by $\theta$. Placing the language model into the reinforcement learning framework yields the correspondence shown in Table 8.1.

| Reinforcement learning concept | Meaning in a large language model |
| --- | --- |
| State $s_t$ | The prompt and already generated tokens |
| Action $a_t$ | The next token |
| Policy $\pi_\theta(a_t\mid s_t)$ | The model's probability distribution over the next token |
| Trajectory $\tau$ | A complete response, or a multi-turn interaction including tool calls |
| Environment | Dataset graders, search engines, code interpreters, etc. |
| Reward $R(\tau)$ | Feedback such as answer correctness, format, and tool execution results |

One model generation is usually called a rollout. The goal of training is to raise the probability of actions appearing in high-reward trajectories while lowering the probability of actions appearing in low-reward trajectories. The most intuitive form of the policy gradient is:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau\sim\pi_\theta} \left[ A(\tau) \nabla_\theta \log \pi_\theta(\tau) \right],$$

where $A(\tau)$ is the advantage function. A positive advantage means this trajectory is relatively worth encouraging; a negative advantage means the current policy should reduce similar behavior.

The action space of a large language model is the entire vocabulary, and a single response may contain hundreds to thousands of tokens. During training we must also control the magnitude of change between the old and new policies, preventing a single update from destroying the model's existing capabilities. Therefore, the truly difficult parts usually concentrate on two questions: how to obtain stable advantages, and how to safely use those advantages to update the policy.

### 8.1.2 From PPO to GRPO

PPO typically uses a Value Model to estimate the state value $V_\phi(s_t)$, then combines it with the return to compute the advantage. In large language model settings, the Value Model is often close in scale to the policy model, bringing additional memory, training, and synchronization costs. Value estimation errors are also further propagated into policy updates.

GRPO instead uses relative comparison within a group. For a question $x$, first sample $G$ responses from the old policy $\pi_{\theta_{\mathrm{old}}}$:

$$\{y_1,y_2,\ldots,y_G\} \sim \pi_{\theta_{\mathrm{old}}}(\cdot\mid x).$$

The grader gives rewards $r_1,r_2,\ldots,r_G$ respectively. Classic GRPO normalizes the rewards using the group mean and standard deviation:

$$A_i = \frac{r_i-\mathrm{mean}(r_1,\ldots,r_G)}{\mathrm{std}(r_1,\ldots,r_G)+\varepsilon}.$$

In this way, high-scoring responses to the same question receive positive advantages, and low-scoring responses receive negative advantages. The intrinsic difficulty of the question is offset by the group baseline, allowing the policy to focus more on learning "under the same conditions, which generation approaches are better."

<div align='center'>
    <img src="https://raw.githubusercontent.com/datawhalechina/happy-llm/main/docs/images/8-images/8-2-grpo.png" alt="GRPO group sampling and policy update" width="90%">
    <p>Figure 8.2 GRPO's group sampling and policy update</p>
</div>

For example, sample 4 responses to the same math question, and a rule-based grader yields rewards $[1,0,0,1]$. The group mean is $0.5$; with mean centering only, the 4 advantages are:

$$[0.5,-0.5,-0.5,0.5].$$

The two correct responses are encouraged, and the two incorrect responses are suppressed. To directly demonstrate the group baseline, this chapter's code adopts by default:

$$A_i=r_i-\bar r.$$

Full experiments may also add standard-deviation normalization. The two formulations express the same idea of group-relative comparison, but their gradient scales differ; when comparing experiments, the advantage computation should be fixed.

There is also a degenerate case that is easy to overlook. When all responses in a group are correct, or all are wrong, every $A_i$ equals 0, and this group of data produces no valid policy gradient. If the proportion of degenerate groups in the training logs remains high for a long time, you usually need to adjust question difficulty, sampling temperature, base model capability, or group size.

### 8.1.3 Rewards and Reinforcement Learning with Verifiable Rewards

GRPO only converts in-group rewards into relative advantages; reward quality still determines what the model ultimately learns. Math and code tasks can usually use deterministic rule-based graders, and such training is also called Reinforcement Learning with Verifiable Rewards (RLVR).

This chapter uses GSM8K as an example, requiring the model to write the final numerical answer in `\boxed{}`. The minimal grading logic is as follows:

```python
def extract_boxed(text: str) -> str | None:
    matches = re.findall(r"\\boxed\{([^}]+)\}", text)
    if not matches:
        return None
    return matches[-1].strip()


def normalize_answer(text: str) -> str:
    return text.replace(",", "").strip().rstrip(".")


def grade_answer(response: str, ground_truth: str) -> float:
    answer = extract_boxed(response)
    if answer is None:
        return 0.0
    return 1.0 if normalize_answer(answer) == normalize_answer(ground_truth) else 0.0
```

Rule-based rewards are simple, cheap, and reproducible. However, once a rule has loopholes, the model may learn to exploit them for high scores. When designing a reward, you should check at least the following:

- **Is answer extraction robust**: the same correct answer may appear as an integer, a decimal, a fraction, or a number with commas;
- **Does the format reward outweigh correctness**: format can only serve as an auxiliary constraint; a wrong answer with correct format must not receive the main reward;
- **Are the reference answers reliable**: wrong labels give the model training signals pointing in the opposite direction;
- **Does the grader leak information**: the model should not be able to read the standard answer directly from the prompt or tool output;
- **Is the reward discriminative**: large numbers of all-0 or all-1 groups strip GRPO of its relative comparison signal.

Before formal training, you can first save the base model's rollouts and manually inspect a batch of "response—extracted answer—reward" triples offline. This step often surfaces problems earlier than directly tuning the learning rate.

### 8.1.4 Policy Ratios and the Update Loss

The rollout is generated by the old policy, while the parameter update happens on the current policy. For the $t$-th token of the $i$-th response, define the probability ratio between the new and old policies:

$$\rho_{i,t}(\theta) = \frac{\pi_\theta(y_{i,t}\mid x,y_{i,\lt t})}{\pi_{\theta_{\mathrm{old}}}(y_{i,t}\mid x,y_{i,\lt t})} = \exp\left(\log\pi_\theta-\log\pi_{\theta_{\mathrm{old}}}\right).$$

The most direct importance sampling objective can be written as:

$$J_{\mathrm{IS}}(\theta) = \mathbb{E}_{i,t} \left[\rho_{i,t}(\theta)A_i\right].$$

If the probability ratio strays too far from 1, a few tokens may dominate the gradient. PPO uses a clipped objective to limit the magnitude of a single update:

$$J_{\mathrm{PPO}}(\theta) = \mathbb{E}_{i,t} \left[\min\left(\rho_{i,t}A_i, \mathrm{clip}(\rho_{i,t},1-\epsilon,1+\epsilon)A_i\right)\right].$$

The classic GRPO objective in DeepSeekMath adopts PPO-style clipping and adds a KL constraint relative to a reference policy. This chapter's synchronous `01-demo-sync.py` defaults to PyTRIO's built-in `importance_sampling`, making it easy to observe the data relationships among `old_logprobs`, `advantages`, and the current policy; changing the command-line argument to `--loss-fn ppo` switches to the built-in PPO loss.

Therefore, GRPO and PPO should be understood at different levels:

- GRPO describes how to group-sample the same question and derive advantages from in-group rewards;
- importance sampling or PPO describes how to convert advantages into policy gradients.

### 8.1.5 Implementing the Synchronous GRPO from Scratch

This section breaks down a complete GRPO training script following the execution order of the synchronous version. The accompanying directory provides two independently runnable files:

| File | Execution mode | Purpose |
| --- | --- | --- |
| `01-demo-sync.py` | Synchronous | The teaching mainline, explained section by section in the text |
| `02-demo-async.py` | Asynchronous | Concurrently processes rollouts for different prompts in the same batch |

The two files use the same data, reward, advantage, Datum, and loss. Below we only expand the synchronous version, so readers can first understand a complete update along a linear control flow.

**(1) Define the training configuration and rollout data**

The script keeps only the two policy-update losses PyTRIO directly provides, `importance_sampling` and `ppo`:

```python
LOSS_FNS = ("importance_sampling", "ppo")
```

`GRPOConfig` centrally stores the model, sampling, optimizer, and SwanLab parameters. `RolloutSample` stores all the information a single completion needs for subsequent training:

```python
@dataclass
class RolloutSample:
    """A single sampled result, plus the old-policy logprobs needed to construct importance_sampling."""

    tokens: list[int]
    logprobs: list[float]
    text: str
    reward: float
    advantage: float
```

The `logprobs` here must come from the Student at rollout time. Recomputing after the policy update yields new-policy logprobs, which cannot replace the old-policy probabilities. `parse_args()` converts command-line arguments into a `GRPOConfig` and restricts usage to these two built-in losses via `choices=LOSS_FNS`.

**(2) Load GSM8K and construct the prompt**

GSM8K's standard answer appears after `####`. The model's output is required to put the final number in `\boxed{}`. The code applies the same lightweight normalization to both sides:

```python
def extract_boxed(text: str) -> str | None:
    """Take the last \\boxed{...} as the model's final answer."""
    matches = re.findall(r"\\boxed\{([^}]+)\}", text)
    if not matches:
        return None
    return matches[-1].strip()


def normalize_answer(text: str) -> str:
    """GSM8K answers only get lightweight normalization, so 1,000 and 1000 are not judged different."""
    return text.replace(",", "").strip().rstrip(".")


def grade_answer(response: str, ground_truth: str) -> float:
    """Give 1 when the boxed answer exactly matches the standard answer, otherwise 0."""
    answer = extract_boxed(response)
    if answer is None:
        return 0.0
    return 1.0 if normalize_answer(answer) == normalize_answer(ground_truth) else 0.0


def extract_gsm8k_answer(answer_text: str) -> str:
    """GSM8K's final answer appears after `####`."""
    match = re.search(r"####\s*(.+)", answer_text)
    if match is None:
        raise ValueError(f"No GSM8K final answer found: {answer_text!r}")
    return normalize_answer(match.group(1))
```

`build_prompt()` passes a few-shot example together with the current question through the model's own chat template, then encodes it into tokens:

```python
def build_prompt(tokenizer: Any, question: str) -> list[int]:
    messages = [
        *FEWSHOT_PREFIX,
        {"role": "user", "content": question + QUESTION_SUFFIX},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
    if not prompt_tokens:
        raise ValueError("Prompt tokens are empty")
    return prompt_tokens
```

The training set is loaded via `load_dataset("openai/gsm8k", "main", split="train")`. `pick_batch()` allows the data to wrap around during small trial runs; once `--all-data` is enabled, each training sample is used at most once.

**(3) Sample a group of responses for the same question**

`run_rollout_group()` is the core of the GRPO rollout. One `sample()` call receives a single prompt and returns a group of completions for the same question via `num_samples=group_size`:

```python
def run_rollout_group(
    sampling_client: Any,
    tokenizer: Any,
    prompt_tokens: list[int],
    ground_truth: str,
    sampling_params: trio.SamplingParams,
    group_size: int,
) -> list[RolloutSample]:
    result = sampling_client.sample(
        prompt=trio.ModelInput.from_ints(prompt_tokens),
        num_samples=group_size,
        sampling_params=sampling_params,
        return_text=True,
    ).result()

    rewards: list[float] = []
    raw_samples: list[tuple[list[int], list[float], str]] = []

    for sequence in result.sequences:
        text = sequence.text
        if text is None:
            text = tokenizer.decode(sequence.tokens, skip_special_tokens=True)

        tokens = list(sequence.tokens)
        logprobs = [float(value) for value in sequence.logprobs]
        if len(tokens) != len(logprobs):
            raise ValueError(
                f"Generated token/logprob length mismatch: "
                f"{len(tokens)} != {len(logprobs)}"
            )

        reward = grade_answer(text, ground_truth)
        rewards.append(reward)
        raw_samples.append((tokens, logprobs, text))

    mean_reward = sum(rewards) / len(rewards)
    return [
        RolloutSample(
            tokens=tokens,
            logprobs=logprobs,
            text=text,
            reward=reward,
            advantage=reward - mean_reward,
        )
        for (tokens, logprobs, text), reward
        in zip(raw_samples, rewards, strict=True)
    ]
```

This code accomplishes four things in sequence:

1. Save the completion tokens;
2. Save the old policy's logprob for each completion token at sampling time;
3. Use the rule-based grader to obtain a reward of 0 or 1;
4. Compute `advantage = reward - mean_reward`.

When all responses in a group are correct or all are wrong, every advantage is 0. The main loop records such a group as degenerate and skips it, avoiding submitting Datums with no training signal.

**(4) Construct the right-shift-aligned PyTRIO Datum**

An autoregressive model uses the token at the current position to predict the next token. Let the prompt length be $m$ and the completion length be $T$; the four arrays the script constructs all have length $m+T-1$:

```python
def build_grpo_datum(
    prompt_tokens: list[int],
    sample: RolloutSample,
) -> trio.Datum:
    if not sample.tokens:
        raise ValueError("Cannot train on an empty completion")

    observation_len = len(prompt_tokens) - 1
    input_tokens = prompt_tokens + sample.tokens[:-1]
    target_tokens = [0] * observation_len + sample.tokens
    padded_logprobs = [0.0] * observation_len + sample.logprobs
    padded_advantages = (
        [0.0] * observation_len
        + [sample.advantage] * len(sample.tokens)
    )

    if not (
        len(input_tokens)
        == len(target_tokens)
        == len(padded_logprobs)
        == len(padded_advantages)
    ):
        raise ValueError("GRPO datum fields must have the same token length")

    return trio.Datum(
        model_input=trio.ModelInput.from_ints(input_tokens),
        loss_fn_inputs={
            "target_tokens": np.asarray(target_tokens, dtype=np.int64),
            "logprobs": np.asarray(padded_logprobs, dtype=np.float32),
            "advantages": np.asarray(padded_advantages, dtype=np.float32),
        },
    )
```

The prompt region only provides context, so `target_tokens`, `logprobs`, and `advantages` use 0 as placeholders there. The completion region holds the real target tokens, the old-policy logprobs, and the group advantage shared by the entire response.

Taking 3 prompt tokens and 2 completion tokens as an example:

```text
prompt       = [x1, x2, x3]
completion   = [y1, y2]

model_input  = [x1, x2, x3, y1]
target       = [ 0,  0, y1, y2]
old_logprob  = [ 0,  0, l1, l2]
advantage    = [ 0,  0,  A,  A]
```

The position corresponding to the last prompt token `x3` is where prediction of `y1` begins, so the prompt mask has length `len(prompt_tokens) - 1`.

**(5) Choose importance sampling or PPO**

`importance_sampling` and `ppo` reuse the same batch of GRPO Datums; you only switch `loss_fn`. Both losses are built into PyTRIO:

```python
fwd_bwd_future = training_client.forward_backward(
    datums,
    loss_fn=config.loss_fn,
)
```

When `importance_sampling` is chosen, training directly multiplies the new/old policy probability ratio by the advantage. When `ppo` is chosen, PyTRIO applies PPO clipping on the same data. GRPO's rollout, reward, and group advantage construction remain unchanged.

**(6) Assemble one round of synchronous training**

`main()` first creates the training client, sampling parameters, and optimizer:

```python
service_client = trio.ServiceClient()
training_client = service_client.create_lora_training_client(
    base_model=config.base_model,
    rank=config.lora_rank,
)
tokenizer = training_client.get_tokenizer()

sampling_params = trio.SamplingParams(
    max_tokens=config.max_tokens,
    temperature=config.temperature,
    top_p=config.top_p,
    stop=get_stop_sequences(tokenizer),
)
adam_params = trio.AdamParams(
    learning_rate=config.learning_rate,
    beta1=config.beta1,
    beta2=config.beta2,
)
```

Each step first creates a new sampler from the current training weights, then performs rollouts in question order:

```python
for step in range(effective_steps):
    batch_rows = pick_batch(
        train_data,
        step,
        config.batch_size,
        config.all_data,
    )
    sampling_client = (
        training_client.save_weights_and_get_sampling_client()
    )

    datums: list[trio.Datum] = []
    prompt_mean_rewards: list[float] = []
    rollout_lengths: list[int] = []
    n_degenerate = 0

    for row in tqdm(
        batch_rows,
        desc=f"GRPO step {step}",
        unit="prompt",
    ):
        prompt_tokens = build_prompt(tokenizer, row["question"])
        ground_truth = extract_gsm8k_answer(row["answer"])
        rollout_samples = run_rollout_group(
            sampling_client=sampling_client,
            tokenizer=tokenizer,
            prompt_tokens=prompt_tokens,
            ground_truth=ground_truth,
            sampling_params=sampling_params,
            group_size=config.group_size,
        )

        rewards = [sample.reward for sample in rollout_samples]
        prompt_mean_rewards.append(sum(rewards) / len(rewards))
        rollout_lengths.extend(
            len(sample.tokens) for sample in rollout_samples
        )

        if all(
            sample.advantage == 0.0
            for sample in rollout_samples
        ):
            n_degenerate += 1
            continue

        for sample in rollout_samples:
            datums.append(
                build_grpo_datum(prompt_tokens, sample)
            )
```

The sampler refresh position reflects the on-policy constraint: after step $k$'s update completes, step $k+1$ regenerates trajectories with the new weights. After submitting the forward-backward and optimization tasks, the synchronous version explicitly waits for the results:

```python
optim_future = training_client.optim_step(adam_params)
fwd_bwd_result = fwd_bwd_future.result()
optim_future.result()
loss_metrics = dict(fwd_bwd_result.metrics)
```

At the end, the script reports the batch average reward, degenerate group ratio, average generation length, number of effective training tokens, and `loss_mean`, and saves LoRA weights usable for sampling:

```python
final_weights = training_client.save_weights_for_sampler(
    name=run_name
).result()
print(
    f"Saved weights name: {run_name}, "
    f"path: {final_weights.path}"
)
```

### 8.1.6 Running the Synchronous and Asynchronous Versions

First log in to TRIO, then run a one-step trial of the synchronous version from the repository root:

```bash
trio login

python docs/chapter8/grpo/01-demo-sync.py \
    --steps 1 \
    --batch-size 1 \
    --group-size 4 \
    --max-tokens 512 \
    --loss-fn importance_sampling \
    --swanlab-mode disabled
```

The asynchronous version uses exactly the same training parameters:

```bash
python docs/chapter8/grpo/02-demo-async.py \
    --steps 1 \
    --batch-size 4 \
    --group-size 4 \
    --max-tokens 512 \
    --loss-fn importance_sampling \
    --swanlab-mode disabled
```

The asynchronous version's algorithm and data are unchanged; its execution changes as follows:

| Training step | Synchronous version | Asynchronous version |
| --- | --- | --- |
| Single-question rollout | `sample(...).result()` | `await sample_async(...)` |
| Multiple questions in a batch | Processed one by one | Concurrently processed with `asyncio.gather()` |
| Refreshing the sampler | Synchronous return | `await save_weights_and_get_sampling_client_async()` |
| Forward-backward | `forward_backward()` then `.result()` | `forward_backward_async()` then `await` the future again |
| Optimizer | `optim_step()` then `.result()` | `optim_step_async()` then `await` the future again |
| Program entry | `main(config)` | `asyncio.run(main(config))` |

The main text uses the synchronous version to build algorithmic intuition. The asynchronous version suits formal experiments where the batch contains multiple independent prompts; it reduces serial waiting time by issuing remote requests concurrently.

During training, prioritize observing the metrics the code actually records:

| Metric | Meaning | Anomalous behavior |
| --- | --- | --- |
| `reward` | Mean of the per-prompt average accuracy across the batch | If it stays flat, check the reward, data difficulty, and learning rate |
| `frac_degenerate` | Proportion of all-correct or all-wrong groups | When too high, effective relative advantage is insufficient |
| `rollout/avg_gen_len` | Average completion token count | When near `max_tokens`, check truncation |
| `train_tokens` | Completion tokens with nonzero advantage | When 0, the current step has no training signal |
| `loss_mean` | Main loss of the current policy update | If it fluctuates sharply, check the learning rate and probability ratios |

At this point we have the first training mainline of this chapter: the current policy group-samples, a rule-based environment gives outcome rewards, in-group comparison produces advantages, and PyTRIO completes the policy update. The upcoming OPD keeps the same on-policy data flow while replacing the training signal with the Teacher's per-token feedback.

## 8.2 On-Policy Distillation

> **Code source:** This section is based on `01-demo-sync.py` and `02-demo-async.py` in the author's open-source repository [agentic-rl-lab/02-opd/general-opd](https://github.com/KMnO4-zx/agentic-rl-lab/tree/main/02-opd/general-opd). The main text explains the synchronous version `01-demo-sync.py` section by section; the asynchronous version is preserved as a complete companion implementation.

Capability distillation for large language models usually involves a stronger Teacher and a smaller Student. Traditional knowledge distillation uses pre-collected data, having the Student fit the Teacher's probability distribution on those fixed samples. As the Student keeps changing during training, a gap may appear between the fixed data and the states the Student will actually visit.

On-Policy Distillation (OPD) has the Student generate responses with its current policy, and then has the Teacher score, token by token, the trajectories the Student has already traversed. In this way, every round of supervision lies on the Student's current state distribution. The Teacher does not need to regenerate a set of answers, nor participate in backpropagation; it only provides dense probability feedback for the Student's actions.

In recent years, OPD has gradually become an important direction in large language model training. It can be used for reasoning-ability transfer, capability recovery, and continual learning, and it combines naturally with the reinforcement learning data flow: Student rollouts correspond to on-policy trajectories, Teacher logprobs correspond to token-by-token training signals, and after the Student updates, it generates the next batch of trajectories.

### 8.2.1 From Offline Distillation to On-Policy Distillation

Let the Teacher be $\pi_T$ and the Student be $\pi_\theta$. Offline distillation first fixes a dataset $\mathcal D$, then minimizes the distribution discrepancy on the states within it:

$$\mathcal L_{\mathrm{offline}} = \mathbb E_{s\sim\mathcal D} \left[D\bigl(\pi_T(\cdot\mid s),\pi_\theta(\cdot\mid s)\bigr)\right].$$

This approach is simple to implement, and its effectiveness depends on whether the dataset covers the states the Student encounters at inference time. Suppose the fixed data mostly contains the Teacher's high-quality trajectories, while the Student produces an erroneous token early during real generation; the states it subsequently enters may never appear in the distillation data, and the Teacher's guidance cannot cover this out-of-distribution trajectory.

OPD changes the state sampling to the current Student:

$$y\sim\pi_{\theta_{\mathrm{old}}}(\cdot\mid x),$$

and then has the Teacher compute the conditional probability of the same Student completion:

$$\log\pi_T(y_t\mid x,y_{\lt t}).$$

One training iteration can be divided into three steps:

1. **Student rollout**: the Student generates responses using its current weights;
2. **Teacher feedback**: the Teacher computes logprobs on exactly the same prompt and completion;
3. **Student update**: construct a token-by-token signal from the probability difference between the two and update the Student.

<div align='center'>
    <img src="https://raw.githubusercontent.com/datawhalechina/happy-llm/main/docs/images/8-images/8-3-opd.png" alt="OPD obtains Teacher feedback on the Student's own trajectories" width="90%">
    <p>Figure 8.3 OPD obtains Teacher feedback on the Student's own trajectories</p>
</div>

In Figure 8.3, the Teacher only performs forward computation. The Student's sampling actions also do not participate in backpropagation; gradients flow from the constructed training targets back to the current Student parameters. This stop-gradient boundary lets OPD directly reuse existing rollout and policy-optimization infrastructure.

### 8.2.2 Reverse KL and Token-by-Token Training Signals

OPD can choose different distribution divergences. This chapter uses reverse KL:

$$D_{\mathrm{KL}}\left(\pi_\theta\,\|\,\pi_T\right) = \mathbb E_{y\sim\pi_\theta} \left[\log\pi_\theta(y\mid x)-\log\pi_T(y\mid x)\right].$$

For the $t$-th token actually sampled by the Student, we obtain a Monte Carlo estimate:

$$d_t = \log\pi_{\theta_{\mathrm{old}}}(y_t\mid x,y_{\lt t})-\log\pi_T(y_t\mid x,y_{\lt t}).$$

When $d_t>0$, the Student's preference for this token is higher than the Teacher's; when $d_t<0$, the Teacher assigns a higher probability. To minimize reverse KL, this chapter writes the training advantage as:

$$A_t=-\beta d_t,$$

where $\beta$ controls the distillation strength. In this way, every completion token receives an independent training signal, whose density far exceeds that of giving a single 0/1 reward only at the end of the response.

Reverse KL has a strong mode-seeking property. The Student preferentially concentrates on regions where the Teacher already assigns high probability, which suits compressing the Teacher's high-confidence behaviors into a small model. At the same time, the capability gap between Teacher and Student, their reasoning styles, and tokenizer compatibility all affect the distillation result. A larger Teacher is only a candidate condition; the Teacher must also provide valuable and learnable new signals on the Student's current trajectories.

Table 8.2 compares the data sources of GRPO and this chapter's OPD example.

| Item | GRPO | OPD |
| --- | --- | --- |
| Trajectory source | Current policy | Current Student |
| Feedback source | Environment or rule-based grader | Teacher's probability distribution |
| Signal granularity | Usually a whole-trajectory reward | Per completion token |
| Requires standard answers | Usually requires verifiable outcomes | Prompt-only data suffices |
| Requires an extra model | Can omit the Value Model | Requires Teacher forward computation |
| Main risks | Reward hacking, degenerate groups | Teacher incompatibility, unsuitable capability gap |

### 8.2.3 The Teacher Must Evaluate the Same Student Trajectory

The key data constraint of OPD is that the Teacher and the Student must evaluate the same target tokens under the same prefix. Having the Teacher generate its own response and then letting the Student learn that response would fall back into the common synthetic-data distillation pipeline.

In PyTRIO, you can first concatenate the prompt and the Student completion, then call the Teacher's `compute_logprobs()`:

```python
def completion_teacher_logprobs(
    teacher_client,
    prompt_ids: list[int],
    completion_ids: list[int],
):
    all_ids = prompt_ids + completion_ids
    all_logprobs = teacher_client.compute_logprobs(
        trio.ModelInput.from_ints(all_ids)
    ).result()

    start = len(prompt_ids)
    end = start + len(completion_ids)
    return [float(x) for x in all_logprobs[start:end]]
```

Different SDKs may define the position of logprobs with an off-by-one difference, so you cannot guess the slicing by array length alone. The most reliable approach is to construct a very short known sequence, confirm whether each returned position represents "the logprob of the current token" or "the logprob of the next token," and then finish the boundary testing between prompt and completion. This chapter's code has already been aligned with PyTRIO 0.2.6's return convention.

The Teacher and Student also need to share a compatible tokenizer. If the same group of token ids represents different strings in the two models, the probability difference has no correct semantics even when the array lengths match. Before formal training, verify:

```python
text = "A short tokenizer alignment test."
student_ids = student_tokenizer.encode(text, add_special_tokens=False)
teacher_ids = teacher_tokenizer.encode(text, add_special_tokens=False)

assert student_ids == teacher_ids
```

For models in the same family but with different chat templates, the Student tokenizer should render the full prompt once, and the Teacher should then directly evaluate the corresponding token sequence. Do not apply the two chat templates separately; otherwise the context each side sees has already changed.

### 8.2.4 Implementing the Synchronous OPD from Scratch

The OPD companion directory likewise provides two independent scripts, synchronous and asynchronous:

| File | Execution mode | Purpose |
| --- | --- | --- |
| `01-demo-sync.py` | Synchronous | The teaching mainline, explained section by section in the text |
| `02-demo-async.py` | Asynchronous | Concurrently processes Student rollouts and Teacher logprob requests |

Following the code order of `01-demo-sync.py`, we now implement a complete round of OPD from a DeepMath prompt to the Student update.

**(1) Prepare parameters and the DeepMath data**

The script downloads data by default to `datasets/DeepMath-103K` under the current code directory:

```python
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR / "datasets" / "DeepMath-103K"
DEEPMATH_SHARDS = 10
```

The arguments of `parse_args()` fall into four groups:

- Data arguments: `dataset_repo`, `dataset_dir`, `num_shards`, `sample_size`;
- Model arguments: the Student base model, LoRA rank, the Teacher base model or Teacher weight path;
- Training arguments: steps, batch, group, sampling length, the KL coefficient, and optimizer parameters;
- Logging arguments: the SwanLab project, experiment name, and log mode.

DeepMath-103K on ModelScope contains 10 parquet shards. `shard_name()` generates the remote file names, `modelscope_file_url()` generates the download URLs, and `download_if_needed()` writes to a temporary file first and then atomically replaces it, avoiding incomplete shards after interruption.

The data-loading function hands the local parquet files to Hugging Face Datasets, keeping only the `question` field needed for training:

```python
def load_deepmath(args: argparse.Namespace):
    shard_paths = []

    for index in range(args.num_shards):
        remote_path = shard_name(index)
        local_path = args.dataset_dir / remote_path
        download_if_needed(
            modelscope_file_url(
                args.dataset_repo,
                args.dataset_revision,
                remote_path,
            ),
            local_path,
            args.force_download,
        )
        shard_paths.append(str(local_path))

    dataset = load_dataset(
        "parquet",
        data_files=shard_paths,
        split="train",
        cache_dir=str(
            args.dataset_dir / ".datasets_cache"
        ),
    )

    if "question" not in dataset.column_names:
        raise ValueError(
            "DeepMath must contain 'question', "
            f"got {dataset.column_names}"
        )

    dataset = dataset.shuffle(seed=args.seed)
    if args.sample_size > 0:
        dataset = dataset.select(
            range(min(args.sample_size, len(dataset)))
        )
    return dataset
```

`--num-shards 1` suits the first integration test, while `--num-shards 10` covers the full dataset. `--sample-size` controls how many samples actually enter the prompt pool after loading.

**(2) Render questions into Student prompts**

OPD uses prompt-only data and does not read DeepMath's reference reasoning. `build_prompt()` hands the question and the output requirement to the Student tokenizer:

```python
def build_prompt(
    tokenizer,
    question: str,
    suffix: str,
    enable_thinking: bool,
) -> list[int]:
    content = (
        question.strip()
        if not suffix
        else f"{question.strip()}\n\n{suffix}"
    )
    messages = [{"role": "user", "content": content}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    return tokenizer.encode(
        prompt,
        add_special_tokens=False,
    )
```

The prompt is rendered only once. The Teacher then directly evaluates the same group of tokens, keeping both sides' context exactly identical.

**(3) Create the Student and the Teacher**

The Student uses a LoRA training client; the Teacher uses a sampling client that only does forward computation:

```python
service_client = trio.ServiceClient()

training_client = (
    service_client.create_lora_training_client(
        base_model=args.base_model,
        rank=args.lora_rank,
        seed=args.seed,
    )
)
tokenizer = training_client.get_tokenizer()

teacher_client = service_client.create_sampling_client(
    base_model=(
        args.teacher_base_model
        or args.base_model
    ),
    model_path=args.teacher_model_path,
)
```

`teacher_model_path` can point to already-saved sampler weights. The Teacher never calls `forward_backward()` or `optim_step()`; its parameters stay fixed throughout the OPD process.

**(4) Have the Teacher evaluate the Student's identical trajectory**

The Student rollout returns completion tokens and old-policy logprobs. The Teacher needs to see `prompt_ids + completion_ids`, and then the completion region is sliced out of the returned result:

```python
def completion_teacher_logprobs(
    teacher_client,
    prompt_ids: list[int],
    completion_ids: list[int],
):
    all_ids = prompt_ids + completion_ids
    all_logprobs = teacher_client.compute_logprobs(
        trio.ModelInput.from_ints(all_ids)
    ).result()

    completion_logprobs = all_logprobs[
        len(prompt_ids):
    ]

    if (
        len(completion_logprobs)
        != len(completion_ids)
        or any(
            value is None
            for value in completion_logprobs
        )
    ):
        raise ValueError(
            "Invalid teacher logprobs "
            "for completion tokens"
        )

    return [
        float(value)
        for value in completion_logprobs
    ]
```

The length check and `None` check protect token alignment. Only after passing the checks can the $t$-th Student completion token, the Student's old logprob, and the Teacher's logprob form the same training position.

**(5) Construct the token-by-token OPD Datum**

OPD and GRPO use the same `importance_sampling` schema; the difference lies in the advantage. GRPO replicates a single group advantage across the whole completion, while OPD writes an independent reverse-KL advantage for each completion token:

```python
def build_opd_datum(
    prompt_ids: list[int],
    completion_ids: list[int],
    old_logprobs,
    advantages,
):
    prompt_loss_len = len(prompt_ids) - 1

    input_ids = (
        prompt_ids
        + completion_ids[:-1]
    )
    target_ids = (
        [0] * prompt_loss_len
        + completion_ids
    )
    padded_logprobs = (
        [0.0] * prompt_loss_len
        + list(old_logprobs)
    )
    padded_advantages = (
        [0.0] * prompt_loss_len
        + list(advantages)
    )

    if not (
        len(input_ids)
        == len(target_ids)
        == len(padded_logprobs)
        == len(padded_advantages)
    ):
        raise ValueError(
            "OPD datum fields must "
            "have the same length"
        )

    return trio.Datum(
        model_input=trio.ModelInput.from_ints(
            input_ids
        ),
        loss_fn_inputs={
            "target_tokens": np.asarray(
                target_ids,
                dtype=np.int64,
            ),
            "logprobs": np.asarray(
                padded_logprobs,
                dtype=np.float32,
            ),
            "advantages": np.asarray(
                padded_advantages,
                dtype=np.float32,
            ),
        },
    )
```

The right-shift rule is exactly the same as GRPO's. The three loss inputs in the prompt region continue to use 0 as placeholders, while the completion region holds, one to one, the Student tokens, the Student's old logprobs, and each token's OPD advantage.

**(6) Generate Student trajectories and compute reverse KL**

At the start of each step, the code refreshes the Student sampler according to `sampler_refresh_steps`. The default is 1, so every step uses the latest Student weights:

```python
if (
    student_sampler is None
    or step % args.sampler_refresh_steps == 0
):
    student_sampler = (
        training_client
        .save_weights_and_get_sampling_client()
    )
```

The synchronous version processes the current batch's questions one by one. For each prompt, the Student first generates `group_size` trajectories, and the Teacher then scores each nonempty completion:

```python
result = student_sampler.sample(
    prompt=trio.ModelInput.from_ints(
        prompt_ids
    ),
    num_samples=args.group_size,
    sampling_params=sampling_params,
    return_text=False,
).result()

for sequence in result.sequences:
    completion_ids = sequence.tokens
    if not completion_ids:
        continue

    student_lps = [
        float(value)
        for value in sequence.logprobs
    ]
    teacher_lps = completion_teacher_logprobs(
        teacher_client,
        prompt_ids,
        completion_ids,
    )

    reverse_kl = (
        np.asarray(student_lps)
        - np.asarray(teacher_lps)
    )
    advantages = (
        -args.kl_penalty_coef
        * reverse_kl
    )

    datums.append(
        build_opd_datum(
            prompt_ids,
            completion_ids,
            student_lps,
            advantages,
        )
    )
    reverse_kls.extend(reverse_kl.tolist())
    completion_token_counts.append(
        len(completion_ids)
    )
```

Here `group_size` widens the coverage of Student states under the same prompt. Each trajectory independently receives the Teacher's signal, without GRPO-style in-group centering.

**(7) Update the Student, record metrics, and save weights**

After the whole batch is collected, the code performs one forward-backward pass and one optimizer update:

```python
if not datums:
    raise RuntimeError(
        "No OPD datums were built"
    )

fwd_bwd = training_client.forward_backward(
    datums,
    loss_fn="importance_sampling",
)
optim = training_client.optim_step(adam)

fwd_bwd_result = fwd_bwd.result()
optim.result()
```

The `tokens/s` of an OPD step divides the total number of completion tokens by the elapsed time of the whole step. That time simultaneously includes Student sampling, Teacher logprobs, the Student's forward-backward pass, the optimizer update, and any necessary sampler refresh:

```python
completion_tokens_total = int(
    sum(completion_token_counts)
)
metrics = {
    "data/datums": len(datums),
    "data/completion_tokens_mean": float(
        np.mean(completion_token_counts)
    ),
    "data/completion_tokens_total": (
        completion_tokens_total
    ),
    "data/completion_tokens_per_second": (
        completion_tokens_total
        / step_elapsed_time
    ),
    "opd/reverse_kl_mean": float(
        np.mean(reverse_kls)
    ),
    "opd/reverse_kl_std": float(
        np.std(reverse_kls)
    ),
    "train/learning_rate": args.learning_rate,
    "time/step_elapsed_time": step_elapsed_time,
}
```

After training finishes, the Student's sampler weights are saved:

```python
save_result = (
    training_client
    .save_weights_for_sampler(
        args.save_weights_name
    )
    .result()
)
print(f"Saved weights: {save_result.path}")
```

### 8.2.5 Running the Synchronous and Asynchronous Versions

A one-step trial of the synchronous version automatically downloads one DeepMath shard and draws at most 20 prompts from it:

```bash
python docs/chapter8/opd/01-demo-sync.py \
    --steps 1 \
    --batch-size 1 \
    --group-size 1 \
    --max-tokens 512 \
    --sample-size 20 \
    --num-shards 1 \
    --swanlab-mode disabled
```

The asynchronous version uses the same algorithm and parameters:

```bash
python docs/chapter8/opd/02-demo-async.py \
    --steps 1 \
    --batch-size 4 \
    --group-size 2 \
    --max-tokens 512 \
    --sample-size 20 \
    --num-shards 1 \
    --swanlab-mode disabled
```

Asynchronous OPD exploits two layers of concurrency at once:

| Training step | Synchronous version | Asynchronous version |
| --- | --- | --- |
| Creating Student / Teacher | Synchronous client interfaces | `create_lora_training_client_async()` and `create_sampling_client_async()` |
| Student rollout | Calls `sample().result()` per prompt | Concurrent `sample_async()` within the batch |
| Teacher scoring | Calls `compute_logprobs().result()` per completion | Concurrent `compute_logprobs_async()` within the same prompt |
| Assembling the batch | Appends Datums sequentially | Two layers of `asyncio.gather()` followed by unified assembly |
| Student update | Synchronous future + `.result()` | Submit via the async API, then `await` the future again |
| Program entry | `main(args)` | `asyncio.run(main(args))` |

The main text only expands the synchronous version, because it directly presents the causal order "Student samples → Teacher scores → reverse KL → Student updates." The asynchronous version keeps the same token alignment, advantages, and optimization objective.

The example defaults to `Qwen/Qwen3.5-4B` as the Student and `Qwen/Qwen3.6-27B` as the Teacher. The actually available models should follow the supported list currently returned by the PyTRIO service. During training, watch in parallel:

- `opd/reverse_kl_mean`: the average difference between the Student's sampled tokens and the Teacher's probabilities;
- `opd/reverse_kl_std`: the dispersion of the distillation signal across different tokens;
- `data/completion_tokens_total`: the total number of tokens the Teacher scored in this step;
- `data/completion_tokens_per_second`: the effective completion-token throughput of the whole OPD step;
- `trainer/*`: training metrics returned by PyTRIO;
- Independent task evaluations and general capability evaluations: to confirm the real capability changes after distillation.

A falling reverse KL indicates that the Student's sampling distribution is approaching the Teacher's. Task accuracy still needs to be confirmed with a fixed held-out evaluation. When loss fluctuates or capability drops, check in order: token alignment, Teacher quality, the KL coefficient, the learning rate, the sampler refresh frequency, and the completion truncation ratio.

So far, GRPO and OPD have both taken place in the single-turn structure of "the model generates a response, an external module provides the training signal." The next section's Search-R1 further changes the shape of the trajectory: before finishing its response, the model may pause generation, call the search environment, read an observation, and then continue deciding its next action.
## 8.3 Search-R1

> **Code source:** This section is based on the complete multi-file implementation in the author's open-source repository [agentic-rl-lab/03-search-r1](https://github.com/KMnO4-zx/agentic-rl-lab/tree/main/03-search-r1). The main text follows the real call chain of `prepare_data.py`, `protocol.py`, `search.py`, `rollout.py`, `reward.py`, `train.py`, `eval.py`, and `analyse.py`.

A large language model stores a vast amount of knowledge internally, but that knowledge is limited by the time range of its training data, its parameter capacity, and its long-tail coverage. Retrieval-Augmented Generation (RAG) usually has the system retrieve documents in advance and then hand the results to the model to answer. Search-R1 hands the search decisions to the model: the model can judge when to search, what to search, whether to keep searching based on new evidence, and when to stop and give an answer.

This change makes a single response a multi-turn trajectory:

$$\tau=(x,a_1,o_1,a_2,o_2,\ldots,a_T),$$

where $a_t$ is the search call or final answer generated by the model, and $o_t$ is the observation returned by the search environment. The model only receives the outcome reward after completing the entire trajectory, and GRPO then distinguishes successful search paths from failed ones.

The companion implementation is not an isolated training script but an execution chain composed of 9 files. When reading the code, you can first grasp the following mainline:

```text
prepare_data.py / data.py
        ↓ reads questions and reference answers
protocol.py
        ↓ constructs the prompt and parses search calls
search.py
        ↓ performs retrieval and returns observations
rollout.py
        ↓ generates complete multi-turn trajectories, rewards, and group advantages
train.py
        ↓ observation mask, Datum, importance_sampling, optim_step
eval.py / analyse.py
        ↓ fixed-environment evaluation and result aggregation
```

What is actually trained are the Assistant tokens generated by the language model. The search service handled by `search.py` belongs to the environment, `protocol.py` connects model actions and environment observations into continuous trajectories, `rollout.py` orchestrates the two, and only `train.py` finally converts trajectories into `Datum` objects PyTRIO can update. Below, the code is expanded along this real call chain.

### 8.3.1 From Fixed Retrieval to Autonomous Search

Search-R1's training samples contain only the question and reference answers, no manually written search terms, and no standard search trajectories. `prepare_data.py` unifies the raw fields of NQ and HotpotQA into the following JSONL structure:

```json
{"id":"...","question":"...","answers":["..."],"data_source":"nq"}
```

The cleaning logic keeps only the fields usable for outcome rewards:

```python
def normalize_row(row: dict[str, Any]) -> dict[str, Any] | None:
    question = str(row.get("question") or "").strip()
    answers = extract_answers(row)
    if not question or not answers:
        return None
    return {
        "id": str(row.get("id") or ""),
        "question": question,
        "answers": answers,
        "data_source": str(
            row.get("data_source") or "unknown"
        ),
    }
```

`data.py` then converts each row into a `SearchExample`:

```python
@dataclass(frozen=True)
class SearchExample:
    id: str
    question: str
    answers: list[str]
    data_source: str
```

This data boundary matters. The search queries, the number of searches, and how evidence is used must all be explored by the current policy itself during rollout; if the training set already provides fixed retrieval trajectories, the task becomes closer to trajectory imitation rather than the Search-R1 discussed in this section.

The differences between fixed RAG and a Search Agent can be seen in Table 8.3.

| Aspect | Fixed RAG | Search-R1 |
| --- | --- | --- |
| Whether to retrieve | Decided by an external pipeline | Decided by the model based on the current state |
| Search terms | Preset or generated once | Can be rewritten multiple times based on observations |
| Number of retrieval rounds | Usually fixed | Decided by the policy, constrained by a maximum |
| Training target | The answer model or the retriever | A unified policy for search actions and final answers |
| Reward | Usually supervised answer loss | Outcome reward on the complete trajectory |

For example, the question "What country was the author of The Little Prince born in?" requires first identifying the author, then querying the author's birthplace. The model can form the following trajectory:

```text
User: What country was the author of The Little Prince born in?

Assistant -> search("The Little Prince author")
Tool: Antoine de Saint-Exupéry ...

Assistant -> search("Antoine de Saint-Exupéry birthplace country")
Tool: He was born in Lyon, France ...

Assistant: Answer: France
```

The second search term depends on the first observation, so it cannot be determined all at once before the rollout begins. The training code needs to maintain environment state and parse the model's actions after each assistant turn.

<div align='center'>
    <img src="https://raw.githubusercontent.com/datawhalechina/happy-llm/main/docs/images/8-images/8-4-search-r1.png" alt="Search-R1's multi-turn search and observation mask" width="90%">
    <p>Figure 8.4 Search-R1's multi-turn search and observation mask</p>
</div>

The Search-R1 paper uses text tags such as `<search>... </search>` and `<information>... </information>` to represent actions and observations. This chapter uses the Qwen3.5-4B model with the model's native function calling template:

```python
SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "Search the web for evidence.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
            },
            "required": ["query"],
        },
    },
}
```

The two protocols carry the same reinforcement learning semantics: the Assistant outputs trainable actions, and the search backend returns environment observations. Using the native template reduces conflicts between the model's base capabilities and the tool format, and also preserves structured tool messages.

After defining the tool, the model's output must also be strictly parsed into three states: a valid search, a final answer, or an invalid output. The parser in `protocol.py` requires that a turn contain at most one complete `search` call, and that no final answer be smuggled in after the tool call:

```python
def parse_assistant(text: str) -> ParsedAssistant:
    matches = list(TOOL_CALL_PATTERN.finditer(text))
    if not matches:
        kind = (
            "invalid" if "<tool_call>" in text else "answer"
        )
        return ParsedAssistant(
            kind=kind,
            content=text.strip(),
        )

    if len(matches) != 1 or text[matches[0].end():].strip():
        return ParsedAssistant(
            kind="invalid",
            content=text.strip(),
        )

    query = matches[0].group(1).strip()
    if not query or "<" in query or ">" in query:
        return ParsedAssistant(
            kind="invalid",
            content=text.strip(),
        )

    content = text[:matches[0].start()].strip()
    return ParsedAssistant(
        kind="tool",
        content=content,
        query=query,
    )
```

No separate "stop" action is defined here. As long as no valid tool call appears in the output, the turn ends as the final answer; `reward.py` then checks whether it conforms to `Answer: <short answer>`. Output containing an incomplete `<tool_call>` is marked `invalid`, likewise ending the trajectory and receiving a format penalty.

### 8.3.2 How the Search Backend Becomes an Environment Observation

`search.py` provides three backends: DeepSeek Search, Wikipedia, and Zhihu. The rollout does not depend directly on any HTTP response format, only on the unified `SearchResult`:

```python
@dataclass(frozen=True)
class SearchItem:
    title: str
    content: str
    source: str | None = None
    url: str | None = None


@dataclass(frozen=True)
class SearchResult:
    ok: bool
    items: list[SearchItem]
    latency: float
    status: int | None = None
    error: str | None = None
```

The training entry creates a backend based on command-line arguments, and the subsequent state machine always calls the same `search(query)` interface:

```python
def create_search_client(
    backend: str,
    env_path: str | Path | None = None,
    *,
    model: str = "deepseek-v4-flash",
    timeout: float | None = None,
) -> SearchClient:
    resolved_timeout = resolve_search_timeout(
        backend, timeout
    )
    if backend == "deepseek":
        return DeepSeekSearchClient.from_env(
            env_path,
            model=model,
            timeout=resolved_timeout,
        )
    if backend == "wikipedia":
        return WikipediaSearchClient(
            timeout=resolved_timeout
        )
    if backend == "zhihu":
        return ZhihuSearchClient.from_env(
            env_path,
            timeout=resolved_timeout,
        )
    raise ValueError(f"不支持的搜索后端: {backend}")  # (unsupported search backend: {backend})
```

Taking the key-free Wikipedia backend as an example, the actual retrieval happens in `_request()`. A single API request performs a full-text search and fetches the titles, body extracts, and URLs of the top three pages:

```python
def _request(
    self,
    query: str,
    started: float,
) -> SearchResult:
    params = urllib.parse.urlencode(
        {
            "action": "query",
            "format": "json",
            "formatversion": 2,
            "generator": "search",
            "gsrsearch": query,
            "gsrlimit": 3,
            "prop": "extracts|info",
            "explaintext": 1,
            "exintro": 1,
            "exchars": 1200,
            "inprop": "url",
            "redirects": 1,
            "utf8": 1,
        }
    )
    request = urllib.request.Request(
        f"{WIKIPEDIA_SEARCH_ENDPOINT}?{params}",
        headers={
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "User-Agent": WIKIPEDIA_USER_AGENT,
        },
    )
    with urllib.request.urlopen(
        request,
        timeout=self.timeout,
    ) as response:
        body = response.read()
        if response.headers.get(
            "Content-Encoding"
        ) == "gzip":
            body = gzip.decompress(body)
        payload = json.loads(body.decode("utf-8"))
        pages = payload.get("query", {}).get(
            "pages", []
        )
        if not isinstance(pages, list):
            raise TypeError(
                "Wikipedia 搜索响应 pages 不是列表"  # (Wikipedia search response pages is not a list)
            )
        if any(
            not isinstance(page, dict)
            for page in pages
        ):
            raise TypeError(
                "Wikipedia 搜索响应 page 不是对象"  # (Wikipedia search response page is not an object)
            )
        ordered_pages = sorted(
            pages,
            key=lambda page: int(
                page.get("index", 1_000_000)
            ),
        )
        items = [
            SearchItem(
                title=str(
                    page.get("title") or "Untitled"
                ).strip(),
                content=str(
                    page.get("extract") or ""
                ).strip(),
                source="Wikipedia",
                url=str(
                    page.get("fullurl") or ""
                ).strip(),
            )
            for page in ordered_pages
            if str(page.get("extract") or "").strip()
        ]
        return SearchResult(
            ok=True,
            items=items,
            latency=time.perf_counter() - started,
            status=response.status,
        )
```

The outer `search()` also handles request rate limiting, timeout, and bounded retries on 429 and 5xx, and accumulates success rate and latency. If retries still fail, it does not let the whole rollout raise an exception and exit; instead it returns `SearchResult(ok=False, error=...)`. `rollout.py` feeds the error text to the model as an observation too, so the model can continue answering or rewrite the query.

The results of different backends are ultimately formatted into readable evidence:

```text
[1] Title: ...
    Content: ...
    Source: ...
    URL: ...
```

Returning only titles or links is usually insufficient, because what the model really needs is `content` that can support the next step of reasoning. `max_tool_response_tokens` and `max_trajectory_tokens` again limit the evidence length before observations enter the trajectory.

### 8.3.3 The Multi-Turn Rollout State Machine

Single-turn GRPO needs only one `sample()` call. In Search-R1, each trajectory may go through a different number of search calls, and the trainer must repeatedly execute "generate—parse—call the environment—append the observation."

This chapter stores a trajectory as a `Trajectory`:

```python
@dataclass
class AssistantTurn:
    prompt_tokens: list[int]
    completion_tokens: list[int]
    logprobs: list[float]
    text: str


@dataclass
class Trajectory:
    example: SearchExample
    group_index: int
    messages: list[dict[str, Any]]
    next_prompt_tokens: list[int] | None = None
    question_index: int = 0
    turns: list[AssistantTurn] = field(default_factory=list)
    search_calls: int = 0
    final_text: str = ""
    reward: float = -0.1
    advantage: float = 0.0
    valid_format: bool = False
    exact_match: bool = False
    done: bool = False
```

Here, each `AssistantTurn` records the complete prompt before this turn's sampling, the model's generated tokens, and the corresponding old-policy logprobs. Search results live in `messages` and in the next turn's prompt, but they carry no rollout logprobs, because those tokens come from the environment.

What actually sends requests to PyTRIO is `sample_requests_async()`. Within the same round, different questions or already-diverged trajectories can be sampled concurrently:

```python
async def sample_requests_async(
    sampling_client: Any,
    requests: list[SampleRequest],
    config: RolloutConfig,
    tokenizer: Any,
) -> list[Any]:
    tasks = []
    for request in requests:
        params = trio.SamplingParams(
            max_tokens=request.max_tokens,
            seed=request.seed,
            stop=stop_sequences(tokenizer),
            temperature=config.temperature,
            top_p=config.top_p,
        )
        tasks.append(
            sampling_client.sample_async(
                prompt=trio.ModelInput.from_ints(
                    request.prompt_tokens
                ),
                num_samples=request.num_samples,
                sampling_params=params,
                return_text=True,
            )
        )
    return list(await asyncio.gather(*tasks))
```

After sampling returns, `consume_assistant()` first saves the raw tokens and old logprobs, then calls the protocol parser from the previous section. A valid search becomes a `PendingSearch`; a final answer, an invalid format, or an exhausted budget all end the current trajectory:

```python
def consume_assistant(
    trajectory: Trajectory,
    prompt_tokens: list[int],
    sequence: Any,
    tokenizer: Any,
    config: RolloutConfig,
) -> PendingSearch | None:
    tokens, logprobs, text = read_sequence(
        sequence, tokenizer
    )
    trajectory.turns.append(
        AssistantTurn(
            prompt_tokens,
            tokens,
            logprobs,
            text,
        )
    )
    parsed = parse_assistant(text)

    can_search = (
        parsed.kind == "tool"
        and trajectory.search_calls
        < config.max_search_calls
        and len(trajectory.turns)
        < config.max_assistant_turns
    )
    if not can_search:
        trajectory.messages.append(
            {"role": "assistant", "content": text}
        )
        trajectory.final_text = text
        trajectory.done = True
        return None

    call_id = (
        f"search-{trajectory.question_index}-"
        f"{trajectory.group_index}-"
        f"{trajectory.search_calls + 1}"
    )
    messages_before_assistant = list(
        trajectory.messages
    )
    trajectory.messages.append(
        {"role": "assistant", "content": text}
    )
    return PendingSearch(
        trajectory=trajectory,
        messages_before_assistant=(
            messages_before_assistant
        ),
        assistant_text=text,
        prompt_tokens=prompt_tokens,
        completion_tokens=tokens,
        call_id=call_id,
        query=parsed.query or "",
    )
```

`resolve_searches()` uses a thread pool to concurrently call the search backend for different trajectories, then pairs results with pending trajectories in the original order:

```python
def resolve_searches(
    pending_searches: list[PendingSearch],
    search_client: SearchClient,
    tokenizer: Any,
    config: RolloutConfig,
) -> int:
    if not pending_searches:
        return 0
    workers = min(
        config.search_concurrency,
        len(pending_searches),
    )
    with ThreadPoolExecutor(
        max_workers=workers
    ) as pool:
        results = list(
            pool.map(
                search_client.search,
                [
                    pending.query
                    for pending in pending_searches
                ],
            )
        )
    return sum(
        finish_search(
            pending,
            result,
            tokenizer,
            config,
        )
        for pending, result in zip(
            pending_searches,
            results,
            strict=True,
        )
    )
```

Each result is then written back to its original trajectory by `finish_search()`:

```python
def finish_search(
    pending: PendingSearch,
    result: SearchResult,
    tokenizer: Any,
    config: RolloutConfig,
) -> bool:
    fitted = fit_tool_content(
        tokenizer,
        pending.messages_before_assistant,
        pending.assistant_text,
        pending.prompt_tokens,
        pending.completion_tokens,
        pending.call_id,
        result,
        config,
    )
    if fitted is None:
        pending.trajectory.final_text = (
            pending.assistant_text
        )
        pending.trajectory.done = True
        return True

    content, next_prompt_tokens = fitted
    pending.trajectory.messages.append(
        tool_message(pending.call_id, content)
    )
    pending.trajectory.next_prompt_tokens = (
        next_prompt_tokens
    )
    pending.trajectory.search_calls += 1
    return False
```

`fit_tool_content()` adds search evidence item by item. As soon as an observation would exceed the per-tool budget or the whole-trajectory budget, it stops adding further evidence; if not even one result fits, the trajectory ends directly. This way, how much text the search service returns and how much text the model actually sees are two clearly separated boundaries.

A group of GRPO trajectories shares exactly the same prompt in the first turn. The code generates `group_size` branches with a single request:

```python
first_requests: list[SampleRequest] = []
for index, trajectory in enumerate(roots):
    request = make_request(
        tokenizer,
        trajectory,
        index,
        config.group_size,
        config.seed + index,
        config,
    )
    if request:
        first_requests.append(request)
```

Each sequence in the sampling response is deep-copied from the root trajectory into an independent branch, and its search actions are then parsed separately:

```python
responses = asyncio.run(
    sample_requests_async(
        sampling_client,
        first_requests,
        config,
        tokenizer,
    )
)
pending_searches: list[PendingSearch] = []
for request, response in zip(
    first_requests,
    responses,
    strict=True,
):
    root = roots[request.trajectory_index]
    if len(response.sequences) != config.group_size:
        raise ValueError(
            "首轮采样数量与 group_size 不一致"  # (first-round sample count inconsistent with group_size)
        )

    for group_index, sequence in enumerate(
        response.sequences
    ):
        branch = copy.deepcopy(root)
        branch.group_index = group_index
        pending = consume_assistant(
            branch,
            request.prompt_tokens,
            sequence,
            tokenizer,
            config,
        )
        trajectories.append(branch)
        if pending is not None:
            pending_searches.append(pending)

resolve_searches(
    pending_searches,
    search_client,
    tokenizer,
    config,
)
```

After different branches execute their searches, the observations and past actions already differ. In subsequent turns, each unfinished trajectory gets `num_samples=1` individually:

```python
while any(not item.done for item in trajectories):
    requests: list[SampleRequest] = []
    for index, trajectory in enumerate(trajectories):
        if trajectory.done:
            continue
        request = make_request(
            tokenizer,
            trajectory,
            index,
            1,
            (
                config.seed
                + index
                + len(trajectory.turns) * 10_000
            ),
            config,
        )
        if request:
            requests.append(request)

    if not requests:
        break
    responses = asyncio.run(
        sample_requests_async(
            sampling_client,
            requests,
            config,
            tokenizer,
        )
    )

    pending_searches = []
    for request, response in zip(
        requests, responses, strict=True
    ):
        if len(response.sequences) != 1:
            raise ValueError(
                "后续轮次必须只采样一个分支"  # (subsequent turns must sample exactly one branch)
            )
        trajectory = trajectories[
            request.trajectory_index
        ]
        pending = consume_assistant(
            trajectory,
            request.prompt_tokens,
            response.sequences[0],
            tokenizer,
            config,
        )
        if pending is not None:
            pending_searches.append(pending)

    resolve_searches(
        pending_searches,
        search_client,
        tokenizer,
        config,
    )
```

This "shared root node, then branch" implementation satisfies two conditions at once: same-question trajectories can be compared within the GRPO group, and each branch can keep acting on its own search results. Multiple PyTRIO sampling requests and search requests can all execute concurrently, reducing the waiting time caused by multi-turn environment interaction.

The state machine must also set explicit boundaries:

- `max_search_calls`: the maximum number of searches a trajectory may call;
- `max_assistant_turns`: the maximum number of Assistant messages that may be generated;
- `max_assistant_tokens`: the token cap for a single turn's action;
- `max_tool_response_tokens`: the token cap for a single observation;
- `max_trajectory_tokens`: the total token cap for the entire context;
- Search timeout and concurrency: limiting external-service waiting time and concurrency pressure.

These limits both control training cost and define the range of environment an Agent can access. Evaluation must keep the same configuration; otherwise a larger search budget by itself may raise answer accuracy.

### 8.3.4 Tool Protocol and Token Continuity

The most common problem in multi-turn training is that the tokens after re-rendering the chat template do not match the tokens actually sampled. The tokenizer may clean whitespace, normalize tool-call text, or automatically append turn-end tokens. If the training stage re-encodes the model's text, the old logprobs can no longer correspond one to one with the target tokens.

This chapter follows two principles:

1. Assistant actions always keep the raw tokens returned by the sampler;
2. Only newly added end tokens and tool observation tokens are computed via the chat template.

The full responsibility of `build_next_prompt()` is to extract from the chat template only the incremental segment "Assistant end tokens + new observation," and then append it after the real completion returned by the sampler:

```python
def build_next_prompt(
    tokenizer: Any,
    messages_before_assistant: list[dict[str, Any]],
    assistant_text: str,
    previous_prompt_tokens: list[int],
    completion_tokens: list[int],
    next_tool_message: dict[str, Any],
) -> list[int]:
    canonical_prompt = build_prompt(
        tokenizer,
        messages_before_assistant,
    )

    empty_assistant_end = _render_chat(
        tokenizer,
        [
            *messages_before_assistant,
            {"role": "assistant", "content": ""},
        ],
        add_generation_prompt=False,
    )
    if empty_assistant_end[
        :len(canonical_prompt)
    ] != canonical_prompt:
        raise ValueError(
            "chat template 无法从空 assistant 提取结束边界"  # (chat template cannot extract the end boundary from an empty assistant)
        )
    assistant_closing_tokens = empty_assistant_end[
        len(canonical_prompt):
    ]

    assistant_message = {
        "role": "assistant",
        "content": assistant_text,
    }
    messages_with_assistant = [
        *messages_before_assistant,
        assistant_message,
    ]
    canonical_assistant_end = _render_chat(
        tokenizer,
        messages_with_assistant,
        add_generation_prompt=False,
    )
    canonical_next_prompt = build_prompt(
        tokenizer,
        [*messages_with_assistant, next_tool_message],
    )
    if canonical_next_prompt[
        :len(canonical_assistant_end)
    ] != canonical_assistant_end:
        raise ValueError(
            "加入 tool observation 后 chat template 改写了历史消息"  # (after adding the tool observation, the chat template rewrote historical messages)
        )

    observation_tokens = canonical_next_prompt[
        len(canonical_assistant_end):
    ]
    overlap = _suffix_prefix_overlap(
        completion_tokens,
        assistant_closing_tokens,
    )
    return [
        *previous_prompt_tokens,
        *completion_tokens,
        *assistant_closing_tokens[overlap:],
        *observation_tokens,
    ]
```

`overlap` handles the case where the sampler has already returned part of the Assistant's end tokens, avoiding duplicated token supplementation. `messages` serves protocol record-keeping and debugging, while `next_prompt_tokens` is the model input actually submitted to the sampler in the next turn; the two must not be conflated.

When constructing the next turn's training data, this is verified again:

```python
if turn.prompt_tokens[:len(full_tokens)] != full_tokens:
    raise ValueError(
        "下一轮 prompt 不是已有轨迹的前缀扩展"  # (the next turn's prompt is not a prefix extension of the existing trajectory)
    )
```

These two layers of validation catch template rewriting, duplicated end tokens, and observation splicing errors early. For Agentic RL, token continuity is part of algorithmic correctness; a trajectory can look reasonable at the text level yet still produce wrong gradients due to misaligned token boundaries.
### 8.3.5 Reward, In-Group Advantage, and the Observation Mask

This chapter uses the short-answer tasks of NQ and HotpotQA. The model must finally output exactly one line:

```text
Answer: <short answer>
```

The rule-based reward has three tiers:

$$R(\tau)=\begin{cases}1, & \text{format correct and the answer exactly matches},\\ 0, & \text{format correct but the answer is wrong},\\ -0.1, & \text{the final format is invalid}.\end{cases}$$

`reward.py` first extracts the single line containing the final answer, then unifies case, punctuation, English articles, and whitespace:

```python
def normalize_answer(text: str) -> str:
    lowered = text.lower()
    without_punctuation = "".join(
        char
        for char in lowered
        if not unicodedata.category(char).startswith("P")
    )
    without_articles = ARTICLE_PATTERN.sub(
        " ", without_punctuation
    )
    return " ".join(without_articles.split())


def extract_answer(text: str) -> str | None:
    matches = ANSWER_PATTERN.findall(text)
    if len(matches) != 1:
        return None
    answer = matches[0].strip()
    return answer or None


def score_answer(
    text: str,
    references: list[str],
) -> RewardResult:
    answer = extract_answer(text)
    if answer is None:
        return RewardResult(
            -0.1, False, False, None
        )
    normalized = normalize_answer(answer)
    exact_match = any(
        normalized == normalize_answer(reference)
        for reference in references
    )
    return RewardResult(
        float(exact_match),
        True,
        exact_match,
        answer,
    )
```

This code evaluates only the final task outcome, adding no extra score for "how many searches were made" or "whether the query looks good." For the $G$ complete trajectories of the same question, the GRPO in-group baseline is still used:

$$A_i=R(\tau_i)-\frac{1}{G}\sum_{j=1}^{G}R(\tau_j).$$

The Search-R1 paper discusses optimization methods including both PPO and GRPO. This chapter chooses GRPO's same-question grouping scheme, so that Search-R1 uses the same advantage construction method as Section 8.1.

`rollout_batch()` must wait until every trajectory in the group has finished before executing the following function:

```python
def assign_group_advantages(
    trajectories: list[Trajectory],
) -> int:
    groups: dict[int, list[Trajectory]] = {}
    for trajectory in trajectories:
        groups.setdefault(
            trajectory.question_index, []
        ).append(trajectory)

    degenerate = 0
    for group in groups.values():
        mean_reward = sum(
            item.reward for item in group
        ) / len(group)
        for item in group:
            item.advantage = (
                item.reward - mean_reward
            )
        if all(
            item.advantage == 0.0 for item in group
        ):
            degenerate += 1
    return degenerate
```

When all rewards in a group are identical, every advantage is 0, and this group produces no valid gradient. The training code skips such trajectories and records the degenerate group ratio. The in-group mean cannot be computed after splitting into micro-batches; otherwise the comparison baseline has already been altered.

This advantage is assigned to the tokens generated by the Assistant in every turn of the trajectory. Search observations only constitute the state needed for the next decision; they are not actions taken by the policy, so the observation tokens' logprobs and advantages are both set to 0:

```python
full_tokens.extend(delta_observation)
old_logprobs_by_token.extend(
    [0.0] * len(delta_observation)
)
advantages_by_token.extend(
    [0.0] * len(delta_observation)
)

full_tokens.extend(turn.completion_tokens)
old_logprobs_by_token.extend(turn.logprobs)
advantages_by_token.extend(
    [trajectory.advantage]
    * len(turn.completion_tokens)
)
```

This observation mask serves two purposes. First, the trainer does not require the model to predict the text returned by the search engine; second, the observations remain in the context, so subsequent Assistant tokens can condition on that evidence.

The search calls and the final answer share one trajectory-level advantage. If the final answer is correct, the successful query planning, search terms, and answer tokens are all encouraged; if the final answer is wrong, the entire chain of actions is suppressed. This is exactly why outcome rewards can train tool-use policies.

### 8.3.6 Constructing the Multi-Turn PyTRIO Datum

`build_datum()` in `train.py` brings together the token continuity and observation mask from the preceding sections. Here is the complete core implementation:

```python
def build_datum(
    trajectory: Trajectory,
) -> TrainingDatum:
    if not trajectory.turns:
        raise ValueError(
            "不能用没有 assistant turn 的轨迹构造训练 Datum"  # (cannot construct a training Datum from a trajectory without assistant turns)
        )

    full_tokens: list[int] = []
    old_logprobs_by_token: list[float] = []
    advantages_by_token: list[float] = []
    assistant_token_count = 0

    for turn_index, turn in enumerate(
        trajectory.turns
    ):
        if len(turn.completion_tokens) != len(
            turn.logprobs
        ):
            raise ValueError(
                f"第 {turn_index + 1} 个 assistant turn "
                "的 token 与 logprob 长度不一致"  # (the tokens and logprobs of assistant turn {turn_index + 1} have inconsistent lengths)
            )

        if turn_index == 0:
            delta_observation = turn.prompt_tokens
        elif turn.prompt_tokens[
            :len(full_tokens)
        ] == full_tokens:
            delta_observation = turn.prompt_tokens[
                len(full_tokens):
            ]
        else:
            raise ValueError(
                f"第 {turn_index + 1} 个 assistant turn "
                "的 prompt 不是已有轨迹的前缀扩展，"
                "无法安全对齐采样 logprob"  # (the prompt of assistant turn {turn_index + 1} is not a prefix extension of the existing trajectory; sampled logprobs cannot be safely aligned)
            )

        full_tokens.extend(delta_observation)
        full_tokens.extend(turn.completion_tokens)
        old_logprobs_by_token.extend(
            [0.0] * len(delta_observation)
        )
        old_logprobs_by_token.extend(turn.logprobs)
        advantages_by_token.extend(
            [0.0] * len(delta_observation)
        )
        advantages_by_token.extend(
            [trajectory.advantage]
            * len(turn.completion_tokens)
        )
        assistant_token_count += len(
            turn.completion_tokens
        )

    if assistant_token_count == 0:
        raise ValueError(
            "不能用没有 assistant token 的轨迹构造训练 Datum"  # (cannot construct a training Datum from a trajectory without assistant tokens)
        )
    if not (
        len(full_tokens)
        == len(old_logprobs_by_token)
        == len(advantages_by_token)
    ):
        raise ValueError(
            "完整轨迹的 token、logprob 和 advantage 长度不一致"  # (the full trajectory's tokens, logprobs, and advantages have inconsistent lengths)
        )

    input_tokens = full_tokens[:-1]
    target_tokens = full_tokens[1:]
    old_logprobs = old_logprobs_by_token[1:]
    advantages = advantages_by_token[1:]
    if not (
        len(input_tokens)
        == len(target_tokens)
        == len(old_logprobs)
        == len(advantages)
    ):
        raise ValueError(
            "Datum 的 input、target、logprobs 和 advantages 长度不一致"  # (the Datum's input, target, logprobs, and advantages have inconsistent lengths)
        )
    if len(input_tokens) > MAX_TRAIN_CONTEXT_TOKENS:
        raise ValueError(
            f"Datum 超过 {MAX_TRAIN_CONTEXT_TOKENS} token"  # (Datum exceeds {MAX_TRAIN_CONTEXT_TOKENS} tokens)
        )

    datum = trio.Datum(
        model_input=trio.ModelInput.from_ints(
            input_tokens
        ),
        loss_fn_inputs={
            "target_tokens": np.asarray(
                target_tokens,
                dtype=np.int64,
            ),
            "logprobs": np.asarray(
                old_logprobs,
                dtype=np.float32,
            ),
            "advantages": np.asarray(
                advantages,
                dtype=np.float32,
            ),
        },
    )
    return TrainingDatum(datum, len(input_tokens))
```

In the first turn, `delta_observation` is the system prompt, tool definitions, and the user question; in subsequent turns, it is the previous turn's end tokens, the search results, and the new Assistant prefix. They all keep real target tokens, but their old logprobs and advantages are 0. Only the Assistant tokens actually generated by the sampler carry rollout old logprobs and the trajectory advantage.

The complete group's rewards must all be computed before the correct in-group mean can be obtained. You cannot update the policy as soon as a single trajectory ends, because the rewards of the other branches in the same group are still unknown.

Multi-turn trajectories vary greatly in length, and putting all data into one padded batch wastes many tokens. The companion code packs `Datum` objects into multiple micro-batches by sample length, and re-weights the loss by each batch's share of trajectories out of the full rollout trajectory count. This keeps the gradient semantics of the global sample average while controlling per-batch padding size.

### 8.3.7 From Rollout to One Parameter Update

The outer loop of `train.py` strings the preceding modules into a complete training step. The key order must not be swapped:

```python
batch = take_batch(
    examples,
    step * args.questions_per_batch,
    args.questions_per_batch,
)

sampling_client = (
    training_client
    .save_weights_and_get_sampling_client()
)

trajectories = rollout_batch(
    sampling_client=sampling_client,
    tokenizer=tokenizer,
    search_client=search_client,
    examples=batch,
    config=rollout_config,
)

datums = build_training_datums(trajectories)
micro_batches = pack_micro_batches(datums)

trainer_results = []
for micro_batch in micro_batches:
    weighted_datums = (
        weight_micro_batch_for_global_mean(
            micro_batch,
            total_samples=len(trajectories),
        )
    )
    result = training_client.forward_backward(
        weighted_datums,
        loss_fn="importance_sampling",
    ).result()
    trainer_results.append(result)

if micro_batches:
    training_client.optim_step(
        adam_params
    ).result()
```

First, `save_weights_and_get_sampling_client()` exports the current LoRA weights at the start of each step, so the old logprobs come from the current policy before this update. Second, `rollout_batch()` has already completed the whole group's rewards and advantages internally, and `build_training_datums()` skips trajectories whose advantages are all 0. Third, all micro-batches only accumulate gradients, and `optim_step()` is called exactly once for the whole logical batch at the end.

PyTRIO averages the samples within one `forward_backward()` call. If the $k$-th micro-batch has $n_k$ trajectories while the full rollout batch has $N$ trajectories, the code first multiplies that batch's advantages by $n_k/N$:

$$\sum_k\frac{n_k}{N}\operatorname{mean}(\mathcal L_k)=\operatorname{mean}(\mathcal L_{\mathrm{global}}).$$

The corresponding implementation only re-wraps the `Datum` objects; it does not modify the target tokens or the old logprobs:

```python
micro_batch_weight = np.float32(
    len(micro_batch) / total_samples
)
weighted_datums = []
for item in micro_batch:
    loss_inputs = item.datum.loss_fn_inputs
    weighted_datums.append(
        trio.Datum(
            model_input=item.datum.model_input,
            loss_fn_inputs={
                "target_tokens": (
                    loss_inputs["target_tokens"]
                    .to_numpy()
                ),
                "logprobs": (
                    loss_inputs["logprobs"]
                    .to_numpy()
                ),
                "advantages": (
                    loss_inputs["advantages"]
                    .to_numpy()
                    * micro_batch_weight
                ),
            },
        )
    )
```

Therefore, dynamic batch splitting only changes the padding and memory usage at execution time, not the sample-averaged gradient of the logical batch.

### 8.3.8 Running, Evaluating, and Experimental Boundaries

The Search-R1 code is split by the responsibilities of data, protocol, environment, rollout, and training:

| File | Role |
| --- | --- |
| `prepare_data.py` | Prepares the NQ and HotpotQA data |
| `data.py` | Loads the unified JSONL samples |
| `protocol.py` | Defines the search tool and the multi-turn message protocol |
| `search.py` | Wraps the search backends and unifies the result format |
| `rollout.py` | Runs the multi-turn search state machine |
| `reward.py` | Extracts the short answer and computes the rule-based reward |
| `train.py` | Builds the observation mask and micro-batches, and updates the policy |
| `eval.py` | Evaluates the Base Model or a checkpoint in the same environment |
| `analyse.py` | Aggregates evaluation outputs |

The evaluation does not write a separate simplified search logic; it reuses the same `rollout_batch()`. The main difference is that each question keeps only one trajectory, with no need for the training stage's in-group comparison:

```python
config = RolloutConfig(
    group_size=1,
    max_search_calls=args.max_search_calls,
    max_assistant_turns=args.max_assistant_turns,
    max_trajectory_tokens=(
        args.max_trajectory_tokens
    ),
    max_assistant_tokens=args.max_assistant_tokens,
    max_tool_response_tokens=(
        args.max_tool_response_tokens
    ),
    search_concurrency=args.search_concurrency,
    temperature=args.temperature,
    top_p=args.top_p,
    seed=args.seed,
)

batch_trajectories = rollout_batch(
    sampling_client,
    tokenizer,
    search_client,
    batch,
    config,
)
```

When `model_path` is left empty, `eval.py` creates a Base Model sampler; when given the path returned by `save_weights_for_sampler()`, it evaluates the trained checkpoint. Both kinds of model still use the same protocol, search backend, budget, question set, and answer grader.

First prepare the data:

```bash
python docs/chapter8/search-r1/prepare_data.py
```

The Wikipedia backend needs no API key and suits validating the complete pipeline:

```bash
python docs/chapter8/search-r1/train.py \
    --max-steps 1 \
    --questions-per-batch 1 \
    --group-size 4 \
    --max-search-calls 2 \
    --search-backend wikipedia \
    --search-concurrency 3 \
    --swanlab-mode disabled
```

Evaluate the Base Model with the same Wikipedia environment:

```bash
python docs/chapter8/search-r1/eval.py \
    --search-backend wikipedia \
    --batch-size 8 \
    --output docs/chapter8/search-r1/eval_result/base.jsonl
```

When evaluating the trained sampler weights, keep all other parameters unchanged and only add the checkpoint path:

```bash
python docs/chapter8/search-r1/eval.py \
    --search-backend wikipedia \
    --batch-size 8 \
    --model-path 'trio://runxxxxxxxxxx' \
    --output docs/chapter8/search-r1/eval_result/checkpoint.jsonl
```

After confirming the pipeline, gradually scale up the number of questions, group size, search count, and trajectory budget. Experiments should record at least:

- Exact match of the final answer;
- Proportion of valid answer formats;
- Average number of search calls and the zero-search ratio;
- Search failure rate, timeout rate, and observation truncation rate;
- Average trajectory length and effective training tokens per step;
- Degenerate group ratio;
- Time spent in the rollout, search, and training stages respectively.

Search results have external dependencies. Online pages get updated, and the ranking of search services may change. When comparing the Base Model with a trained checkpoint, fix the question set, search backend, search budget, sampling parameters, and evaluation time window. If you need strictly reproducible paper-style experiments, fix the retrieval corpus and index as a read-only snapshot.

This chapter only updates the language model policy, keeping the search backend fixed. If the retriever itself also participates in learning, you would need to additionally define a retrieval action space, index versions, and cross-module credit assignment, which is beyond the scope of this section's example.

Search-R1 shows how to extend GRPO's in-group relative advantages to multi-turn search trajectories containing external observations. The next section's ReTool keeps the same multi-turn rollout and observation mask, swaps the environment from a search engine to a code interpreter, and uses PyTRIO's built-in PPO to update the policy.

## 8.4 ReTool

> **Code source:** This section is based on the complete multi-file implementation in the author's open-source repository [agentic-rl-lab/05-retool](https://github.com/KMnO4-zx/agentic-rl-lab/tree/main/05-retool). The main text follows the real call chain of `prepare_data.py`, `protocol.py`, `sandbox.py`, `rollout.py`, `reward.py`, `train.py`, `eval.py`, and `analysis.py`.

Search tools help the model acquire external knowledge; a code interpreter helps the model perform precise computation, symbolic manipulation, and enumeration-based verification. The model can already call Python tools under the guidance of prompts or supervised data, but "being able to call" is not the same as "knowing when to call." Some questions are faster by mental arithmetic, some suit a short program, and some code executions fail and need fixing based on the error message.

ReTool uses reinforcement learning to train this strategic tool-use ability. The model autonomously decides whether to call the code interpreter, what code to write, how to read the execution results, and when to give the final answer. The environment provides only an outcome reward based on the final math answer, and the tool decisions in successful trajectories are reinforced along with the whole trajectory.

> **Safety notice:** The accompanying `sandbox.py` is only a resource-limited local subprocess, not a trusted security sandbox. Model-generated code can still read files accessible to the current account, inherit environment variables, and access the network. Real training should run inside a disposable container, a low-privilege virtual machine, or a dedicated sandbox service, with PyTRIO, SwanLab, SSH, and cloud service credentials removed.

ReTool's code pipeline is structurally similar to Search-R1's, but the environment changes from a search service to a Python process:

```text
prepare_data.py / data.py
        ↓ reads math questions and reference answers
protocol.py
        ↓ parses code_interpreter calls
sandbox.py
        ↓ executes code, returning stdout / stderr / timeout
rollout.py
        ↓ appends execution results and continues generation
reward.py
        ↓ checks the last \boxed{}
train.py
        ↓ feedback mask, PPO, optim_step
eval.py / analysis.py
        ↓ side-by-side evaluation of text-only and ReTool
```

Below we again explain along the actual function calls, rather than only looking at the text format of tool calls.

### 8.4.1 From Tool-Call Format to Tool-Use Policy

A typical ReTool trajectory looks like this:

```text
User: 计算满足条件的整数解数量。 (Compute the number of integer solutions satisfying the condition.)

Assistant -> code_interpreter(
    "count = ...\nprint(count)"
)
Tool: 37

Assistant: 根据枚举结果，答案为 \boxed{37} (Based on the enumeration result, the answer is \boxed{37})
```

If the code has errors, the model can also use the traceback to self-correct:

```text
Assistant -> code_interpreter("print(sum(values))")
Tool: NameError: name 'values' is not defined

Assistant -> code_interpreter(
    "values = [...]\nprint(sum(values))"
)
Tool: 128

Assistant: \boxed{128}
```

From a reinforcement learning perspective, the code, execution errors, numerical results, and final answer together constitute the trajectory. The code interpreter is the environment, the tool calls and natural-language answers generated by the Assistant are all actions, and stdout or stderr is the observation.

<div align='center'>
    <img src="https://raw.githubusercontent.com/datawhalechina/happy-llm/main/docs/images/8-images/8-5-retool.png" alt="ReTool's multi-turn code execution and outcome reward" width="90%">
    <p>Figure 8.5 ReTool's multi-turn code execution and outcome reward</p>
</div>

ReTool does not need to award separate points for "having called a tool." The final answer's reward is propagated through the trajectory-level advantage back to the earlier code actions. This way, only tool usage that genuinely helps the task succeed receives positive signal, while redundant or misleading calls are suppressed along failed trajectories.

### 8.4.2 Cold-Start, Training Data, and the Tool Protocol

The original ReTool contains two stages:

1. **Cold-start SFT**: using trajectories with code calls and execution feedback, first let the model master the protocol and basic interaction forms;
2. **Reinforcement Learning**: the model rolls out autonomously, learning when and how to use tools through the final answer's reward.

Cold-start's role is to make the initial policy produce a certain proportion of valid tool calls. If the base model cannot output the tool protocol at all, early trajectories almost all fail, and GRPO can hardly obtain meaningful exploration signal from in-group comparison.

This chapter's code chooses Qwen3.5, which already has native function calling capability, and directly demonstrates the second stage with PyTRIO training, so no dedicated cold-start SFT pipeline is included. This is an important boundary between the teaching implementation and the original paper's setup. When switching to a base model lacking tool-calling ability, you should first prepare SFT data, verify that the valid-call rate reaches a trainable level, and only then start RL.

The RL data comes from DAPO-Math-17k. The raw questions are wrapped in a unified template requiring `Answer: $Answer`, which conflicts with the `\boxed{}` format required in this section, so `prepare_data.py` first strips the outer template and then keeps only the question and reference answer:

```python
def normalize_row(
    index: int,
    row: dict[str, Any],
) -> dict[str, Any] | None:
    prompt = row.get("prompt")
    if not isinstance(prompt, list) or not prompt:
        return None

    question = strip_dapo_template(
        str(prompt[0].get("content") or "")
    )
    reward_model = row.get("reward_model") or {}
    ground_truth = (
        reward_model.get("ground_truth")
        if isinstance(reward_model, dict)
        else None
    )
    if isinstance(ground_truth, list):
        ground_truth = (
            ground_truth[0] if ground_truth else None
        )
    answer = str(ground_truth or "").strip()
    if not question or not answer:
        return None

    return {
        "id": str(
            row.get("extra_info", {}).get("index")
            or index
        ),
        "question": question,
        "answer": answer,
        "data_source": str(
            row.get("data_source") or "dapo_math"
        ),
    }
```

As with Search-R1, the training samples contain neither standard code nor tool-call counts. The policy must explore by itself whether to call the interpreter and what code to generate.

The code interpreter uses a structured tool definition:

```python
CODE_TOOL = {
    "type": "function",
    "function": {
        "name": "code_interpreter",
        "description": (
            "Run Python code and return printed output. "
            "Each execution starts fresh."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string"},
            },
            "required": ["code"],
        },
    },
}
```

`protocol.py` likewise parses tool calls strictly. Python code can contain comparison operators, quotes, and multi-line text, so here only the call count, call position, and whether the code is empty are validated:

```python
def parse_assistant(text: str) -> ParsedAssistant:
    matches = list(TOOL_CALL_PATTERN.finditer(text))
    if not matches:
        kind = (
            "invalid" if "<tool_call>" in text else "answer"
        )
        return ParsedAssistant(
            kind=kind,
            content=text.strip(),
        )

    if len(matches) != 1 or text[matches[0].end():].strip():
        return ParsedAssistant(
            kind="invalid",
            content=text.strip(),
        )

    code = matches[0].group(1).strip()
    if not code:
        return ParsedAssistant(
            kind="invalid",
            content=text.strip(),
        )

    content = text[:matches[0].start()].strip()
    return ParsedAssistant(
        kind="tool",
        content=content,
        code=code,
    )
```

Valid output has only three possible outcomes: `tool` enters the code execution environment, `answer` ends the trajectory and receives the outcome reward, and `invalid` ends the trajectory and is treated as a wrong answer. The system makes no guess-based repair of tool calls that are nearly but not fully well-formed; otherwise the definition of an action in the training environment becomes unstable.

The rollout state machine is similar to Search-R1's, except the environment action changes from `search(query)` to `code_interpreter(code)`. Every trajectory is constrained by the following budgets:

- Maximum number of code calls;
- Maximum number of Assistant turns;
- Token counts for a single turn's generation and execution result;
- Total token count of the complete trajectory;
- Duration of a single execution;
- Concurrency of the local executor.

The same code call does not preserve Python variables or process state. The model must redefine the required data in every call, which makes the trajectory semantics clearer and reduces implicit dependencies between executions.

### 8.4.3 How Execution Results Reconnect to the Real Token Trajectory

ReTool likewise cannot decode the Assistant text and re-tokenize it. `protocol.py` uses a placeholder Assistant to extract the template's closing tokens, then appends the observation tokens corresponding to the actual execution result after the real completion:

```python
def build_next_prompt(
    tokenizer: Any,
    messages_before_assistant: list[dict[str, Any]],
    previous_prompt_tokens: list[int],
    completion_tokens: list[int],
    next_tool_message: dict[str, Any],
) -> list[int]:
    canonical_prompt = build_prompt(
        tokenizer,
        messages_before_assistant,
    )
    placeholder_message = {
        "role": "assistant",
        "content": "x",
    }
    messages_with_assistant = [
        *messages_before_assistant,
        placeholder_message,
    ]
    canonical_assistant_end = _render_chat(
        tokenizer,
        messages_with_assistant,
        add_generation_prompt=False,
    )

    placeholder_tokens = _encoded_text_tokens(
        tokenizer, "x"
    )
    canonical_action = [
        *canonical_prompt,
        *placeholder_tokens,
    ]
    if canonical_assistant_end[
        :len(canonical_action)
    ] != canonical_action:
        raise ValueError(
            "chat template 无法定位 assistant 结束边界"
            "（占位内容也不匹配）"  # (chat template cannot locate the assistant end boundary, even the placeholder content does not match)
        )
    assistant_closing_tokens = canonical_assistant_end[
        len(canonical_action):
    ]

    canonical_next_prompt = build_prompt(
        tokenizer,
        [*messages_with_assistant, next_tool_message],
    )
    if canonical_next_prompt[
        :len(canonical_assistant_end)
    ] != canonical_assistant_end:
        raise ValueError(
            "加入 tool observation 后 chat template 改写了历史消息"  # (after adding the tool observation, the chat template rewrote historical messages)
        )
    observation_tokens = canonical_next_prompt[
        len(canonical_assistant_end):
    ]

    overlap = _suffix_prefix_overlap(
        completion_tokens,
        assistant_closing_tokens,
    )
    return [
        *previous_prompt_tokens,
        *completion_tokens,
        *assistant_closing_tokens[overlap:],
        *observation_tokens,
    ]
```

A placeholder is used here because the Qwen chat template may re-segment the reasoning and content of the actually sampled text. The placeholder message is only used to compute the fixed template boundary; the real model actions always come from `completion_tokens`. The next turn's `prompt_tokens` must have the previous turn's full tokens as a prefix; otherwise `train.py` refuses to construct the `Datum`.

### 8.4.4 Outcome Reward and the Interpreter Feedback Mask

This chapter extracts the last $\boxed{}$ from the end of the final answer, using `math_verify` to judge whether the prediction and the reference answer are mathematically equivalent:

```python
def extract_last_boxed(text: str) -> str | None:
    marker = "\\boxed{"
    index = text.rfind(marker)
    if index < 0:
        return None

    start = index + len(marker)
    depth = 1
    for position in range(start, len(text)):
        if text[position] == "{":
            depth += 1
        elif text[position] == "}":
            depth -= 1
            if depth == 0:
                return text[start:position]
    return None


def answers_equivalent(
    prediction: str,
    reference: str,
) -> bool:
    try:
        return bool(
            verify(
                parse(f"${reference}$"),
                parse(f"${prediction}$"),
            )
        )
    except Exception:
        return False


def score_answer(
    text: str,
    reference: str,
) -> RewardResult:
    answer = extract_last_boxed(
        text[-ANSWER_WINDOW_CHARS:]
    )
    if answer is None:
        return RewardResult(
            -1.0, False, False, None
        )

    correct = answers_equivalent(
        answer.strip(),
        reference.strip(),
    )
    return RewardResult(
        1.0 if correct else -1.0,
        correct,
        True,
        answer,
    )
```

The reward function is:

$$R(\tau)=\begin{cases}+1, & \text{the final answer is correct},\\ -1, & \text{the answer is wrong or the format is invalid}.\end{cases}$$

Then the centered advantage is computed over the same question's trajectories:

$$A_i=R(\tau_i)-\bar R.$$

Code execution results are environment observations. They should appear in the model's context while being excluded from the policy loss. The ReTool paper calls this treatment the interpreter feedback mask; this chapter adopts the same token alignment approach as Search-R1:

| Token region | Kept in context | Old logprob | Advantage |
| --- | --- | --- | --- |
| Initial prompt | Yes | 0 | 0 |
| Assistant reasoning and code calls | Yes | rollout logprob | Trajectory advantage |
| stdout, stderr, or timeout message | Yes | 0 | 0 |
| Final answer | Yes | rollout logprob | Trajectory advantage |

The companion code uses PyTRIO's PPO loss:

```python
PPO_LOSS_CONFIG = {
    "clip_low_threshold": 0.8,
    "clip_high_threshold": 1.28,
}

training_client.forward_backward(
    datums,
    loss_fn="ppo",
    loss_fn_config=PPO_LOSS_CONFIG,
).result()
```

The original ReTool paper's reinforcement learning stage uses PPO. This chapter keeps the multi-turn code trajectories, the outcome reward, and the interpreter feedback mask, while adding same-question group sampling and centered advantages; therefore, the training pipeline can be summarized as "GRPO constructs relative advantages, PPO clipping controls policy updates."

An asymmetric probability-ratio boundary is used here. When the old-policy probability ratio falls below 0.8 or exceeds 1.28, PPO limits the corresponding update. Regardless of the clip parameters, the rollout sampler must be refreshed from the current weights before every training step begins.

### 8.4.5 The Code Execution Environment and Safety Boundaries

Executing model-generated code carries real risk. This chapter's `sandbox.py` uses a local subprocess with the following engineering constraints:

- Launches Python with an argument list, not through a shell;
- Creates an independent process for every call;
- Sets a wall-clock timeout and a CPU time limit;
- Kills the entire subprocess group after timeout;
- Limits the thread count of BLAS and other numeric libraries;
- Writes stdout and stderr to temporary files and limits how much is read back;
- Limits the number of concurrent executions;
- Cleans special tokens that could break the chat template.

The execution entry point is `LocalPythonSandbox.run_code()`. The model's code is first embedded into a bootstrap that sets CPU limits, then passed via an argument list to a fresh Python process:

```python
def run_code(self, code: str) -> ExecResult:
    started = time.perf_counter()
    self.stats.calls += 1
    bootstrap = _BOOTSTRAP.format(
        cpu=int(self.timeout) + 5,
        code=repr(code),
    )
    env = {
        **os.environ,
        **_CHILD_ENV_OVERRIDES,
    }

    with self._semaphore:
        with (
            tempfile.TemporaryFile() as stdout_file,
            tempfile.TemporaryFile() as stderr_file,
        ):
            process = subprocess.Popen(
                [
                    self.python,
                    "-B",
                    "-c",
                    bootstrap,
                ],
                stdout=stdout_file,
                stderr=stderr_file,
                env=env,
                start_new_session=True,
            )
            timed_out = False
            try:
                returncode = process.wait(
                    timeout=self.timeout
                )
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(
                    process.pid,
                    signal.SIGKILL,
                )
                returncode = None
                process.wait()

            stdout = _truncate_tail(
                _read_capped(stdout_file)
            )
            stderr = _truncate_tail(
                _read_capped(stderr_file)
            )

    latency = time.perf_counter() - started
    ok = not timed_out and returncode == 0
    if timed_out:
        self.stats.timeouts += 1
    elif ok:
        self.stats.successes += 1
    else:
        self.stats.errors += 1
    self.stats.latency_total += latency
    return ExecResult(
        ok,
        stdout,
        stderr,
        returncode,
        timed_out,
        latency,
    )
```

stdout and stderr are written to temporary files rather than `PIPE`, and the parent process only reads a limited number of bytes from the end of the files, so even if the model prints continuously, the entire output is not kept in the parent's memory. On timeout, the independent process group created with `start_new_session=True` is terminated in one shot with `killpg()`.

The execution result is uniformly converted into a single tool observation:

```python
def format_tool_content(
    self,
    result: ExecResult,
) -> str:
    if result.timed_out:
        return (
            "Error: execution timed out after "
            f"{self.timeout:.0f} seconds."
        )
    if result.ok:
        return sanitize_tool_content(result.stdout)
    if result.stderr:
        return sanitize_tool_content(result.stderr)
    return (
        "Error: process exited with code "
        f"{result.returncode}."
    )
```

On success it returns stdout, syntax errors and runtime exceptions return the raw stderr, and timeouts return a fixed error text. The current `run_code()` does not automatically add `print()` for the model, so the system prompt explicitly requires the model to print the values it wants to observe; with nothing printed, a successful observation is an empty string.

These measures mainly guard against accidental infinite loops, output floods, and resource exhaustion. They do not constitute trusted security isolation. The local subprocess can still read files accessible to the running account, access the network, read inherited environment variables, or exploit vulnerabilities in the host environment.

Therefore, before running ReTool you must observe the following safety requirements:

1. Do not execute model code directly on a host containing production credentials, private keys, or sensitive data;
2. Use a disposable container, a low-privilege virtual machine, or a dedicated sandbox service;
3. Disable the network by default, and set a read-only filesystem with a minimized set of visible directories;
4. Remove unrelated environment variables and cloud service credentials;
5. Set hard limits on CPU, memory, process count, disk, and execution time;
6. Save the code, exit status, and resource metrics to make auditing anomalous trajectories easier.

The "independent execution" in the tool description means that each call shares no Python state; it does not represent host-level security isolation. Even a teaching trial run should first be placed in an isolated environment.

### 8.4.6 How the Multi-Turn Code Rollout Advances

`rollout.py` separately stores the trainable Assistant actions, the full trajectory state, and the code calls that have "been generated but not yet executed":

```python
@dataclass
class AssistantTurn:
    prompt_tokens: list[int]
    completion_tokens: list[int]
    logprobs: list[float]
    text: str


@dataclass
class Trajectory:
    example: MathExample
    group_index: int
    messages: list[dict[str, Any]]
    next_prompt_tokens: list[int] | None = None
    question_index: int = 0
    turns: list[AssistantTurn] = field(
        default_factory=list
    )
    code_calls: int = 0
    final_text: str = ""
    reward: float = -1.0
    advantage: float = 0.0
    valid_format: bool = False
    correct: bool = False
    done: bool = False


@dataclass(frozen=True)
class PendingExecution:
    trajectory: Trajectory
    code: str
    call_id: str
    messages_before_assistant: list[dict[str, Any]]
    assistant_text: str
    prompt_tokens: list[int]
    completion_tokens: list[int]
```

`begin_advance()` consumes one Assistant output and decides whether the trajectory ends or enters the execution environment:

```python
def begin_advance(
    trajectory: Trajectory,
    prompt_tokens: list[int],
    sequence: Any,
    tokenizer: Any,
    config: RolloutConfig,
) -> PendingExecution | None:
    tokens, logprobs, text = read_sequence(
        sequence, tokenizer
    )
    text = text.strip()
    trajectory.turns.append(
        AssistantTurn(
            prompt_tokens,
            tokens,
            logprobs,
            text,
        )
    )
    parsed = parse_assistant(text)

    can_code = (
        parsed.kind == "tool"
        and trajectory.code_calls
        < config.max_code_calls
        and len(trajectory.turns)
        < config.max_assistant_turns
    )
    if not can_code:
        trajectory.messages.append(
            {"role": "assistant", "content": text}
        )
        trajectory.final_text = text
        trajectory.done = True
        return None

    call_id = (
        f"code-{trajectory.question_index}-"
        f"{trajectory.group_index}-"
        f"{trajectory.code_calls + 1}"
    )
    messages_before_assistant = list(
        trajectory.messages
    )
    trajectory.messages.append(
        {"role": "assistant", "content": text}
    )
    return PendingExecution(
        trajectory,
        parsed.code or "",
        call_id,
        messages_before_assistant,
        text,
        prompt_tokens,
        tokens,
    )
```

Many trajectories within the same round may request code execution simultaneously. They can run concurrently, but each trajectory must still wait for its own observation to return before starting the next round of generation:

```python
async def execute_pending_async(
    pendings: list[PendingExecution],
    sandbox: LocalPythonSandbox,
) -> list[ExecResult]:
    return list(
        await asyncio.gather(
            *(
                sandbox.arun_code(p.code)
                for p in pendings
            )
        )
    )


async def advance_round(
    responses: list[Any],
    requests: list[SampleRequest],
    trajectories: list[Trajectory],
    tokenizer: Any,
    sandbox: LocalPythonSandbox,
    config: RolloutConfig,
    progress_callback: Callable[[int], None] | None,
) -> None:
    pendings: list[PendingExecution] = []
    for request, response in zip(
        requests, responses, strict=True
    ):
        trajectory = trajectories[
            request.trajectory_index
        ]
        pending = begin_advance(
            trajectory,
            request.prompt_tokens,
            response.sequences[0],
            tokenizer,
            config,
        )
        if pending is not None:
            pendings.append(pending)
        elif (
            trajectory.done
            and progress_callback is not None
        ):
            progress_callback(1)

    if not pendings:
        return
    results = await execute_pending_async(
        pendings, sandbox
    )
    for pending, result in zip(
        pendings, results, strict=True
    ):
        finish_advance(
            pending,
            result,
            tokenizer,
            sandbox,
            config,
        )
        if (
            pending.trajectory.done
            and progress_callback is not None
        ):
            progress_callback(1)
```

`finish_advance()` calls `format_tool_content()` from the previous section, then uses `build_next_prompt()` from Section 8.4.3 to construct continuous tokens. If the execution result is too long, `fit_tool_content()` repeatedly keeps the tail, because tracebacks and the final numeric values usually sit at the end; when the result still does not fit the trajectory budget, that trajectory ends directly.

The first-round branching is identical to Search-R1's: each question submits only one prompt with `num_samples=group_size`. Each returned sequence becomes an independent trajectory via `copy.deepcopy(root)`. After execution results return, each branch's prompt differs, so all subsequent requests set `num_samples=1`, until a final answer is output or the code-call, turn, and token budgets are exhausted.

Only after all trajectories finish does the state machine call `score_trajectory()` and `assign_group_advantages()` in turn. Therefore, one trajectory's traceback can influence its own later code generation, but cannot cross the trajectory boundary to affect other branches in the same group.

### 8.4.7 Training ReTool with PyTRIO

First, create a sampling client from the current weights and generate multi-turn trajectories for the same question:

```python
sampling_client = (
    training_client.save_weights_and_get_sampling_client()
)

trajectories = rollout_batch(
    sampling_client=sampling_client,
    tokenizer=tokenizer,
    sandbox=sandbox,
    examples=batch,
    config=rollout_config,
)
```

`rollout_batch()` accomplishes the following:

1. In the first round, samples `group_size` branches for the same question;
2. Parses the `code_interpreter` calls in each branch;
3. Concurrently executes the model's code and appends tool observations;
4. Continues sampling trajectories that are not yet finished;
5. Extracts the final answer and computes the reward and the in-group advantage.

Every training step calls `save_weights_and_get_sampling_client()` again, guaranteeing that the rollout old logprobs come from the current policy before the update. If all trajectories for a question answer correctly or all answer incorrectly, the centered advantages are all 0, and `build_training_datums()` skips that group; when no `Datum` remains, the step does not execute `optim_step()`.

Then each complete trajectory is constructed into one `Datum`. ReTool's `build_datum()` follows the same prefix-extension rules as Section 8.3.6; the core loop is as follows:

```python
for turn_index, turn in enumerate(trajectory.turns):
    if turn_index == 0:
        delta_observation = turn.prompt_tokens
    elif turn.prompt_tokens[
        :len(full_tokens)
    ] == full_tokens:
        delta_observation = turn.prompt_tokens[
            len(full_tokens):
        ]
    else:
        raise ValueError(
            "下一轮 prompt 不是已有轨迹的前缀扩展"  # (the next turn's prompt is not a prefix extension of the existing trajectory)
        )

    full_tokens.extend(delta_observation)
    full_tokens.extend(turn.completion_tokens)
    old_logprobs_by_token.extend(
        [0.0] * len(delta_observation)
    )
    old_logprobs_by_token.extend(turn.logprobs)
    advantages_by_token.extend(
        [0.0] * len(delta_observation)
    )
    advantages_by_token.extend(
        [trajectory.advantage]
        * len(turn.completion_tokens)
    )

input_tokens = full_tokens[:-1]
target_tokens = full_tokens[1:]
old_logprobs = old_logprobs_by_token[1:]
advantages = advantages_by_token[1:]

datum = trio.Datum(
    model_input=trio.ModelInput.from_ints(
        input_tokens
    ),
    loss_fn_inputs={
        "target_tokens": np.asarray(
            target_tokens, dtype=np.int64
        ),
        "logprobs": np.asarray(
            old_logprobs, dtype=np.float32
        ),
        "advantages": np.asarray(
            advantages, dtype=np.float32
        ),
    },
)
```

`delta_observation` contains the initial question, the Assistant end tokens, and the stdout, stderr, or timeout text, all with advantage 0; the Assistant's reasoning, code calls, and final answer share the trajectory advantage. This is the actual code form of the interpreter feedback mask.

Because trajectory lengths vary greatly, the code then performs dynamic micro-batch packing:

```python
datums = build_training_datums(trajectories)
micro_batches = pack_micro_batches(datums)

for micro_batch in micro_batches:
    weighted = weight_micro_batch_for_global_mean(
        micro_batch,
        total_samples=len(trajectories),
    )
    training_client.forward_backward(
        weighted,
        loss_fn="ppo",
        loss_fn_config=PPO_LOSS_CONFIG,
    ).result()

if micro_batches:
    training_client.optim_step(adam_params).result()
```

PyTRIO averages the samples within a single `forward_backward()` call. When micro-batches differ in size, accumulating them directly would amplify the weight of small batches. This chapter scales the advantages by each micro-batch's share of samples out of the global trajectory count:

$$\sum_k \frac{n_k}{N}\mathrm{mean}(\mathcal L_k) = \mathrm{mean}(\mathcal L_{\mathrm{global}}).$$

This way, dynamic batch splitting only changes how execution is carried out, not the gradient meaning of the global sample average.

### 8.4.8 Running and Evaluating ReTool

The companion directory contains the following files:

| File | Role |
| --- | --- |
| `prepare_data.py` | Prepares the DAPO-Math-17k training set and dev set |
| `data.py` | Loads the unified math samples |
| `protocol.py` | Defines the code tool and the multi-turn message protocol |
| `sandbox.py` | Executes Python code and records resource metrics |
| `rollout.py` | Runs the multi-turn "generate—execute—observe" state machine |
| `reward.py` | Extracts $\boxed{}$ and performs mathematical-equivalence checking |
| `train.py` | Builds the feedback mask and micro-batches, and runs PPO |
| `eval.py` | Uniformly evaluates the text-only and ReTool modes |
| `analysis.py` | Aggregates metrics across checkpoints |

The `train.jsonl` of DAPO-Math-17k generated by `prepare_data.py` is used for RL, and an additional 50-question `dev.jsonl` is split off for quick checks. The formal `eval.py` uses a fixed set of 30 AIME 2025 questions, avoiding reporting results directly on training questions.

In ReTool mode, each question's `val_n` candidate responses are still generated by the same multi-turn state machine:

```python
config = RolloutConfig(
    group_size=args.val_n,
    max_code_calls=args.max_code_calls,
    max_assistant_turns=args.max_assistant_turns,
    max_trajectory_tokens=(
        args.max_trajectory_tokens
    ),
    max_assistant_tokens=args.max_assistant_tokens,
    max_tool_response_tokens=(
        args.max_tool_response_tokens
    ),
    temperature=args.temperature,
    top_p=args.top_p,
    seed=args.seed,
)

trajectories = await rollout_batch_async(
    sampling_client,
    tokenizer,
    sandbox,
    examples,
    config,
)
```

The `text-only` mode registers no code tool and directly samples `val_n` single-turn answers for the same question. Both modes ultimately write per-question JSONL and aggregate:

- `Average@N`: the average accuracy across all generated results;
- `Pass@N`: the proportion of questions answered correctly by at least one candidate;
- `format_rate`: the proportion from which a valid `\boxed{}` can be extracted;
- `mean_code_calls` and `mean_turns`: tool cost and trajectory length.

Therefore, comparing checkpoints requires at least fixing the question set, `val_n`, temperature, top-p, the maximum trajectory length, and the answer extraction logic; when comparing text-only with ReTool, you should additionally report execution latency and code-call counts.

First prepare the data:

```bash
python docs/chapter8/retool/prepare_data.py
```

After confirming the code executor runs in an isolated environment, do a one-step trial run:

```bash
python docs/chapter8/retool/train.py \
    --max-steps 1 \
    --questions-per-batch 1 \
    --group-size 4 \
    --max-code-calls 2 \
    --sandbox-workers 2 \
    --swanlab-mode disabled
```

Once the local AIME 2025 data is ready, first run the no-tool baseline, then the ReTool evaluation with the same sampling parameters:

```bash
python docs/chapter8/retool/eval.py \
    --mode text-only \
    --val-n 12 \
    --temperature 1.0 \
    --top-p 0.7 \
    --output docs/chapter8/retool/eval-results/aime25-text-only-base.jsonl

python docs/chapter8/retool/eval.py \
    --mode retool \
    --val-n 12 \
    --temperature 1.0 \
    --top-p 0.7 \
    --output docs/chapter8/retool/eval-results/aime25-retool-base.jsonl
```

When evaluating the trained model, add the same `--model-path trio://...` to both commands. Only when the Base Model, the checkpoint, text-only, and ReTool share the same sampling setup can the difference be attributed to training or to the tool environment.

Training metrics should cover three layers at once: task, policy, and environment:

- **Task outcomes**: accuracy, the proportion of valid $\boxed{}$, the degenerate group ratio;
- **Tool behavior**: code-call rate, average calls per trajectory, accuracy with zero calls;
- **Self-correction**: the proportion of trajectories that retry after a first execution error and ultimately answer correctly;
- **Environment state**: execution success rate, exception rate, timeout rate, average latency;
- **Policy stability**: PPO probability ratio, clip fraction, response length;
- **System efficiency**: time spent in rollout, the executor, remote training, and checkpointing.

During evaluation you can run the text-only and ReTool modes separately. The two must use the same question set, identical sampling parameters, and the same answer grader to reveal the net benefit brought by the tool environment. The tool mode should also report the extra latency and execution cost, because accuracy gains usually come with more environment calls.

At this point we have moved from GRPO's single-turn outcome rewards and OPD's token-by-token Teacher feedback to the multi-turn environment interaction of Search-R1 and ReTool. The chapter summary below gathers the four modules back into the same "rollout—feedback—advantage—policy update" data flow.

## Chapter Summary

Starting from GRPO, this chapter gradually built a large language model reinforcement learning path that can scale to Agents.

GRPO uses multiple samples of the same question to construct in-group relative advantages, eliminating the need to train a separate Value Model. It splits the algorithm into three clear layers: the policy generates rollouts, the environment gives rewards, and the update loss optimizes the model based on the probability ratio between new and old policies. A correct reward, token alignment, and on-policy sampler refresh jointly determine whether training is effective.

OPD keeps the on-policy rollout, replacing the rule-based reward with the Teacher's token-by-token probability feedback on the Student's trajectories. Reverse KL provides a dense signal for every action, letting a small model learn the Teacher's abilities on the states it will actually visit. The Teacher and Student's tokenizers, reasoning modes, and complementary capabilities need to be verified before training.

Search-R1 and ReTool further expand the trajectory into multi-turn environment interaction. Search results and code execution results enter the context, yet are excluded from the policy loss via the observation mask; the model-generated search terms, code, reasoning, and final answer share the trajectory-level outcome signal. Both reuse GRPO's group sampling and relative advantages, replacing only the tool protocol, the environment implementation, and the reward.

The four examples ultimately converge into the same PyTRIO data flow:

$$\text{current policy}\rightarrow\text{diverse rollouts}\rightarrow\text{environment or Teacher feedback}\rightarrow\text{token-level advantage}\rightarrow\text{policy update}\rightarrow\text{new policy}.$$

After mastering this data flow, a new Agentic RL task can be designed starting from five questions:

1. Which actions can the Agent perform, and what observations will the environment return?
2. Under what conditions does a trajectory end, and how do budgets constrain it?
3. Can the reward reliably reflect the final task objective?
4. Which tokens are policy actions, and which tokens must be masked?
5. Are the rollout, old logprobs, advantages, and current weights strictly aligned?

Only when these five questions all have verifiable answers does a tool Demo have the foundation to enter reinforcement learning training.

> **Keep learning:** Limited by space and this chapter's scope, this chapter concentrates on GRPO, OPD, Search-R1, and ReTool, and many Agentic RL methods and interactive environments could not be covered one by one. If you want to keep learning OPSD, DAPO, GSPO, ALFWorld, and more, visit the author's continuously maintained [agentic-rl-lab](https://github.com/KMnO4-zx/agentic-rl-lab). The repository will keep collecting the principles of related algorithms, runnable code, and real training processes.

## References

1. Shao, Z. et al. [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300). 2024.
2. Schulman, J. et al. [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347). 2017.
3. Agarwal, R. et al. [On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes](https://arxiv.org/abs/2306.13649). 2023.
4. Li, Y. et al. [Rethinking On-Policy Distillation of Large Language Models: Phenomenology, Mechanism, and Recipe](https://arxiv.org/abs/2604.13016). 2026.
5. Sun, J. et al. [EasyOPD: An Easy-to-use On-Policy Distillation Framework for Large Language Models](https://arxiv.org/abs/2607.11012). 2026.
6. Jin, B. et al. [Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning](https://arxiv.org/abs/2503.09516). 2025.
7. Feng, J. et al. [ReTool: Reinforcement Learning for Strategic Tool Use in LLMs](https://arxiv.org/abs/2504.11536). 2025.
8. PyTRIO. [Loss Function Guide](https://docs.pytrio.com/docs/guide/loss_fn).
9. PyTRIO. [Search-R1 Example](https://docs.pytrio.com/docs/example/search-r1).
10. Agentic-RL Lab (不要葱姜蒜, lit. "No Scallions, Ginger, or Garlic"). [agentic-rl-lab](https://github.com/KMnO4-zx/agentic-rl-lab).
