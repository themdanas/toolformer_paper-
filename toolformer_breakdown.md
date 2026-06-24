# Toolformer: Language Models Can Teach Themselves to Use Tools
### Paper Breakdown — Meta AI Research (Feb 2023)

---

## 1. Background

Large language models (LLMs) like GPT-3 showed remarkable zero-shot and few-shot generalization — they could answer questions, write code, summarize text, and reason across domains from just a few examples or natural language instructions.

But LLMs are trained on a static snapshot of text. Everything they "know" is baked into their weights at training time. They have no live connection to the world.

---

## 2. The Core Problem (The Need)

Despite their impressive capabilities, LLMs have fundamental, hard-to-fix limitations:

| Limitation | Example |
|---|---|
| No access to current information | Can't tell you today's weather or stock price |
| Hallucination | Confidently makes up facts |
| Poor arithmetic | Fails at "What is 17% of 4,832?" |
| No sense of time | Doesn't know what day/year it is |
| Weak low-resource language understanding | Struggles with non-English languages |

These aren't bugs — they are **structural limitations**. Scaling the model bigger helps only partially. The root cause is that LLMs are pure text prediction machines with no dynamic world access.

---

## 3. What Was Done Before (Prior Approaches)

Prior work tried two broad strategies to give LMs tool access:

### Strategy A: Human Supervision
- Train models on large human-annotated datasets of tool-use examples
- e.g., LaMDA, BlenderBot — humans explicitly labeled when and how to use a search engine
- **Cost**: Thousands of human annotations per tool, per task

### Strategy B: Task-Specific Few-Shot Prompting
- At inference time, engineer a prompt that shows the model how to use a tool for *that specific task*
- e.g., PAL (Program-Aided Language Models), ReAct
- Works by prompting: "here's how to use a calculator for math word problems"

### TALM (Most Related Prior Work)
- Used self-supervised training for tool use
- BUT: only worked in task-specific fine-tuning settings — not general-purpose zero-shot use

---

## 4. Why Prior Approaches Failed (The Gap)

Both strategies had serious limitations:

**Human supervision** is expensive and doesn't scale — you need new annotations for every new tool and task. Also, what *humans* find useful to annotate may differ from what the *model* actually needs.

**Task-specific prompting** requires you to know *in advance* which tool is needed for which task. You hardcode "use calculator for math" — but in the real world, a model should decide this autonomously. It also doesn't generalize across tasks.

**The fundamental gap**: No prior method could train a model to **decide for itself — in a general, task-agnostic way — when to use a tool, which tool to use, and what to pass to it**, without massive human labeling.

---

## 5. Main Idea of the Paper

> **Let the model teach itself to use tools using its own judgment as the training signal.**

The insight: if inserting an API call + its result into a piece of text makes the model's future predictions *easier* (lower loss), then that API call was useful. This is a pure self-supervised signal — no human needed.

This produces **Toolformer**: a GPT-J (6.7B) model fine-tuned to autonomously:
- Decide **when** to call a tool
- Choose **which** tool to call
- Formulate the **correct input** (query/expression)
- Incorporate the **result** into its generation

Tools included:
- **Calculator** — arithmetic expressions
- **Wikipedia Search** — BM25 retrieval over Wikipedia
- **Q&A System** — Atlas retrieval-augmented LM (factoid questions)
- **Machine Translation** — NLLB-600M (200 languages)
- **Calendar** — returns current date (no input needed)

---

## 6. How It Was Achieved — The Pipeline

### Step 0: Represent API Calls as Text
API calls are linearized into special token sequences inserted inline into text:

```
e(c)    = <API> tool_name(input) </API>
e(c, r) = <API> tool_name(input) → result </API>
```

Example: `"Out of 1400 participants, 400 (or [Calculator(400/1400) → 0.29] 29%) passed."`

---

### Step 1: Sample API Call Candidates (using in-context learning)
For each tool, a **few-shot prompt** (3–5 human examples) is written showing how the API can be used. This prompt is prepended to a document from the training corpus (CCNet), and the LM generates candidate API calls by predicting where the `<API>` token is likely.

Formally, for each position `i` in a text, the model computes:

```
p_i = P_M(<API> | prompt + text[:i])
```

If `p_i > τ_s` (sampling threshold), that position is kept as a candidate. Up to `k=5` positions and `m=5` API calls per position are sampled.

---

### Step 2: Execute the API Calls
Each sampled API call is actually **executed** — the calculator runs the expression, the search engine queries Wikipedia, the calendar returns the current date, etc. Each returns a text response `r`.

---

### Step 3: Filter API Calls (The Key Innovation)
This is where self-supervision happens. For each candidate API call `c_i` with result `r_i`, compute:

```
L⁺ = loss(model | <API> c_i → r_i </API> as prefix)   # call + result given
L⁻ = min(loss(no API call), loss(<API> c_i </API> no result))  # no call, or call but no result
```

**Keep the API call only if:**  `L⁻ - L⁺ ≥ τ_f`  (i.e., the call+result reduces loss by at least τ_f)

Intuition: the API call is useful if knowing its output makes predicting the rest of the text easier, compared to either not calling it at all, or calling it but getting no result.

---

### Step 4: Fine-tune on the Augmented Dataset
- Useful API calls are inserted back into the original texts, forming dataset `C*`
- The model is fine-tuned on `C*` using standard language modeling loss (next-token prediction)
- Crucially, `C*` still contains all the original text — the model doesn't lose its general LM ability

**At inference**, the model generates normally until it produces the `→` token, which signals "I need a tool result here." Generation pauses, the API is called, the result is inserted, and generation resumes.

---

### Results Summary

| Task | GPT-J (6.7B) | GPT-3 (175B) | Toolformer (6.7B) |
|---|---|---|---|
| LAMA (SQuAD) | 17.8 | 26.8 | **33.8** |
| Math (ASDiv) | 7.5 | 14.0 | **40.4** |
| Math (SVAMP) | 5.2 | 10.0 | **29.4** |
| QA (WebQS) | 18.5 | 29.0 | **26.3** |
| Temporal (DATESET) | 3.9 | 0.8 | **27.3** |

Toolformer (6.7B) beats GPT-3 (175B) — 25× larger — on most tasks.

### Scaling Observation
Tool use ability only **emerges around 775M parameters**. Below this, models can't reliably decide when/how to call tools. The gap between "with tools" and "without tools" stays large even at 6.7B — meaning tools are complementary to scale, not a substitute for it.

---

## 7. Limitations

The paper is honest about several gaps:

1. **No chained tool use**: API calls are generated independently. The model can't use output of Tool A as input to Tool B. (e.g., first call Calendar, then pass the date to QA.)

2. **No interactive tool use**: Can't reformulate a search query if the first result was bad. Can't browse through multiple results like a human would.

3. **Prompt sensitivity**: The model is sensitive to exact input wording when deciding whether to call an API — a known LLM fragility.

4. **Sample inefficiency**: Processing 1M+ documents might yield only a few thousand useful calculator examples. The filtering is harsh, especially for rare tool types.

5. **No cost awareness**: Toolformer doesn't factor in computational cost of API calls — a cheap calendar call and an expensive neural QA call are treated identically.

6. **At most one API call per input**: A hard constraint to prevent the model from looping on API calls. Limits expressiveness.

7. **MLQA inconsistency**: Finetuning on CCNet hurt multilingual performance for some languages due to distribution shift from GPT-J's original pretraining data.

---

## 8. Code Walkthrough

### Core Algorithm Sketch (Python-style pseudocode)

```python
# ── STEP 1: Sample API call candidates ──────────────────────────────────────

def sample_api_calls(model, text, prompt, tool_name, tau_s=0.05, k=5, m=5):
    candidates = {}
    for i, position in enumerate(text):
        # Compute probability of <API> token at this position
        p_api = model.prob("<API>", context=prompt + text[:i])
        if p_api > tau_s:
            candidates[i] = []
            # Sample m API calls from this position
            for _ in range(m):
                call = model.generate(
                    context=prompt + text[:i] + "<API>",
                    stop_token="</API>"
                )
                candidates[i].append(call)
    # Keep only top-k positions by probability
    return top_k(candidates, k)


# ── STEP 2: Execute API calls ────────────────────────────────────────────────

def execute_calls(candidates):
    results = {}
    for position, calls in candidates.items():
        results[position] = []
        for call in calls:
            tool, input_text = parse_call(call)   # e.g., ("Calculator", "400/1400")
            result = call_api(tool, input_text)    # actually runs the tool
            results[position].append((call, result))
    return results


# ── STEP 3: Filter by loss reduction ────────────────────────────────────────

def compute_loss(model, text, prefix, position):
    """Weighted cross-entropy over tokens from `position` onward."""
    return model.cross_entropy_loss(
        text_from=position,
        prefix=prefix,
        weights=[max(0, 1 - 0.2 * t) for t in range(len(text) - position)]
    )

def filter_api_calls(model, text, results, tau_f=1.0):
    useful_calls = []
    for position, call_result_pairs in results.items():
        for call, result in call_result_pairs:
            # Loss WITH the API call and its result
            L_plus = compute_loss(
                model, text,
                prefix=f"<API> {call} → {result} </API>",
                position=position
            )
            # Loss WITHOUT any API call
            L_no_call = compute_loss(model, text, prefix="", position=position)
            # Loss with call but NO result
            L_no_result = compute_loss(
                model, text,
                prefix=f"<API> {call} </API>",
                position=position
            )
            L_minus = min(L_no_call, L_no_result)

            # Keep only if inserting the call+result reduces loss enough
            if L_minus - L_plus >= tau_f:
                useful_calls.append((position, call, result))
    return useful_calls


# ── STEP 4: Build augmented dataset and fine-tune ────────────────────────────

def build_augmented_dataset(corpus, model, tools, tau_s, tau_f):
    augmented = []
    for text in corpus:
        augmented_text = text
        all_useful_calls = []
        for tool in tools:
            prompt = get_tool_prompt(tool)
            candidates = sample_api_calls(model, text, prompt, tool, tau_s)
            executed = execute_calls(candidates)
            useful = filter_api_calls(model, text, executed, tau_f)
            all_useful_calls.extend(useful)

        # Insert all useful API calls into text at their positions
        augmented_text = insert_api_calls(text, all_useful_calls)
        augmented.append(augmented_text)
    return augmented

# Fine-tune using standard language modeling (next-token prediction)
augmented_dataset = build_augmented_dataset(CCNet, model, tools, tau_s=0.05, tau_f=1.0)
fine_tune(model, augmented_dataset, lr=1e-5, steps=2000, batch_size=128)


# ── INFERENCE ────────────────────────────────────────────────────────────────

def toolformer_generate(model, prompt, k=10):
    output = ""
    while not end_of_sequence(output):
        # Modified greedy: trigger API if <API> is in top-k tokens
        top_k_tokens = model.top_k_tokens(context=prompt + output, k=k)
        if "<API>" in top_k_tokens:
            # Generate the full API call
            api_call = model.generate(context=prompt + output + "<API>",
                                       stop_token="→")
            # Execute the tool
            tool, inp = parse_call(api_call)
            result = call_api(tool, inp)
            # Insert result and continue
            output += f"<API> {api_call} → {result} </API>"
        else:
            output += model.next_token(context=prompt + output)
    return output
```

### Key Hyperparameters (from the paper)
```
τ_s = 0.05        # min probability to consider inserting an API call
τ_f = 1.0         # min loss reduction to keep a call (0.5 for rare tools)
k   = 5           # max candidate positions per text
m   = 5           # max API call samples per position
k_decode = 10     # top-k for inference-time API trigger
max 1 API call per input at inference
Training: 8× A100 40GB, BF16, DeepSpeed ZeRO-3, 2000 steps, lr=1e-5
```

---

## 9. Conclusion

Toolformer is a landmark paper for a few reasons:

**What it solved**: It showed that a 6.7B model with tools can beat a 175B model without them — tools are a *multiplicative* capability upgrade, not just additive.

**Why the idea matters**: The self-supervised training pipeline is the real contribution. You don't need labeled "tool-use datasets." The model bootstraps its own tool-use training data using perplexity as the quality signal.

**Why it's foundational**: This paper directly influenced the design of function-calling in GPT-4, Claude's tool use, and the broader "agentic AI" paradigm. Modern agent frameworks (LangChain, LangGraph, AutoGen) are all downstream of this idea.

**What it didn't solve** (and where the field went next):
- Chained/sequential tool use → ReAct, LangGraph
- Interactive tool use with feedback → WebGPT, Reflexion
- Multiple tools in complex pipelines → modern agent systems

The core Toolformer insight — *let the model's own loss signal determine which tool calls are worth learning* — remains elegant and is still used in various forms in RLHF/RLAIF pipelines today.
