# Agent S Replication — Team 4

Replication and analysis of [Agent S: An Open Agentic Framework That Uses Computers Like a Human](https://arxiv.org/abs/2410.08164) (ICLR 2025).

**Neural Networks Course | Texas Tech University | Team 4**
Reese Farrell, Tafara Mhangami

## Overview

Full Python re-implementation of Agent S with four novel improvements:

- **Grounding Verifier** — verifies element selection before execution via vision model
- **Stagnation Detector** — triggers replanning when agent repeats actions or UI is frozen
- **App-State-Aware Memory** — enriches episodic memory keys with application state
- **Unified LLM Client** — supports OpenAI and Gemini via single environment variable

## Setup

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=your-key-here
export LLM_PROVIDER=openai
```

## Run

```bash
python agent.py --task "Open LibreOffice Calc and sum column A"
```

## Structure
eof
