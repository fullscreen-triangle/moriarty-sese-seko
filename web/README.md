# NPC Persuasion Demo

A 3D character (`xbot`) runs with a stated **purpose**. You chat with it and try to
make a compelling case for it to stop or change what it's doing. An LLM scores your
argument on two factors — **purpose gain** and **coherence** — and a deterministic gate
decides whether the character is persuaded.

This is a live instantiation of three manuscripts in [`../publications/`](../publications):

- **The Purpose of a Character** → the character *has* a purpose (an attractor).
- **The Physiology of Response** → the **two-factor relevance** judge, the price `p*`, and irreversibility.
- **Propagation Mechanics of an Embedded Agent** → replies are *searched*, and the verdict is the observer's, not a state the agent represents.

## Quick start

```bash
cd web
npm install
cp .env.local.example .env.local   # then edit it (see below)
npm run dev                        # http://localhost:3000
```

## Choosing an LLM backend

The judge runs through one API route (`/api/persuade`) behind a provider switch, set by
`LLM_PROVIDER` in `.env.local`.

### Option A — Ollama (local, offline, no cost)

```bash
# install from https://ollama.com, then:
ollama pull llama3.2
ollama serve            # usually already running as a service
```

```env
LLM_PROVIDER=ollama
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2
```

### Option B — Hugging Face Inference API (works when deployed)

Create a token at <https://huggingface.co/settings/tokens>.

```env
LLM_PROVIDER=hf
HF_TOKEN=hf_xxx
HF_MODEL=mistralai/Mistral-7B-Instruct-v0.3
```

## How it works

1. Your message is an **interaction**. The system prompt (built from `src/lib/persona.js`)
   tells the model the character's purpose and rules.
2. The model returns strict JSON scoring `purposeGain` and `coherence` (0–1), plus an
   in-character `reply` and an optional `requestedAnimation`.
3. The deterministic gate in `src/lib/scorer.js` decides: **persuaded iff**
   `purposeGain > p*` **and** `coherence` holds. The model never decides the verdict.
4. On a persuaded verdict the character cross-fades to a new clip and the **committed
   count** increments (irreversible — it never decreases).

Turn on **Research mode** in the UI to see both meters, the `p*` line, the committed
count, and the per-verdict reasoning.

## Project layout

```
src/lib/persona.js        the character's purpose + rules (source of truth)
src/lib/scorer.js         the deterministic two-factor gate
src/lib/prompt.js         system-prompt builder + JSON parse/repair
src/lib/llm/ollama.js     local Ollama adapter
src/lib/llm/hf.js         Hugging Face Inference adapter
src/pages/api/persuade.js the judge endpoint
src/components/demo/      Scene, Character, ChatPanel, InstrumentPanel, ...
src/pages/index.js        the demo home
src/pages/papers.js       the three manuscripts
src/pages/how-it-works.js the judge explained
public/xbot_multiple_animations.glb   clips: Idle, Jump, Running, Walking
public/papers/*.pdf       the compiled manuscripts
```

## Notes

- The 3D scene is client-only (`next/dynamic` with `ssr: false`) — three.js never runs on the server.
- If the LLM backend is unreachable, the character simply "keeps running" — the demo never crashes.
- Styling (Tailwind theme, dark mode, framer-motion, page transitions) is inherited from the
  portfolio template this project was built on.
