# Arcade Fighter — Bhuru-sukurin: Chapter One

A fighting game with a fixed two-character roster — **Bhuru-sukurin** vs.
**Heinrich** — playable either as local 2P or 1P vs. an AI-controlled Heinrich.
Moves aren't chosen from a per-character list; each fighter's "moves" are
accumulated attempts at raw body actions (punches/kicks), resolved through a
shared physics/skill model and biased per-character by small stat multipliers.

- No HP bar. Fighters have **stamina** (gas tank) and **composure** (poise);
  composure hitting zero ends the match in a stumble/KO.
- Every attack can miss, land clean, or land glancing, and a bad miss can injure
  the limb that threw it (a whiffed haymaker can sprain your own wrist).
- Injuries and fatigue are **soft debuffs only** — they degrade accuracy, reach,
  and pose, but never lock out input.
- Getting hit builds disorientation, which delays and jitters your next input —
  again, never blocks it outright.
- Either player can flee at any moment. It has a short (250ms) windup during
  which you can still be caught and knocked into a stumble, but if it completes
  the match ends immediately — fleeing is not scored as a win or a loss.
- Nothing persists across a page reload. Move proficiency is a per-session
  "muscle memory," not a save file.
- In 1P mode, Heinrich is an AI opponent: a fast per-tick policy reacts every
  frame, and a slow background call to a local Ollama model periodically
  updates its "case file" on you — see [Design notes](#design-notes).
- **Imagined Mode**: the fight randomly (and briefly) drops into "someone's
  head" — both fighters strike an imagined persona (Bhuru: Fighter Pilot,
  Heinrich: Olympic Wrestler) and can throw anything, but nothing in this
  window is real: zero composure drain, zero self-injury, zero counter-hit
  injury. It's spectacle, not damage — "...that tickles," "...no damage."
  What you attempt while imagining still leaves a trace: Heinrich's AI
  remembers the shapes you favored, even though the window itself did nothing.

## Running

```bash
npm install
npm run dev
```

Open the printed local URL.

### Mode select (at the title screen)

| Key | Mode |
|-----|------|
| `1` | 1P vs. AI — you play Bhuru, Heinrich is AI-controlled |
| `2` | 2P local — same-keyboard, both players human |

### Character select (after choosing a mode)

| Key | Pick |
|-----|------|
| `1` | Bhuru-sukurin |
| `2` | Heinrich |

In 1P mode, picking here only sets your character — Heinrich is assigned to the
AI automatically. In 2P mode, player 1 picks first, then player 2 picks the
remaining character.

### In a match

| Action       | Player 1 | Player 2 |
|--------------|----------|----------|
| Move         | A / D    | ← / →    |
| Up-tilt      | W        | ↑        |
| Down-tilt    | S        | ↓        |
| Punch        | F        | 4        |
| Kick         | G        | 5        |
| Block (hold) | H        | 6        |
| Flee         | Q        | /        |

Punch/kick shape (jab vs. hook vs. uppercut, low kick vs. roundhouse vs. front
kick) is determined automatically from the direction held and how long the
button is charged when released — there is no separate "special move" input.
(Player 2's controls apply only in 2P mode; in 1P mode Heinrich is driven by
the AI.)

### Debug

| Key | Effect |
|-----|--------|
| `Tab` | Toggle the debug overlay — live proficiency, hit chance, injury state, and (in 1P mode) Heinrich's top KnowledgeGraph observations per fighter. |
| `I` | Force-trigger Imagined Mode immediately, for testing the buff window without waiting for it to fire randomly. |

## Design notes

See the implementation plan this was built from for the full move-resolution
formula, injury/composure math, and milestone breakdown. In short: `MoveResolver
.resolveAction()` is the heart of the game — it takes a raw intent plus both
fighters' current state and a seeded RNG, and produces a hit/miss/injury outcome
that then feeds back into that specific action's proficiency for that specific
fighter.

In 1P mode, Heinrich's behavior comes from two layers sharing one
`KnowledgeGraph`: `PolicyResolver` reacts every tick using the graph as it
currently stands (a bounded-attention, floored-response policy — never an LLM
call on the hot path), while `OllamaStrategist` runs on a slow, independent
cadence and asks a local Ollama model for graph updates, merging the response
back in as committed structure. If Ollama isn't running, this degrades to "the
graph stops updating," never a crash or a stall.
