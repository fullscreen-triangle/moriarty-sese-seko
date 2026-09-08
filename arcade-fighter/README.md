# Arcade Fighter

A small local 2-player fighting game with no predefined movesets. Each fighter's
"moves" are just accumulated attempts at raw body actions — punches and kicks are
resolved through a physics/skill model, not chosen from a character roster.

- No characters to pick. Both players start identical and blank.
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

## Running

```bash
npm install
npm run dev
```

Open the printed local URL. Press **Space** to start the match, **Tab** to
toggle the debug overlay (shows live proficiency, hit chance, and injury state
per fighter — this is where you can watch a jab actually get better with reps).

## Controls

| Action     | Player 1 | Player 2 |
|------------|----------|----------|
| Move       | A / D    | ← / →    |
| Up-tilt    | W        | ↑        |
| Down-tilt  | S        | ↓        |
| Punch      | F        | 4        |
| Kick       | G        | 5        |
| Block (hold) | H      | 6        |
| Flee       | Q        | /        |

Punch/kick shape (jab vs. hook vs. uppercut, low kick vs. roundhouse vs. front
kick) is determined automatically from the direction held and how long the
button is charged when released — there is no separate "special move" input.

## Design notes

See the implementation plan this was built from for the full move-resolution
formula, injury/composure math, and milestone breakdown. In short: `MoveResolver
.resolveAction()` is the heart of the game — it takes a raw intent plus both
fighters' current state and a seeded RNG, and produces a hit/miss/injury outcome
that then feeds back into that specific action's proficiency for that specific
fighter. There is no shared or global learning, and no AI opponent — this is a
same-keyboard 2-player game only.
