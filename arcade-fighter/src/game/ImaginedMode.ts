import type { FighterState } from '../entities/FighterState';
import type { RNG } from '../physics/RNG';
import type { ActionShape } from '../entities/MoveHistory';

/**
 * "The fight is happening inside someone's head." A randomly-triggered, timed
 * window during which both fighters strike their imagined-persona costume
 * (Bhuru: Fighter Pilot/Parachute, Heinrich: Olympic Wrestler/Slingkini) and
 * throw whatever they want — but per the sketch sheet's own captions
 * ("...that tickles.", "...no damage."), nothing in this window is real. It's
 * spectacle, not damage: Simulation still resolves hits/outcomes/proficiency
 * normally, but composure drain is zeroed for the duration.
 *
 * Trigger scales with fight intensity (recent composure swings) rather than
 * firing at a flat rate — a calm fight rarely imagines; a brutal one drifts.
 */

const BASE_TRIGGER_CHANCE_PER_SEC = 0.006; // ~1 window every ~170s of calm fighting
const INTENSITY_TRIGGER_SCALE = 0.09; // extra chance/sec per unit of recent composure-drain intensity
const MIN_WINDOW_MS = 5000;
const MAX_WINDOW_MS = 10000;
const INTENSITY_DECAY_PER_MS = 0.0006;
const COOLDOWN_MS = 8000; // no back-to-back windows

export type ImaginedPersona = 'bhuru-pilot' | 'heinrich-wrestler';

export interface ImaginedWindowTally {
  shapesAttempted: Partial<Record<ActionShape, number>>;
}

export class ImaginedMode {
  private active = false;
  private remainingMs = 0;
  private cooldownMs = 0;
  private intensity = 0;
  private tally: Record<'p1' | 'p2', ImaginedWindowTally> = {
    p1: { shapesAttempted: {} },
    p2: { shapesAttempted: {} },
  };

  isActive(): boolean {
    return this.active;
  }

  /** Feed real composure drain in as it happens, to build up trigger intensity. */
  noteComposureDrain(drain: number): void {
    if (drain > 0) this.intensity = Math.min(1, this.intensity + drain);
  }

  /** Record an attempted action's shape during the window, for post-window feedback. */
  noteAttempt(fighterId: 'p1' | 'p2', shape: ActionShape): void {
    if (!this.active) return;
    const bucket = this.tally[fighterId].shapesAttempted;
    bucket[shape] = (bucket[shape] ?? 0) + 1;
  }

  personaFor(characterId: FighterState['characterId']): ImaginedPersona {
    return characterId === 'bhuru' ? 'bhuru-pilot' : 'heinrich-wrestler';
  }

  /**
   * Advance the trigger/window clock. Returns the tally (and clears state) the
   * instant a window closes, so the caller can apply persistent feedback.
   */
  tick(dtMs: number, rng: RNG): Record<'p1' | 'p2', ImaginedWindowTally> | null {
    this.intensity = Math.max(0, this.intensity - INTENSITY_DECAY_PER_MS * dtMs);

    if (this.active) {
      this.remainingMs -= dtMs;
      if (this.remainingMs <= 0) {
        this.active = false;
        this.cooldownMs = COOLDOWN_MS;
        const closed = this.tally;
        this.tally = { p1: { shapesAttempted: {} }, p2: { shapesAttempted: {} } };
        return closed;
      }
      return null;
    }

    if (this.cooldownMs > 0) {
      this.cooldownMs -= dtMs;
      return null;
    }

    const chancePerSec = BASE_TRIGGER_CHANCE_PER_SEC + this.intensity * INTENSITY_TRIGGER_SCALE;
    if (rng.next() < chancePerSec * (dtMs / 1000)) {
      this.active = true;
      this.remainingMs = MIN_WINDOW_MS + rng.next() * (MAX_WINDOW_MS - MIN_WINDOW_MS);
    }
    return null;
  }

  /** Debug hook: force a window open right now, skipping the random gate. */
  forceTrigger(): void {
    this.active = true;
    this.remainingMs = MAX_WINDOW_MS;
    this.cooldownMs = 0;
  }
}
