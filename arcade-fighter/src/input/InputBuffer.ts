import type { ActionIntent, DirectionBucket } from './ActionIntent';
import type { RNG } from '../physics/RNG';

const MAX_DELAY_MS = 180;

const ADJACENT: Record<DirectionBucket, DirectionBucket[]> = {
  neutral: ['forward', 'back'],
  forward: ['neutral', 'downForward'],
  back: ['neutral', 'downBack'],
  up: ['neutral'],
  down: ['downForward', 'downBack'],
  downForward: ['down', 'forward'],
  downBack: ['down', 'back'],
};

type Buffered = { intent: ActionIntent; readyAtMs: number };

export class InputBuffer {
  private pending: Buffered[] = [];

  /** Queue a raw intent, delaying/jittering it based on the acting fighter's disorientation. */
  push(intent: ActionIntent, nowMs: number, disorientation: number, rng: RNG): void {
    const delay = disorientation * MAX_DELAY_MS;
    const jittered = this.jitterDirection(intent.direction, disorientation, rng);
    this.pending.push({
      intent: { ...intent, direction: jittered },
      readyAtMs: nowMs + delay,
    });
  }

  private jitterDirection(direction: DirectionBucket, disorientation: number, rng: RNG): DirectionBucket {
    const jitterChance = 0.4 * disorientation;
    if (rng.next() >= jitterChance) return direction;
    const options = ADJACENT[direction];
    if (options.length === 0) return direction;
    const idx = Math.floor(rng.next() * options.length);
    return options[Math.min(idx, options.length - 1)];
  }

  /** Drain and return all intents whose delay has elapsed by nowMs. */
  drain(nowMs: number): ActionIntent[] {
    const ready: ActionIntent[] = [];
    const remaining: Buffered[] = [];
    for (const item of this.pending) {
      if (item.readyAtMs <= nowMs) ready.push(item.intent);
      else remaining.push(item);
    }
    this.pending = remaining;
    return ready;
  }
}
