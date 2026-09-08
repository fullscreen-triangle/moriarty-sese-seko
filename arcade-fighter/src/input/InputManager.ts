import type { FighterId } from '../entities/FighterState';
import type { ActionIntent, DirectionBucket, IntentType } from './ActionIntent';

type KeyMap = {
  left: string;
  right: string;
  up: string;
  down: string;
  punch: string;
  kick: string;
  block: string;
  flee: string;
};

const P1_KEYS: KeyMap = {
  left: 'a',
  right: 'd',
  up: 'w',
  down: 's',
  punch: 'f',
  kick: 'g',
  block: 'h',
  flee: 'q',
};

const P2_KEYS: KeyMap = {
  left: 'arrowleft',
  right: 'arrowright',
  up: 'arrowup',
  down: 'arrowdown',
  punch: '4',
  kick: '5',
  block: '6',
  flee: '/',
};

type ChargeTracker = { punch: number | null; kick: number | null };

export class InputManager {
  private pressed = new Set<string>();
  private keymaps: Record<FighterId, KeyMap> = { p1: P1_KEYS, p2: P2_KEYS };
  private charge: Record<FighterId, ChargeTracker> = {
    p1: { punch: null, kick: null },
    p2: { punch: null, kick: null },
  };
  private pendingReleases: ActionIntent[] = [];
  private blockHeld: Record<FighterId, boolean> = { p1: false, p2: false };
  private fleeHeld: Record<FighterId, boolean> = { p1: false, p2: false };
  private nowMs = 0;

  constructor() {
    window.addEventListener('keydown', (e) => this.onKeyDown(e));
    window.addEventListener('keyup', (e) => this.onKeyUp(e));
  }

  private onKeyDown(e: KeyboardEvent): void {
    const key = e.key.toLowerCase();
    if (this.pressed.has(key)) return;
    this.pressed.add(key);

    for (const playerId of ['p1', 'p2'] as FighterId[]) {
      const keymap = this.keymaps[playerId];
      if (key === keymap.punch) this.charge[playerId].punch = this.nowMs;
      if (key === keymap.kick) this.charge[playerId].kick = this.nowMs;
      if (key === keymap.block) this.blockHeld[playerId] = true;
      if (key === keymap.flee) this.fleeHeld[playerId] = true;
    }
  }

  private onKeyUp(e: KeyboardEvent): void {
    const key = e.key.toLowerCase();
    this.pressed.delete(key);

    for (const playerId of ['p1', 'p2'] as FighterId[]) {
      const keymap = this.keymaps[playerId];
      if (key === keymap.block) this.blockHeld[playerId] = false;

      let type: IntentType | null = null;
      let startedAt: number | null = null;
      if (key === keymap.punch) {
        type = 'punch';
        startedAt = this.charge[playerId].punch;
        this.charge[playerId].punch = null;
      } else if (key === keymap.kick) {
        type = 'kick';
        startedAt = this.charge[playerId].kick;
        this.charge[playerId].kick = null;
      }

      if (type && startedAt !== null) {
        this.pendingReleases.push({
          type,
          direction: this.readDirection(playerId),
          chargeMs: Math.max(0, this.nowMs - startedAt),
          playerId,
        });
      }
    }
  }

  private readDirection(playerId: FighterId): DirectionBucket {
    const keymap = this.keymaps[playerId];
    const left = this.pressed.has(keymap.left);
    const right = this.pressed.has(keymap.right);
    const up = this.pressed.has(keymap.up);
    const down = this.pressed.has(keymap.down);

    const forwardKey = playerId === 'p1' ? right : left;
    const backKey = playerId === 'p1' ? left : right;

    if (down && forwardKey) return 'downForward';
    if (down && backKey) return 'downBack';
    if (up) return 'up';
    if (down) return 'down';
    if (forwardKey) return 'forward';
    if (backKey) return 'back';
    return 'neutral';
  }

  isBlocking(playerId: FighterId): boolean {
    return this.blockHeld[playerId];
  }

  isFleeing(playerId: FighterId): boolean {
    return this.fleeHeld[playerId];
  }

  movementAxis(playerId: FighterId): number {
    const keymap = this.keymaps[playerId];
    const left = this.pressed.has(keymap.left) ? -1 : 0;
    const right = this.pressed.has(keymap.right) ? 1 : 0;
    return left + right;
  }

  /** Called once per simulation tick; advances internal clock and drains completed intents. */
  poll(nowMs: number): ActionIntent[] {
    this.nowMs = nowMs;
    const released = this.pendingReleases;
    this.pendingReleases = [];
    return released;
  }
}
