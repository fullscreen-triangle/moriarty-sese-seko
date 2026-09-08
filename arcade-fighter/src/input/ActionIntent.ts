import type { FighterId } from '../entities/FighterState';

export type IntentType = 'punch' | 'kick' | 'block' | 'dodge' | 'flee';

export type DirectionBucket = 'neutral' | 'forward' | 'back' | 'up' | 'down' | 'downForward' | 'downBack';

export type ActionIntent = {
  type: IntentType;
  direction: DirectionBucket;
  chargeMs: number;
  playerId: FighterId;
};
