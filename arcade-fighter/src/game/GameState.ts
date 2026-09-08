import type { FighterId } from '../entities/FighterState';
import type { CharacterId } from '../entities/Character';

export type MatchMode = '2P' | '1P_VS_AI';

export type MatchPhase =
  | { kind: 'INTRO' }
  | { kind: 'CHARACTER_SELECT'; mode: MatchMode; picks: Partial<Record<FighterId, CharacterId>> }
  | { kind: 'FIGHTING'; remainingMs: number }
  | { kind: 'ROUND_END'; reason: 'stumble' | 'timer'; winner: FighterId | 'draw' }
  | { kind: 'FLED'; fledPlayer: FighterId };

export const createInitialPhase = (): MatchPhase => ({ kind: 'INTRO' });
