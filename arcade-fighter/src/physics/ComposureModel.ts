import type { FighterState } from '../entities/FighterState';
import type { ResolutionOutcome } from '../entities/MoveHistory';
import { ALL_LIMB_IDS } from '../entities/LimbState';
import { BASE_POWER } from './Kinematics';
import type { ActionShape } from '../entities/MoveHistory';
import { ROSTER } from '../entities/Character';

const REGEN_RATE_PER_MS = 0.00004;
const DISORIENT_DECAY_PER_MS = 0.00008;

const totalInjurySeverity = (fighter: FighterState): number =>
  ALL_LIMB_IDS.reduce((sum, id) => sum + fighter.limbs[id].injurySeverity, 0) / ALL_LIMB_IDS.length;

export const applyHitToComposure = (
  defender: FighterState,
  shape: ActionShape,
  chargeBonus: number,
  outcome: ResolutionOutcome,
  imagined = false
): number => {
  if (outcome === 'MISS') return 0;

  const guardPartial = defender.blocking ? 0.55 : 0;
  const magnitude = outcome === 'GLANCING' ? 0.35 : 1.0;
  const drain =
    BASE_POWER[shape] * (0.5 + 0.5 * chargeBonus) * (1 - guardPartial) * magnitude;

  // Imagined Mode: the hit still happens narratively (outcome/proficiency are
  // computed normally upstream), but it is only in someone's head — it tickles,
  // it does not injure. No real composure/disorientation cost.
  if (imagined) return 0;

  defender.composure = Math.max(0, defender.composure - drain);

  const isUpperBody = shape !== 'lowKick' && shape !== 'roundhouseKick' && shape !== 'frontKick';
  if (isUpperBody) {
    defender.disorientation = Math.min(1, defender.disorientation + 0.3 * drain);
  }

  return drain;
};

export const regenComposure = (fighter: FighterState, dtMs: number): void => {
  const isBeingHit = fighter.actionState.kind === 'stumble';
  if (isBeingHit) return;
  const injuryDrag = 1 - 0.5 * totalInjurySeverity(fighter);
  const bias = ROSTER[fighter.characterId].statBias.composureRegen;
  fighter.composure = Math.min(1, fighter.composure + REGEN_RATE_PER_MS * dtMs * injuryDrag * bias);
};

export const decayDisorientation = (fighter: FighterState, dtMs: number): void => {
  fighter.disorientation = Math.max(0, fighter.disorientation - DISORIENT_DECAY_PER_MS * dtMs);
};

export const disorientPenalty = (fighter: FighterState): number =>
  1 - 0.6 * fighter.disorientation;
