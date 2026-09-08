import type { FighterState } from '../entities/FighterState';
import type { ActionShape } from '../entities/MoveHistory';

const STAMINA_COST: Record<ActionShape, number> = {
  jab: 0.02,
  hook: 0.04,
  uppercut: 0.06,
  lowKick: 0.03,
  roundhouseKick: 0.07,
  frontKick: 0.05,
  block: 0.005,
  dodge: 0.03,
  flee: 0.01,
};

const STAMINA_REGEN_PER_MS = 0.00006;

export const consumeStamina = (fighter: FighterState, shape: ActionShape, chargeBonus: number): void => {
  const cost = STAMINA_COST[shape] * (1 + 0.5 * chargeBonus);
  fighter.stamina = Math.max(0, fighter.stamina - cost);
};

export const regenStamina = (fighter: FighterState, dtMs: number): void => {
  const isActing = fighter.actionState.kind === 'startup' || fighter.actionState.kind === 'active';
  if (isActing) return;
  fighter.stamina = Math.min(1, fighter.stamina + STAMINA_REGEN_PER_MS * dtMs);
};

export const staminaPenalty = (fighter: FighterState): number => 0.5 + 0.5 * fighter.stamina;
