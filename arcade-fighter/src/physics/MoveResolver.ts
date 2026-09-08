import type { FighterState } from '../entities/FighterState';
import type { ActionIntent, DirectionBucket, IntentType } from '../input/ActionIntent';
import type { ActionShape, ResolutionOutcome } from '../entities/MoveHistory';
import { updateProficiency } from '../entities/MoveHistory';
import { REACH, powerTier, BASE_DURATIONS_MS } from './Kinematics';
import { limbPenalty } from './InjuryModel';
import { rollInjury } from './InjuryModel';
import { staminaPenalty, consumeStamina } from './StaminaModel';
import { disorientPenalty, applyHitToComposure } from './ComposureModel';
import type { RNG } from './RNG';
import type { LimbId } from '../entities/LimbState';
import { ROSTER } from '../entities/Character';

export type ResolutionResult = {
  shape: ActionShape;
  limb: LimbId;
  outcome: ResolutionOutcome;
  hitChance: number;
  roll: number;
  composureDrain: number;
  selfInjuryOccurred: boolean;
};

const clamp = (v: number, lo: number, hi: number): number => Math.min(hi, Math.max(lo, v));

const determineShape = (type: IntentType, direction: DirectionBucket, chargeMs: number): ActionShape => {
  const tier = powerTier(chargeMs);
  if (type === 'punch') {
    if (direction === 'downBack' || direction === 'down') return 'uppercut';
    if (tier === 0) return 'jab';
    return 'hook';
  }
  if (type === 'kick') {
    if (direction === 'downForward') return 'frontKick';
    if (tier === 0) return 'lowKick';
    return 'roundhouseKick';
  }
  return 'jab';
};

const limbForShape = (shape: ActionShape, facing: 1 | -1): LimbId => {
  const leadIsRight = facing === 1;
  switch (shape) {
    case 'jab':
      return leadIsRight ? 'rightArm' : 'leftArm';
    case 'hook':
    case 'uppercut':
      return leadIsRight ? 'leftArm' : 'rightArm';
    case 'lowKick':
    case 'frontKick':
      return leadIsRight ? 'rightLeg' : 'leftLeg';
    case 'roundhouseKick':
      return leadIsRight ? 'leftLeg' : 'rightLeg';
    default:
      return 'rightArm';
  }
};

export const resolveAction = (
  attacker: FighterState,
  defender: FighterState,
  intent: ActionIntent,
  rng: RNG,
  imagined = false
): ResolutionResult | null => {
  if (intent.type !== 'punch' && intent.type !== 'kick') return null;

  const shape = determineShape(intent.type, intent.direction, intent.chargeMs);
  const limbId = limbForShape(shape, attacker.facing);
  const limb = attacker.limbs[limbId];

  const distance = Math.abs(attacker.position.x - defender.position.x);
  const chargeBonus = powerTier(intent.chargeMs);

  const bias = ROSTER[attacker.characterId].statBias;
  const proficiency = attacker.moveHistory[shape].proficiency;
  const effectiveReach =
    REACH[shape] * bias.reach * (1 - 0.4 * limb.injurySeverity) * (1 - 0.2 * limb.fatigue) * (1 + 0.1 * proficiency);

  consumeStamina(attacker, shape, chargeBonus);
  limb.fatigue = Math.min(1, limb.fatigue + 0.08 + 0.05 * chargeBonus);

  if (distance > effectiveReach) {
    const overextension = chargeBonus * (1 - proficiency) * (limb.fatigue + 0.3);
    const injuryChance = imagined ? 0 : clamp(overextension * 0.4 * bias.injuryRisk, 0, 0.6);
    const selfInjuryOccurred = rng.next() < injuryChance;
    if (selfInjuryOccurred) rollInjury('overextension', limbId, limb, rng);
    updateProficiency(attacker.moveHistory[shape], 'MISS', selfInjuryOccurred, bias.learnRate);
    return {
      shape,
      limb: limbId,
      outcome: 'MISS',
      hitChance: 0,
      roll: 1,
      composureDrain: 0,
      selfInjuryOccurred,
    };
  }

  const lp = limbPenalty(limb);
  const sp = staminaPenalty(attacker);
  const dp = disorientPenalty(attacker);

  const defenderEvasion = defender.blocking
    ? 0.55
    : defender.actionState.kind === 'fleeing'
      ? 0.8
      : -0.15 * defender.disorientation;

  const baseChance = clamp(0.35 + 0.5 * proficiency + 0.15 * chargeBonus, 0.05, 0.95);
  const hitChance = clamp(baseChance * lp * sp * dp * (1 - defenderEvasion), 0.02, 0.97);

  const roll = rng.next();
  const outcome: ResolutionOutcome = roll > hitChance ? 'MISS' : roll > hitChance * 0.7 ? 'GLANCING' : 'CLEAN_HIT';

  let selfInjuryOccurred = false;
  if (outcome === 'MISS') {
    const overextension = chargeBonus * (1 - proficiency) * (limb.fatigue + 0.3);
    const injuryChance = imagined ? 0 : clamp(overextension * 0.4 * bias.injuryRisk, 0, 0.6);
    selfInjuryOccurred = rng.next() < injuryChance;
    if (selfInjuryOccurred) rollInjury('overextension', limbId, limb, rng);
  }

  let composureDrain = 0;
  if (outcome !== 'MISS') {
    composureDrain = applyHitToComposure(defender, shape, chargeBonus, outcome, imagined);

    if (!imagined) {
      const struckLimb = shape === 'jab' || shape === 'hook' || shape === 'uppercut'
        ? (defender.facing === 1 ? 'leftArm' : 'rightArm')
        : (defender.facing === 1 ? 'leftLeg' : 'rightLeg');
      const defLimb = defender.limbs[struckLimb];
      if (defLimb.fatigue > 0.6) {
        rollInjury('counterHit', struckLimb, defLimb, rng);
      }
    }
  }

  updateProficiency(attacker.moveHistory[shape], outcome, selfInjuryOccurred, bias.learnRate);

  return { shape, limb: limbId, outcome, hitChance, roll, composureDrain, selfInjuryOccurred };
};

export const durationForShape = (shape: ActionShape, proficiency: number) => {
  const base = BASE_DURATIONS_MS[shape];
  const scale = 1 - 0.25 * proficiency;
  return {
    startup: base.startup * scale,
    active: base.active * scale,
    recovery: base.recovery * scale,
  };
};
