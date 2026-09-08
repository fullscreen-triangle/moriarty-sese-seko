import type { LimbId, LimbState } from '../entities/LimbState';
import type { RNG } from './RNG';

export type InjuryCause = 'overextension' | 'counterHit';

export type InjuryRoll = {
  limb: LimbId;
  severityDelta: number;
  type: LimbState['injuryType'];
};

const BASE_SEVERITY: Record<InjuryCause, number> = {
  overextension: 0.15,
  counterHit: 0.25,
};

const injuryTypeForLimb = (limb: LimbId): LimbState['injuryType'] =>
  limb === 'leftArm' || limb === 'rightArm' ? 'strain' : 'bruise';

export const rollInjury = (
  cause: InjuryCause,
  limbId: LimbId,
  limb: LimbState,
  rng: RNG
): InjuryRoll | null => {
  const severityDelta = BASE_SEVERITY[cause] * (0.5 + 0.5 * rng.next());
  const reinjuryMultiplier = 1 + 0.5 * limb.reinjuryCount;
  const applied = severityDelta * reinjuryMultiplier;

  if (limb.injurySeverity > 0) {
    limb.reinjuryCount += 1;
  }
  limb.injurySeverity = Math.min(1, limb.injurySeverity + applied);
  limb.injuryType = limb.injuryType === 'none' ? injuryTypeForLimb(limbId) : limb.injuryType;

  return { limb: limbId, severityDelta: applied, type: limb.injuryType };
};

const BASE_HEAL_RATE_PER_MS = 0.3 / 75_000;
const BASE_FATIGUE_HEAL_RATE_PER_MS = 1 / 4_000;

export const decayLimb = (limb: LimbState, dtMs: number): void => {
  if (limb.injurySeverity > 0) {
    const healRate = BASE_HEAL_RATE_PER_MS / (1 + limb.reinjuryCount * 0.4);
    limb.injurySeverity = Math.max(0, limb.injurySeverity - healRate * dtMs);
    if (limb.injurySeverity === 0) limb.injuryType = 'none';
  }
  if (limb.fatigue > 0) {
    limb.fatigue = Math.max(0, limb.fatigue - BASE_FATIGUE_HEAL_RATE_PER_MS * dtMs);
  }
};

export const limbPenalty = (limb: LimbState): number =>
  1 - (0.5 * limb.injurySeverity + 0.3 * limb.fatigue);
