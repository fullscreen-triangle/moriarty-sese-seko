export type LimbId = 'leftArm' | 'rightArm' | 'leftLeg' | 'rightLeg';

export type InjuryType = 'none' | 'strain' | 'sprain' | 'bruise';

export type LimbState = {
  fatigue: number;
  injurySeverity: number;
  injuryType: InjuryType;
  reinjuryCount: number;
};

export const createLimbState = (): LimbState => ({
  fatigue: 0,
  injurySeverity: 0,
  injuryType: 'none',
  reinjuryCount: 0,
});

export const ALL_LIMB_IDS: LimbId[] = ['leftArm', 'rightArm', 'leftLeg', 'rightLeg'];
