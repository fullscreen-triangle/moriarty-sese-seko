import type { JointId } from '../entities/FighterState';
import { IDLE_POSE } from '../entities/FighterState';
import { lerp } from '../utils/vector2';

export type PoseKeyframes = Record<JointId, number>;

const base = IDLE_POSE;

const withOverrides = (overrides: Partial<PoseKeyframes>): PoseKeyframes => ({
  ...base,
  ...overrides,
});

export const POSES: Record<string, PoseKeyframes> = {
  idle: { ...base },

  jabStartup: withOverrides({ rightShoulder: -Math.PI * 0.05, rightElbow: -Math.PI * 0.6 }),
  jabActive: withOverrides({ rightShoulder: -Math.PI * 0.02, rightElbow: -Math.PI * 0.95 }),
  jabRecovery: withOverrides({ rightShoulder: -Math.PI * 0.15, rightElbow: -Math.PI * 0.45 }),

  hookStartup: withOverrides({ rightShoulder: -Math.PI * 0.5, rightElbow: -Math.PI * 0.7 }),
  hookActive: withOverrides({ rightShoulder: -Math.PI * 0.05, rightElbow: -Math.PI * 0.3 }),
  hookRecovery: withOverrides({ rightShoulder: -Math.PI * 0.2, rightElbow: -Math.PI * 0.4 }),

  uppercutStartup: withOverrides({ rightShoulder: -Math.PI * 0.1, rightElbow: -Math.PI * 0.8 }),
  uppercutActive: withOverrides({ rightShoulder: -Math.PI * 0.7, rightElbow: -Math.PI * 0.9 }),
  uppercutRecovery: withOverrides({ rightShoulder: -Math.PI * 0.25, rightElbow: -Math.PI * 0.4 }),

  lowKickStartup: withOverrides({ rightHip: -Math.PI * 0.15, rightKnee: -Math.PI * 0.3 }),
  lowKickActive: withOverrides({ rightHip: -Math.PI * 0.5, rightKnee: -Math.PI * 0.05 }),
  lowKickRecovery: withOverrides({ rightHip: -Math.PI * 0.1, rightKnee: -Math.PI * 0.15 }),

  roundhouseKickStartup: withOverrides({ rightHip: -Math.PI * 0.3, rightKnee: -Math.PI * 0.6 }),
  roundhouseKickActive: withOverrides({ rightHip: -Math.PI * 0.85, rightKnee: -Math.PI * 0.1 }),
  roundhouseKickRecovery: withOverrides({ rightHip: -Math.PI * 0.1, rightKnee: -Math.PI * 0.2 }),

  frontKickStartup: withOverrides({ rightHip: -Math.PI * 0.2, rightKnee: -Math.PI * 0.5 }),
  frontKickActive: withOverrides({ rightHip: -Math.PI * 0.6, rightKnee: -Math.PI * 0.02 }),
  frontKickRecovery: withOverrides({ rightHip: -Math.PI * 0.1, rightKnee: -Math.PI * 0.15 }),

  block: withOverrides({
    leftShoulder: Math.PI * 0.4,
    leftElbow: Math.PI * 0.8,
    rightShoulder: -Math.PI * 0.4,
    rightElbow: -Math.PI * 0.8,
  }),

  dodge: withOverrides({ neck: -Math.PI * 0.45 }),

  stumble: withOverrides({
    neck: -Math.PI * 0.3,
    leftShoulder: Math.PI * 0.5,
    rightShoulder: -Math.PI * 0.5,
    leftHip: Math.PI * 0.3,
    rightHip: -Math.PI * 0.3,
  }),

  fleeStance: withOverrides({
    neck: -Math.PI * 0.55,
    leftShoulder: Math.PI * 0.25,
    rightShoulder: -Math.PI * 0.25,
  }),
};

export const interpolatePose = (a: PoseKeyframes, b: PoseKeyframes, t: number): PoseKeyframes => {
  const result = {} as PoseKeyframes;
  for (const key in a) {
    const jointId = key as JointId;
    result[jointId] = lerp(a[jointId], b[jointId], t);
  }
  return result;
};
