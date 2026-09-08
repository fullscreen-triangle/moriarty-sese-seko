// Segment length ratios chosen in the spirit of Moriarty's Segment dataclass
// (thigh/shank/foot mass ratios ~10%/5%/1.4% of body mass) to get physically
// plausible proportions for a stick figure - not a literal unit port.
export const SEGMENT_LENGTHS = {
  torso: 0.5,
  upperArm: 0.28,
  forearm: 0.26,
  thigh: 0.32,
  shank: 0.3,
  foot: 0.14,
  headRadius: 0.11,
};

export const ARENA = {
  width: 12, // meters
  floorY: 0,
  gravity: 20,
};

export type PowerTier = 0 | 0.5 | 1;

export const powerTier = (chargeMs: number): PowerTier => {
  if (chargeMs < 120) return 0;
  if (chargeMs < 350) return 0.5;
  return 1;
};

export const REACH: Record<string, number> = {
  jab: 0.9,
  hook: 0.85,
  uppercut: 0.7,
  lowKick: 1.0,
  roundhouseKick: 1.1,
  frontKick: 0.95,
};

export const BASE_POWER: Record<string, number> = {
  jab: 0.12,
  hook: 0.22,
  uppercut: 0.3,
  lowKick: 0.15,
  roundhouseKick: 0.32,
  frontKick: 0.2,
};

export const BASE_DURATIONS_MS: Record<string, { startup: number; active: number; recovery: number }> = {
  jab: { startup: 80, active: 60, recovery: 120 },
  hook: { startup: 140, active: 80, recovery: 220 },
  uppercut: { startup: 180, active: 90, recovery: 280 },
  lowKick: { startup: 120, active: 70, recovery: 180 },
  roundhouseKick: { startup: 220, active: 100, recovery: 320 },
  frontKick: { startup: 150, active: 80, recovery: 200 },
};
