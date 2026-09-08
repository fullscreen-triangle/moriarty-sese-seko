export type ActionShape =
  | 'jab'
  | 'hook'
  | 'uppercut'
  | 'lowKick'
  | 'roundhouseKick'
  | 'frontKick'
  | 'block'
  | 'dodge'
  | 'flee';

export type MoveHistoryEntry = {
  attempts: number;
  successes: number;
  whiffs: number;
  selfInjuries: number;
  proficiency: number;
};

export type MoveHistoryMap = Record<ActionShape, MoveHistoryEntry>;

export const ALL_ACTION_SHAPES: ActionShape[] = [
  'jab',
  'hook',
  'uppercut',
  'lowKick',
  'roundhouseKick',
  'frontKick',
  'block',
  'dodge',
  'flee',
];

const createEntry = (): MoveHistoryEntry => ({
  attempts: 0,
  successes: 0,
  whiffs: 0,
  selfInjuries: 0,
  proficiency: 0.1,
});

export const createMoveHistoryMap = (): MoveHistoryMap => {
  const map = {} as MoveHistoryMap;
  for (const shape of ALL_ACTION_SHAPES) {
    map[shape] = createEntry();
  }
  return map;
};

export type ResolutionOutcome = 'MISS' | 'GLANCING' | 'CLEAN_HIT';

export const updateProficiency = (
  entry: MoveHistoryEntry,
  outcome: ResolutionOutcome,
  selfInjuryOccurred: boolean,
  learnRateBias = 1
): void => {
  entry.attempts += 1;
  if (outcome === 'CLEAN_HIT') entry.successes += 1;
  if (outcome === 'MISS') entry.whiffs += 1;
  if (selfInjuryOccurred) entry.selfInjuries += 1;

  const learningRate = (1 / (1 + entry.attempts * 0.05)) * learnRateBias;
  const target = selfInjuryOccurred
    ? -0.3
    : outcome === 'CLEAN_HIT'
      ? 1.0
      : outcome === 'GLANCING'
        ? 0.6
        : 0.0;

  const raw = entry.proficiency + learningRate * (target - entry.proficiency) * 0.25;
  entry.proficiency = Math.min(0.98, Math.max(0.05, raw));
};
