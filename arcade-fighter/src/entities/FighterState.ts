import type { Vec2 } from '../utils/vector2';
import { type LimbId, type LimbState, createLimbState, ALL_LIMB_IDS } from './LimbState';
import { type ActionShape, type MoveHistoryMap, createMoveHistoryMap } from './MoveHistory';
import type { CharacterId } from './Character';

export type Facing = 1 | -1;

export type JointId =
  | 'neck'
  | 'leftShoulder'
  | 'leftElbow'
  | 'rightShoulder'
  | 'rightElbow'
  | 'leftHip'
  | 'leftKnee'
  | 'rightHip'
  | 'rightKnee'
  | 'leftAnkle'
  | 'rightAnkle';

export const ALL_JOINT_IDS: JointId[] = [
  'neck',
  'leftShoulder',
  'leftElbow',
  'rightShoulder',
  'rightElbow',
  'leftHip',
  'leftKnee',
  'rightHip',
  'rightKnee',
  'leftAnkle',
  'rightAnkle',
];

export type ActionState =
  | { kind: 'idle' }
  | { kind: 'startup'; shape: ActionShape; limb: LimbId; elapsedMs: number; totalMs: number }
  | { kind: 'active'; shape: ActionShape; limb: LimbId; elapsedMs: number; totalMs: number }
  | { kind: 'recovery'; shape: ActionShape; limb: LimbId; elapsedMs: number; totalMs: number }
  | { kind: 'stumble'; elapsedMs: number; totalMs: number }
  | { kind: 'blocking' }
  | { kind: 'fleeing'; elapsedMs: number; totalMs: number };

export type FighterId = 'p1' | 'p2';

export type FighterState = {
  id: FighterId;
  characterId: CharacterId;
  position: Vec2;
  velocity: Vec2;
  facing: Facing;
  grounded: boolean;

  limbs: Record<LimbId, LimbState>;

  stamina: number;
  composure: number;
  disorientation: number;

  moveHistory: MoveHistoryMap;

  actionState: ActionState;
  blocking: boolean;
  hasFled: boolean;

  jointAngles: Record<JointId, number>;
};

export const IDLE_POSE: Record<JointId, number> = {
  neck: -Math.PI / 2,
  leftShoulder: Math.PI * 0.15,
  leftElbow: Math.PI * 0.35,
  rightShoulder: -Math.PI * 0.15,
  rightElbow: -Math.PI * 0.35,
  leftHip: Math.PI * 0.05,
  leftKnee: Math.PI * 0.1,
  rightHip: -Math.PI * 0.05,
  rightKnee: -Math.PI * 0.1,
  leftAnkle: 0,
  rightAnkle: 0,
};

export const createFighterState = (
  id: FighterId,
  characterId: CharacterId,
  position: Vec2,
  facing: Facing
): FighterState => {
  const limbs = {} as Record<LimbId, LimbState>;
  for (const limbId of ALL_LIMB_IDS) {
    limbs[limbId] = createLimbState();
  }

  return {
    id,
    characterId,
    position,
    velocity: { x: 0, y: 0 },
    facing,
    grounded: true,
    limbs,
    stamina: 1,
    composure: 1,
    disorientation: 0,
    moveHistory: createMoveHistoryMap(),
    actionState: { kind: 'idle' },
    blocking: false,
    hasFled: false,
    jointAngles: { ...IDLE_POSE },
  };
};
