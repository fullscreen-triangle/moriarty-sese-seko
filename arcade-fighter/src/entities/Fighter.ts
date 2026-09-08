import type { FighterId, FighterState, Facing } from './FighterState';
import { createFighterState } from './FighterState';
import type { CharacterId } from './Character';
import type { Vec2 } from '../utils/vector2';

export const createFighter = (
  id: FighterId,
  characterId: CharacterId,
  position: Vec2,
  facing: Facing
): FighterState => createFighterState(id, characterId, position, facing);
