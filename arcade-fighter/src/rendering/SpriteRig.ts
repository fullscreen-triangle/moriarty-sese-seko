import type { FighterState } from '../entities/FighterState';
import type { ImaginedPersona } from '../game/ImaginedMode';
import { getImage, isImageReady } from './ImageCache';
import {
  frameRect,
  frameCount,
  getMainSheet,
  getImaginedSheet,
  type SpriteRow,
} from './SpriteData';

const PPM = 90;
const DRAW_HEIGHT_METERS = 1.9;
const FRAME_MS = 90;

// Per-fighter animation clock, keyed by fighter id — a fresh row resets to frame 0.
const animClocks: Record<string, { row: SpriteRow; elapsedMs: number }> = {};

const rowForState = (state: FighterState, imagined: boolean): SpriteRow => {
  const action = state.actionState;

  if (action.kind === 'stumble') return 'hurt';
  if (action.kind === 'fleeing') return 'scared';
  if (action.kind === 'blocking') return 'crouch';
  if (imagined && (action.kind === 'active' || action.kind === 'startup')) return 'special';
  if (action.kind === 'startup' || action.kind === 'active' || action.kind === 'recovery') return 'hit';

  if (!state.grounded) return 'jump';
  if (Math.abs(state.velocity.x) > 2.2) return 'run';
  if (Math.abs(state.velocity.x) > 0.05) return 'walk';
  return 'idle';
};

const advanceFrame = (fighterId: string, row: SpriteRow, dtMs: number): number => {
  const clock = animClocks[fighterId] ?? { row, elapsedMs: 0 };
  if (clock.row !== row) {
    clock.row = row;
    clock.elapsedMs = 0;
  } else {
    clock.elapsedMs += dtMs;
  }
  animClocks[fighterId] = clock;
  return Math.floor(clock.elapsedMs / FRAME_MS) % frameCount(row);
};

export const drawFighterSprite = (
  ctx: CanvasRenderingContext2D,
  state: FighterState,
  floorPixelY: number,
  dtMs: number,
  persona?: ImaginedPersona
): void => {
  const imagined = persona !== undefined;
  const row = rowForState(state, imagined);
  const frameIndex = advanceFrame(state.id, row, dtMs);

  const sheet = imagined ? getImaginedSheet(persona) : getMainSheet(state.characterId);
  const rect = frameRect(sheet, row, frameIndex);
  const img = getImage(rect.src);
  if (!isImageReady(img)) return;

  const destH = DRAW_HEIGHT_METERS * PPM;
  const destW = (rect.sw / rect.sh) * destH;

  const rootX = state.position.x * PPM;
  const rootY = floorPixelY - state.position.y * PPM;

  ctx.save();
  ctx.translate(rootX, rootY);
  if (state.facing === -1) ctx.scale(-1, 1);
  ctx.drawImage(img, rect.sx, rect.sy, rect.sw, rect.sh, -destW / 2, -destH, destW, destH);
  ctx.restore();
};
