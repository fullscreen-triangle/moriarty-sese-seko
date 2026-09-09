import type { CharacterId } from '../entities/Character';
import type { ImaginedPersona } from '../game/ImaginedMode';

// Both characters share one sheet, split left/right at x=656. Each half has an
// 8-row filmstrip below the turnaround-reference art, all sharing the same row
// grid (measured against public/ChatGPT Image 8. Sept. 2026, 06_55_43.png).
const MAIN_SHEET = '/ChatGPT Image 8. Sept. 2026, 06_55_43.png';
const IMAGINED_SHEET = '/ChatGPT Image 8. Sept. 2026, 07_59_05.png';

// All sprite sheet sources the game draws from, for upfront preloading
// (see ui/LoadingScreen.ts + game/Simulation.ts).
export const ALL_SPRITE_SHEETS: string[] = [MAIN_SHEET, IMAGINED_SHEET];

export type SpriteRow =
  | 'idle'
  | 'walk'
  | 'run'
  | 'jump'
  | 'crouch'
  | 'hit'
  | 'hurt'
  | 'scared'
  | 'special';

const ROW_TOP = 570 + 45;
const ROW_H = 73;
const ROW_ORDER: SpriteRow[] = ['idle', 'walk', 'run', 'jump', 'crouch', 'hit', 'hurt', 'scared', 'special'];

const rowY = (row: SpriteRow): number => ROW_TOP + ROW_ORDER.indexOf(row) * ROW_H;

type RowDef = { frames: number };

// Frame counts eyeballed per row from the sheet (varies 5-7 per row).
const ROW_FRAMES: Record<SpriteRow, RowDef> = {
  idle: { frames: 7 },
  walk: { frames: 7 },
  run: { frames: 6 },
  jump: { frames: 5 },
  crouch: { frames: 6 },
  hit: { frames: 5 },
  hurt: { frames: 7 },
  scared: { frames: 6 },
  special: { frames: 7 },
};

const HALF_W = 656;
const CONTENT_LEFT = 8;
const CONTENT_RIGHT = 648;

export type SpriteSheetDef = {
  src: string;
  originX: number; // left edge of this character's half on the sheet
  frameH: number;
};

const mainSheetFor = (characterId: CharacterId): SpriteSheetDef => ({
  src: MAIN_SHEET,
  originX: characterId === 'bhuru' ? 0 : HALF_W,
  frameH: ROW_H,
});

// Imagined-mode alternate sheet: same left/right split convention assumed
// (Bhuru pilot on the left half, Heinrich wrestler on the right half).
const imaginedSheetFor = (persona: ImaginedPersona): SpriteSheetDef => ({
  src: IMAGINED_SHEET,
  originX: persona === 'bhuru-pilot' ? 0 : HALF_W,
  frameH: ROW_H,
});

export type FrameRect = { src: string; sx: number; sy: number; sw: number; sh: number };

export const frameRect = (
  sheet: SpriteSheetDef,
  row: SpriteRow,
  frameIndex: number
): FrameRect => {
  const def = ROW_FRAMES[row];
  const n = Math.max(1, def.frames);
  const idx = ((frameIndex % n) + n) % n;
  const colW = (CONTENT_RIGHT - CONTENT_LEFT) / n;
  return {
    src: sheet.src,
    sx: sheet.originX + CONTENT_LEFT + idx * colW,
    sy: rowY(row),
    sw: colW,
    sh: sheet.frameH,
  };
};

export const frameCount = (row: SpriteRow): number => ROW_FRAMES[row].frames;

export const getMainSheet = mainSheetFor;
export const getImaginedSheet = imaginedSheetFor;
