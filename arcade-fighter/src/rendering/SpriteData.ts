import type { CharacterId } from '../entities/Character';
import type { ImaginedPersona } from '../game/ImaginedMode';

// Both characters share one sheet, split left/right at x=656 (1312px wide total).
// Row/frame rects below were measured directly off the source PNG (see
// scripts intent in git history / PR description) by scanning for non-background
// pixel bands per row and per column — the sheet is a hand-drawn reference sheet,
// not a uniform tile grid, so frame counts and widths vary per row and per
// character and cannot be computed from a formula.
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

type FrameSlice = { sx: number; sw: number };
type RowTable = { sy: number; sh: number; frames: FrameSlice[] };

// Measured main-sheet geometry, per character half. heinrich's sx values are
// already half-relative (i.e. add HALF_W separately when building the rect).
const MAIN_BHURU: Record<SpriteRow, RowTable> = {
  idle: { sy: 612, sh: 61, frames: [{ sx: 103, sw: 26 }, { sx: 154, sw: 27 }, { sx: 204, sw: 26 }, { sx: 253, sw: 26 }, { sx: 304, sw: 26 }, { sx: 356, sw: 27 }, { sx: 409, sw: 26 }] },
  walk: { sy: 674, sh: 60, frames: [{ sx: 103, sw: 26 }, { sx: 156, sw: 25 }, { sx: 211, sw: 26 }, { sx: 265, sw: 26 }, { sx: 319, sw: 27 }, { sx: 371, sw: 29 }, { sx: 421, sw: 30 }] },
  run: { sy: 738, sh: 58, frames: [{ sx: 102, sw: 28 }, { sx: 148, sw: 38 }, { sx: 202, sw: 42 }, { sx: 264, sw: 34 }, { sx: 316, sw: 44 }, { sx: 374, sw: 45 }, { sx: 429, sw: 39 }] },
  jump: { sy: 803, sh: 61, frames: [{ sx: 103, sw: 30 }, { sx: 155, sw: 26 }, { sx: 219, sw: 36 }, { sx: 277, sw: 40 }, { sx: 337, sw: 42 }, { sx: 398, sw: 35 }] },
  crouch: { sy: 880, sh: 44, frames: [{ sx: 103, sw: 26 }, { sx: 160, sw: 27 }, { sx: 217, sw: 28 }, { sx: 269, sw: 27 }, { sx: 314, sw: 27 }, { sx: 360, sw: 28 }] },
  hit: { sy: 932, sh: 54, frames: [{ sx: 103, sw: 33 }, { sx: 154, sw: 31 }, { sx: 211, sw: 49 }, { sx: 283, sw: 41 }, { sx: 336, sw: 49 }, { sx: 400, sw: 35 }] },
  hurt: { sy: 989, sh: 54, frames: [{ sx: 103, sw: 30 }, { sx: 154, sw: 33 }, { sx: 204, sw: 32 }, { sx: 255, sw: 32 }, { sx: 307, sw: 28 }, { sx: 361, sw: 63 }, { sx: 443, sw: 61 }, { sx: 527, sw: 38 }] },
  scared: { sy: 1053, sh: 53, frames: [{ sx: 116, sw: 39 }, { sx: 183, sw: 40 }, { sx: 235, sw: 33 }, { sx: 287, sw: 41 }, { sx: 350, sw: 44 }, { sx: 409, sw: 43 }] },
  special: { sy: 1116, sh: 67, frames: [{ sx: 104, sw: 26 }, { sx: 137, sw: 49 }, { sx: 198, sw: 36 }, { sx: 243, sw: 33 }, { sx: 284, sw: 37 }, { sx: 334, sw: 50 }, { sx: 389, sw: 42 }, { sx: 447, sw: 45 }, { sx: 497, sw: 49 }] },
};

const MAIN_HEINRICH: Record<SpriteRow, RowTable> = {
  idle: { sy: 612, sh: 61, frames: [{ sx: 125, sw: 26 }, { sx: 179, sw: 25 }, { sx: 232, sw: 24 }, { sx: 283, sw: 25 }, { sx: 333, sw: 24 }, { sx: 381, sw: 25 }] },
  walk: { sy: 674, sh: 60, frames: [{ sx: 125, sw: 25 }, { sx: 177, sw: 24 }, { sx: 227, sw: 25 }, { sx: 279, sw: 24 }, { sx: 329, sw: 26 }, { sx: 379, sw: 27 }] },
  run: { sy: 738, sh: 58, frames: [{ sx: 125, sw: 26 }, { sx: 169, sw: 31 }, { sx: 226, sw: 38 }, { sx: 278, sw: 43 }, { sx: 335, sw: 37 }, { sx: 388, sw: 40 }] },
  jump: { sy: 803, sh: 61, frames: [{ sx: 125, sw: 28 }, { sx: 181, sw: 31 }, { sx: 234, sw: 37 }, { sx: 287, sw: 43 }, { sx: 348, sw: 44 }] },
  crouch: { sy: 880, sh: 44, frames: [{ sx: 124, sw: 28 }, { sx: 180, sw: 29 }, { sx: 239, sw: 26 }, { sx: 289, sw: 26 }, { sx: 337, sw: 28 }] },
  hit: { sy: 932, sh: 54, frames: [{ sx: 125, sw: 27 }, { sx: 172, sw: 28 }, { sx: 220, sw: 29 }, { sx: 269, sw: 43 }, { sx: 329, sw: 38 }, { sx: 379, sw: 33 }] },
  hurt: { sy: 989, sh: 54, frames: [{ sx: 126, sw: 27 }, { sx: 180, sw: 27 }, { sx: 231, sw: 30 }, { sx: 290, sw: 45 }, { sx: 353, sw: 46 }, { sx: 420, sw: 49 }, { sx: 479, sw: 50 }, { sx: 561, sw: 27 }] },
  scared: { sy: 1053, sh: 53, frames: [{ sx: 130, sw: 33 }, { sx: 193, sw: 30 }, { sx: 254, sw: 42 }, { sx: 322, sw: 43 }, { sx: 386, sw: 29 }, { sx: 435, sw: 43 }] },
  special: { sy: 1116, sh: 67, frames: [{ sx: 166, sw: 29 }, { sx: 223, sw: 27 }, { sx: 281, sw: 26 }, { sx: 337, sw: 31 }, { sx: 395, sw: 35 }, { sx: 450, sw: 57 }] },
};

const HALF_W = 656;
const ROW_ORDER: SpriteRow[] = ['idle', 'walk', 'run', 'jump', 'crouch', 'hit', 'hurt', 'scared', 'special'];

export type SpriteSheetDef = {
  src: string;
  table: Record<SpriteRow, RowTable>;
  originX: number; // added to each frame's sx (0 for left half, HALF_W for right half)
};

const mainSheetFor = (characterId: CharacterId): SpriteSheetDef => ({
  src: MAIN_SHEET,
  table: characterId === 'bhuru' ? MAIN_BHURU : MAIN_HEINRICH,
  // MAIN_HEINRICH's sx values were measured relative to the sheet's right
  // half (heinrich's frames actually live at x >= HALF_W=656 — confirmed by
  // cropping the raw PNG) — originX must add that offset back, or Heinrich's
  // draws sample frames from inside Bhuru's own half of the sheet instead.
  originX: characterId === 'bhuru' ? 0 : HALF_W,
});

// Imagined-mode alternate sheet: geometry not yet measured against the real
// image (see MAIN_BHURU/MAIN_HEINRICH above for the measured approach) — this
// still uses an assumed uniform grid and will misrender until remeasured the
// same way. Not used by normal (non-imagined) play.
const IMAGINED_ROW_TOP = 570 + 45;
const IMAGINED_ROW_H = 73;
const IMAGINED_FRAME_COUNTS: Record<SpriteRow, number> = {
  idle: 7, walk: 7, run: 6, jump: 5, crouch: 6, hit: 5, hurt: 7, scared: 6, special: 7,
};
const imaginedRowTable = (): Record<SpriteRow, RowTable> => {
  const table = {} as Record<SpriteRow, RowTable>;
  ROW_ORDER.forEach((row, i) => {
    const n = IMAGINED_FRAME_COUNTS[row];
    const colW = (648 - 8) / n;
    table[row] = {
      sy: IMAGINED_ROW_TOP + i * IMAGINED_ROW_H,
      sh: IMAGINED_ROW_H,
      frames: Array.from({ length: n }, (_, idx) => ({ sx: 8 + idx * colW, sw: colW })),
    };
  });
  return table;
};
const IMAGINED_TABLE = imaginedRowTable();

const imaginedSheetFor = (persona: ImaginedPersona): SpriteSheetDef => ({
  src: IMAGINED_SHEET,
  table: IMAGINED_TABLE,
  originX: persona === 'bhuru-pilot' ? 0 : HALF_W,
});

export type FrameRect = { src: string; sx: number; sy: number; sw: number; sh: number };

export const frameRect = (
  sheet: SpriteSheetDef,
  row: SpriteRow,
  frameIndex: number
): FrameRect => {
  const rowTable = sheet.table[row];
  const n = Math.max(1, rowTable.frames.length);
  const idx = ((frameIndex % n) + n) % n;
  const frame = rowTable.frames[idx];
  return {
    src: sheet.src,
    sx: sheet.originX + frame.sx,
    sy: rowTable.sy,
    sw: frame.sw,
    sh: rowTable.sh,
  };
};

export const frameCount = (sheet: SpriteSheetDef, row: SpriteRow): number => sheet.table[row].frames.length;

export const getMainSheet = mainSheetFor;
export const getImaginedSheet = imaginedSheetFor;
