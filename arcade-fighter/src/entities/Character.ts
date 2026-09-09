export type CharacterId = 'bhuru' | 'heinrich';

export type StatBias = {
  // Multiplies effective reach in MoveResolver (power/instability vs. technique tradeoff).
  reach: number;
  // Multiplies proficiency growth rate (how fast a shape's MoveHistory improves).
  learnRate: number;
  // Multiplies composure regen (poise recovery between hits).
  composureRegen: number;
  // Multiplies self-injury chance on overextension/miss.
  injuryRisk: number;
};

export type CharacterDef = {
  id: CharacterId;
  name: string;
  tagline: string;
  color: string;
  portrait: { src: string; sx: number; sy: number; sw: number; sh: number };
  statBias: StatBias;
};

// Portrait crops are the full-body turnaround hero pose on each character's half
// of the reference sheet (see public/ChatGPT Image 8. Sept. 2026, 06_55_43.png),
// measured directly against the source image — used for character-select and the
// title-screen VS promo. In-match rendering uses the animated sprite rows below
// this hero pose (see rendering/SpriteData.ts).
const SPRITE_SHEET = '/ChatGPT Image 8. Sept. 2026, 06_55_43.png';

export const ROSTER: Record<CharacterId, CharacterDef> = {
  bhuru: {
    id: 'bhuru',
    name: 'Bhuru-sukurin',
    tagline: 'The Seer. The Wrestler. The Unstable Mind.',
    color: '#5ec2ff',
    portrait: { src: SPRITE_SHEET, sx: 15, sy: 95, sw: 235, sh: 470 },
    statBias: {
      reach: 1.08,
      learnRate: 0.9,
      composureRegen: 0.85,
      injuryRisk: 1.2,
    },
  },
  heinrich: {
    id: 'heinrich',
    name: 'Heinrich',
    tagline: 'The Analyst. The Bureaucrat. The Accidental Fighter.',
    color: '#ff6b6b',
    portrait: { src: SPRITE_SHEET, sx: 671, sy: 95, sw: 235, sh: 470 },
    statBias: {
      reach: 0.95,
      learnRate: 1.15,
      composureRegen: 1.1,
      injuryRisk: 0.85,
    },
  },
};

export const otherCharacter = (id: CharacterId): CharacterId => (id === 'bhuru' ? 'heinrich' : 'bhuru');
