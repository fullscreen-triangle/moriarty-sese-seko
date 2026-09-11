import { ROSTER, type CharacterId, type StatBias } from '../entities/Character';

// Lore text expanded from the game's actual roster data (ROSTER in
// entities/Character.ts is the single source of truth for name/tagline/color)
// plus prose grounded in what the fighters' real StatBias multipliers do in a
// match. Nothing here is decorative flavor unconnected to mechanics — every
// readout row below is derived from a number MoveResolver/StaminaModel/
// ComposureModel/InjuryModel actually reads.

export const CONSEQUENCE_TIERS = [
  { n: '01', t: 'Social isolation, communication breakdown' },
  { n: '02', t: 'Physical side effects — tremor, stammer, drooling' },
  { n: '03', t: 'Quantum consciousness burden — every timeline felt at once' },
  { n: '04', t: 'Reality distortion in a radius around the violator' },
  { n: '05', t: 'Cosmic-scale thermodynamic rebalancing' },
];

type ReadoutRow = { k: string; v: string };

const biasReadouts = (bias: StatBias): ReadoutRow[] => [
  {
    k: 'Reach',
    v:
      bias.reach >= 1
        ? `+${Math.round((bias.reach - 1) * 100)}% effective range`
        : `${Math.round((bias.reach - 1) * 100)}% effective range`,
  },
  {
    k: 'Learn rate',
    v:
      bias.learnRate >= 1
        ? `${bias.learnRate.toFixed(2)}x — proficiency builds fast`
        : `${bias.learnRate.toFixed(2)}x — proficiency builds slow`,
  },
  {
    k: 'Composure regen',
    v:
      bias.composureRegen >= 1
        ? `${bias.composureRegen.toFixed(2)}x — recovers poise fast`
        : `${bias.composureRegen.toFixed(2)}x — recovers poise slow`,
  },
  {
    k: 'Injury risk',
    v:
      bias.injuryRisk >= 1
        ? `${bias.injuryRisk.toFixed(2)}x — elevated self-injury on overextension`
        : `${bias.injuryRisk.toFixed(2)}x — reduced self-injury on overextension`,
  },
];

export type FighterContent = {
  id: CharacterId;
  name: string;
  callsign: string;
  tag: string;
  archetype: string;
  bio: string;
  quote: string;
  stats: ReadoutRow[];
  system: string;
  color: string;
};

export const FIGHTERS: Record<CharacterId, FighterContent> = {
  bhuru: {
    id: 'bhuru',
    name: ROSTER.bhuru.name,
    callsign: 'The Seer. The Wrestler. The Unstable Mind.',
    tag: 'SUBJECT 01 — SANGOMA / QUANTUM',
    archetype: 'Precision technician — high risk, high reward',
    bio:
      "A ski mask, a beaded necklace, and a knife that isn't there until entropy says it is. Bhuru-sukurin doesn't plan a combo — he watches thousands of timelines resolve at once and picks the frame that already won. The cost is a mind that never fully belongs to him alone: his reach and technique run ahead of his composure, which recovers slower than it should.",
    quote:
      '"Timeline 47,297: why do I know thith ith a fourteenth-thentury grappling counter?"',
    stats: biasReadouts(ROSTER.bhuru.statBias),
    system:
      'Every knife angle, feint, or counter you invent in a match is logged to his combat graph and reinforced next time you play him — nothing is preloaded, and nothing you build is ever wiped.',
    color: ROSTER.bhuru.color,
  },
  heinrich: {
    id: 'heinrich',
    name: ROSTER.heinrich.name,
    callsign: 'The Analyst. The Bureaucrat. The Accidental Fighter.',
    tag: 'SUBJECT 02 — PHARMACEUTICAL / WRESTLING',
    archetype: 'Pressure grappler — commit and close',
    bio:
      "A pharmaceutical executive who treats the ring like a hostile boardroom takeover. Suplex, clothesline, submission — Heinrich doesn't need forty timelines, he needs one good grip. His reach is shorter and his risk of self-injury lower: what he lacks in quantum insight he makes up for by learning faster and recovering poise quicker than his opponent.",
    quote:
      '"CORPORATE EPISTEMOLOGICAL BREACH! You\'re accessing our prediction algorithms!"',
    stats: biasReadouts(ROSTER.heinrich.statBias),
    system:
      'Every throw, chain, or reversal you land with him is remembered and refined match to match — his moveset is entirely the one you build, not one we shipped.',
    color: ROSTER.heinrich.color,
  },
};

export const COMPARE_ROWS: Array<[string, string, string]> = [
  ['Reach', FIGHTERS.bhuru.stats[0].v, FIGHTERS.heinrich.stats[0].v],
  ['Learn rate', FIGHTERS.bhuru.stats[1].v, FIGHTERS.heinrich.stats[1].v],
  ['Composure regen', FIGHTERS.bhuru.stats[2].v, FIGHTERS.heinrich.stats[2].v],
  ['Injury risk', FIGHTERS.bhuru.stats[3].v, FIGHTERS.heinrich.stats[3].v],
  ['Learns from', 'Every timeline you generate', 'Every grip you land'],
  ['Best punished by', 'Commitment, closing distance', 'Patience, baiting overextension'],
  ['Signature law violation', 'Entropy (information)', 'Momentum (mass)'],
];
