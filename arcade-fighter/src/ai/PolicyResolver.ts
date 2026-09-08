import type { FighterState } from '../entities/FighterState';
import type { ActionIntent, IntentType } from '../input/ActionIntent';
import type { ActionShape } from '../entities/MoveHistory';
import { REACH } from '../physics/Kinematics';
import { ROSTER } from '../entities/Character';
import type { StatBias } from '../entities/Character';
import type { RNG } from '../physics/RNG';
import { KnowledgeGraph } from './KnowledgeGraph';

/**
 * Fast per-tick policy for Heinrich, shaped by non-playing-character-physiology.tex:
 * - Axiom[Response floor]: no response is free — every committed action costs >= floorc.
 * - Axiom[Bounded attention]: a fixed attention budget per unit time caps how often we
 *   even evaluate committing to a substantive action.
 * - Theorem[Optimal policy is a threshold on gain density]: score candidates by
 *   gain/cost, sort, admit only those clearing a single price p*.
 * - Corollary[token boundary]: when nothing clears p*, fall back to the cheapest
 *   floor-cost response (guard) rather than idling — never truly idle.
 *
 * Reads KnowledgeGraph (the resting catalogue) plus MoveResolver-consistent terms
 * (REACH, proficiency, StatBias.reach) to weight candidates; never calls the LLM.
 */

const ATTENTION_BUDGET_PER_SEC = 2.2; // att: substantive-decision budget per second
const FLOOR_COST = 0.15; // floorc: cost of the guaranteed minimal (block) response
const GAIN_DENSITY_THRESHOLD = 0.35; // p*: candidates must clear this gain/cost ratio

type Candidate = {
  type: IntentType;
  shape: ActionShape;
  chargeMs: number;
  cost: number;
  gain: number;
};

export class PolicyResolver {
  private attentionRemaining = ATTENTION_BUDGET_PER_SEC;

  constructor(private readonly graph: KnowledgeGraph) {}

  /** Call once per sim tick before drawing an intent. Recharges the attention budget. */
  refillAttention(dtMs: number): void {
    this.attentionRemaining = Math.min(
      ATTENTION_BUDGET_PER_SEC,
      this.attentionRemaining + ATTENTION_BUDGET_PER_SEC * (dtMs / 1000)
    );
  }

  /**
   * Update the resting catalogue from this tick's observed fight state. Cheap,
   * synchronous, no LLM — this is the "read the table at rest" step.
   */
  observe(self: FighterState, opponent: FighterState, nowMs: number): void {
    if (opponent.actionState.kind === 'active' || opponent.actionState.kind === 'recovery') {
      const shape = opponent.actionState.shape;
      this.graph.observe(`favors:${shape}`, 0.04, nowMs);
    }
    for (const limbId of ['leftArm', 'rightArm', 'leftLeg', 'rightLeg'] as const) {
      if (opponent.limbs[limbId].injurySeverity > 0.3) {
        this.graph.observe('opponent:injured', 0.05, nowMs);
      }
    }
    if (opponent.stamina < 0.35) this.graph.observe('opponent:low-stamina', 0.05, nowMs);
    if (opponent.blocking) this.graph.observe('opponent:turtling', 0.03, nowMs);
    if (self.composure < opponent.composure) this.graph.observe('opponent:aggressive', 0.02, nowMs);
  }

  /**
   * Emit an intent for `self` (Heinrich), or null if the action-state gate
   * (idle/blocking only) isn't open — mirrors Simulation.tryStartAction's own gate.
   */
  resolve(self: FighterState, opponent: FighterState, rng: RNG): ActionIntent | null {
    if (self.actionState.kind !== 'idle' && self.actionState.kind !== 'blocking') return null;

    const distance = Math.abs(opponent.position.x - self.position.x);
    const bias = ROSTER[self.characterId].statBias;

    const candidates = this.buildCandidates(self, opponent, distance, bias);
    const affordable = candidates.filter((c) => c.cost <= this.attentionRemaining);

    let best: Candidate | null = null;
    let bestDensity = -Infinity;
    for (const c of affordable) {
      const density = c.gain / c.cost;
      if (density > bestDensity) {
        bestDensity = density;
        best = c;
      }
    }

    if (best && bestDensity >= GAIN_DENSITY_THRESHOLD) {
      this.attentionRemaining -= best.cost;
      return {
        type: best.type,
        direction: 'neutral',
        chargeMs: best.chargeMs,
        playerId: self.id,
      };
    }

    // Token boundary: nothing clears p*, or attention is exhausted — take the
    // floor-cost response (guard) if it itself clears the bar, else stay put.
    if (this.attentionRemaining >= FLOOR_COST) {
      this.attentionRemaining -= FLOOR_COST;
      const wantsGuard = distance < 1.8 && (opponent.actionState.kind === 'startup' || rng.next() < 0.3);
      if (wantsGuard) {
        return { type: 'block', direction: 'neutral', chargeMs: 0, playerId: self.id };
      }
    }

    return null;
  }

  private buildCandidates(self: FighterState, opponent: FighterState, distance: number, bias: StatBias): Candidate[] {
    const shapes: Array<{ type: IntentType; shape: ActionShape; charge: boolean }> = [
      { type: 'punch', shape: 'jab', charge: false },
      { type: 'punch', shape: 'hook', charge: true },
      { type: 'kick', shape: 'lowKick', charge: false },
      { type: 'kick', shape: 'roundhouseKick', charge: true },
    ];

    const goalWeight =
      this.graph.goalPriority('contain-anomaly') * 0.6 + this.graph.goalPriority('document-pattern') * 0.4;

    return shapes.map(({ type, shape, charge }) => {
      const reach = REACH[shape] * bias.reach;
      const inRange = distance <= reach + 0.3;
      const proficiency = self.moveHistory[shape].proficiency;

      const punishSignal = this.graph.weightOf(`punishable:${shape}`);
      const favoredCounterSignal = this.graph.weightOf(`favors:${shape}`) * 0.5;
      const openOpponent =
        this.graph.weightOf('opponent:low-stamina') * 0.3 +
        this.graph.weightOf('opponent:injured') * 0.4 +
        (opponent.actionState.kind === 'recovery' ? 0.3 : 0);

      const rangeGain = inRange ? 1 : 0.15;
      const gain =
        rangeGain *
        (0.4 + proficiency * 0.5 + punishSignal + favoredCounterSignal + openOpponent) *
        (0.6 + goalWeight);

      const cost = FLOOR_COST + (charge ? 0.35 : 0.2) * (1 - proficiency * 0.3);

      return { type, shape, chargeMs: charge ? 280 : 0, cost, gain };
    });
  }
}
