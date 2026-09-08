import type { ActionShape } from '../entities/MoveHistory';

/**
 * Heinrich's per-match "resting catalogue" (causal-propagation-table.tex: the
 * periodic catalogue is the table evaluated at rest). Read every tick, cheaply,
 * no LLM involved. OllamaStrategist is the same table under propagation; its
 * output is the residue that commits back into these nodes/edges — permanently,
 * never withdrawn (see Theorem[Floored, irreversible response] in
 * non-playing-character-physiology.tex).
 */

export type ObservationNodeId =
  | `favors:${ActionShape}`
  | `punishable:${ActionShape}`
  | `overextends:${ActionShape}`
  | 'opponent:injured'
  | 'opponent:low-stamina'
  | 'opponent:aggressive'
  | 'opponent:turtling';

export type GoalNodeId = 'contain-anomaly' | 'document-pattern' | 'avoid-injury';

export interface ObservationNode {
  id: ObservationNodeId;
  weight: number; // confidence/salience, [0, 1], monotone non-decreasing per Theorem[Floored, irreversible response]
  lastUpdatedMs: number;
}

export interface GoalNode {
  id: GoalNodeId;
  priority: number; // [0, 1], standing — adjusted only by residue commits, never by the fast policy
}

/** Directed weighted edge: cause -> effect, e.g. favors:jab -> punishable:jab. */
export interface Edge {
  from: ObservationNodeId;
  to: ObservationNodeId | GoalNodeId;
  weight: number; // [-1, 1]
}

const RESIDUE_FLOOR = 0.02; // floorc: no committed update is free/zero — this is beta.

export class KnowledgeGraph {
  private observations = new Map<ObservationNodeId, ObservationNode>();
  private goals = new Map<GoalNodeId, GoalNode>();
  private edges: Edge[] = [];

  constructor() {
    this.goals.set('contain-anomaly', { id: 'contain-anomaly', priority: 0.6 });
    this.goals.set('document-pattern', { id: 'document-pattern', priority: 0.3 });
    this.goals.set('avoid-injury', { id: 'avoid-injury', priority: 0.5 });
  }

  /** Cheap per-tick observation reinforcement — the resting read, not a propagation. */
  observe(id: ObservationNodeId, deltaWeight: number, nowMs: number): void {
    const existing = this.observations.get(id);
    if (existing) {
      existing.weight = clamp01(existing.weight + deltaWeight);
      existing.lastUpdatedMs = nowMs;
    } else {
      this.observations.set(id, { id, weight: clamp01(Math.abs(deltaWeight)), lastUpdatedMs: nowMs });
    }
  }

  weightOf(id: ObservationNodeId): number {
    return this.observations.get(id)?.weight ?? 0;
  }

  goalPriority(id: GoalNodeId): number {
    return this.goals.get(id)?.priority ?? 0;
  }

  allObservations(): ObservationNode[] {
    return Array.from(this.observations.values());
  }

  allGoals(): GoalNode[] {
    return Array.from(this.goals.values());
  }

  allEdges(): Edge[] {
    return this.edges.slice();
  }

  /**
   * Merge propagation residue (OllamaStrategist output) into the resting graph.
   * Irreversible: existing edges are only reweighted toward the proposal (never
   * removed), and every commit deposits at least RESIDUE_FLOOR of change —
   * mirrors Theorem[Floored, irreversible response].
   */
  commitResidue(residue: {
    edgeUpdates?: Array<{ from: ObservationNodeId; to: ObservationNodeId | GoalNodeId; weightDelta: number }>;
    goalAdjustments?: Array<{ id: GoalNodeId; priorityDelta: number }>;
  }): void {
    for (const update of residue.edgeUpdates ?? []) {
      const delta = signedFloor(update.weightDelta, RESIDUE_FLOOR);
      if (delta === 0) continue;
      const existing = this.edges.find((e) => e.from === update.from && e.to === update.to);
      if (existing) {
        existing.weight = clamp(-1, 1, existing.weight + delta);
      } else {
        this.edges.push({ from: update.from, to: update.to, weight: clamp(-1, 1, delta) });
      }
    }

    for (const adj of residue.goalAdjustments ?? []) {
      const delta = signedFloor(adj.priorityDelta, RESIDUE_FLOOR);
      if (delta === 0) continue;
      const goal = this.goals.get(adj.id);
      if (goal) goal.priority = clamp01(goal.priority + delta);
    }
  }

  /** Compact summary for OllamaStrategist's prompt — the resting table snapshot handed to propagation. */
  summarize(): string {
    const obs = this.allObservations()
      .filter((o) => o.weight > 0.05)
      .sort((a, b) => b.weight - a.weight)
      .slice(0, 8)
      .map((o) => `${o.id}=${o.weight.toFixed(2)}`)
      .join(', ');
    const goals = this.allGoals()
      .map((g) => `${g.id}=${g.priority.toFixed(2)}`)
      .join(', ');
    const edges = this.edges
      .slice(-8)
      .map((e) => `${e.from}->${e.to}:${e.weight.toFixed(2)}`)
      .join(', ');
    return `observations: ${obs || 'none'}\ngoals: ${goals}\nedges: ${edges || 'none'}`;
  }
}

const clamp = (min: number, max: number, v: number): number => Math.min(max, Math.max(min, v));
const clamp01 = (v: number): number => clamp(0, 1, v);

/** A residue delta below the floor is not a real deposit — round it away rather than committing noise. */
const signedFloor = (delta: number, floor: number): number => {
  if (Math.abs(delta) < floor) return 0;
  return delta;
};
