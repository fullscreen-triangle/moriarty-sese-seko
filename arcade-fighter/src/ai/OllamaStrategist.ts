import { KnowledgeGraph } from './KnowledgeGraph';
import type { ObservationNodeId, GoalNodeId } from './KnowledgeGraph';

/**
 * The propagation side of the same causal-propagation table (causal-propagation-table.tex):
 * KnowledgeGraph is the catalogue at rest; this is the catalogue read under propagation
 * (an LLM call). Its output residue — proposed edge/goal deltas — is merged back into the
 * graph by KnowledgeGraph.commitResidue, permanently. Runs on a slow, independent cadence,
 * off the 120Hz sim tick; a missing/slow Ollama instance degrades to "graph stops updating,"
 * never a crash or a stall of the fight itself.
 */

const OLLAMA_URL = 'http://localhost:11434/api/generate';
const MODEL = 'llama3.2';
const REQUEST_TIMEOUT_MS = 4000;
const MIN_INTERVAL_MS = 1500;

interface ResidueProposal {
  edgeUpdates?: Array<{ from: ObservationNodeId; to: ObservationNodeId | GoalNodeId; weightDelta: number }>;
  goalAdjustments?: Array<{ id: GoalNodeId; priorityDelta: number }>;
}

export class OllamaStrategist {
  private inFlight = false;
  private msSinceLast = MIN_INTERVAL_MS; // fire promptly on the first eligible tick
  private available = true;

  constructor(private readonly graph: KnowledgeGraph) {}

  /** Call every sim tick; only actually fires a request when idle and due. */
  tick(dtMs: number, triggeredNow = false): void {
    this.msSinceLast += dtMs;
    if (this.inFlight) return;
    if (!this.available) return;
    if (!triggeredNow && this.msSinceLast < MIN_INTERVAL_MS) return;

    this.msSinceLast = 0;
    this.inFlight = true;
    this.propagate().finally(() => {
      this.inFlight = false;
    });
  }

  private async propagate(): Promise<void> {
    const prompt = this.buildPrompt();
    try {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

      const response = await fetch(OLLAMA_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: MODEL, prompt, stream: false, format: 'json' }),
        signal: controller.signal,
      });
      clearTimeout(timeout);

      if (!response.ok) return;
      const body = (await response.json()) as { response?: string };
      if (!body.response) return;

      const residue = parseResidue(body.response);
      if (residue) this.graph.commitResidue(residue);
    } catch {
      // Ollama unreachable/slow/malformed — the graph simply stops updating this beat.
      this.available = false;
      setTimeout(() => {
        this.available = true;
      }, 10000);
    }
  }

  private buildPrompt(): string {
    return [
      'You are Heinrich, a bureaucratic analyst investigator fighting an anomalous opponent.',
      'You maintain a causal knowledge graph about your opponent. Given the current graph state,',
      'propose small structural updates: reweight existing causal links or standing goals based',
      'on the pattern so far. Respond with ONLY compact JSON of this shape:',
      '{"edgeUpdates":[{"from":"favors:jab","to":"punishable:jab","weightDelta":0.1}],',
      '"goalAdjustments":[{"id":"contain-anomaly","priorityDelta":0.05}]}',
      'Valid observation ids look like favors:<shape>, punishable:<shape>, overextends:<shape>,',
      'opponent:injured, opponent:low-stamina, opponent:aggressive, opponent:turtling.',
      'Valid goal ids: contain-anomaly, document-pattern, avoid-injury.',
      'weightDelta and priorityDelta must be small, in [-0.2, 0.2].',
      '',
      'Current graph:',
      this.graph.summarize(),
    ].join('\n');
  }
}

const parseResidue = (raw: string): ResidueProposal | null => {
  try {
    const parsed = JSON.parse(raw) as ResidueProposal;
    if (typeof parsed !== 'object' || parsed === null) return null;
    return parsed;
  } catch {
    return null;
  }
};
