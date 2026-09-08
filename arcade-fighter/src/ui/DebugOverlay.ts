import type { FighterState } from '../entities/FighterState';
import { ALL_ACTION_SHAPES } from '../entities/MoveHistory';
import type { ResolutionResult } from '../physics/MoveResolver';
import type { KnowledgeGraph } from '../ai/KnowledgeGraph';

export class DebugOverlay {
  visible = false;
  private lastResolve: Record<'p1' | 'p2', ResolutionResult | null> = { p1: null, p2: null };

  toggle(): void {
    this.visible = !this.visible;
  }

  recordResolve(playerId: 'p1' | 'p2', result: ResolutionResult): void {
    this.lastResolve[playerId] = result;
  }

  draw(ctx: CanvasRenderingContext2D, p1: FighterState, p2: FighterState, knowledgeGraph?: KnowledgeGraph): void {
    if (!this.visible) return;

    ctx.font = '12px monospace';
    ctx.fillStyle = '#c9cee0';
    ctx.textAlign = 'left';

    const renderFighter = (fighter: FighterState, x: number) => {
      let y = 140;
      ctx.fillText(`${fighter.id} stamina:${fighter.stamina.toFixed(2)} composure:${fighter.composure.toFixed(2)}`, x, y);
      y += 16;
      ctx.fillText(`disorientation:${fighter.disorientation.toFixed(2)} state:${fighter.actionState.kind}`, x, y);
      y += 20;
      for (const shape of ALL_ACTION_SHAPES) {
        const entry = fighter.moveHistory[shape];
        ctx.fillText(
          `${shape}: prof=${entry.proficiency.toFixed(3)} a=${entry.attempts} s=${entry.successes} w=${entry.whiffs} inj=${entry.selfInjuries}`,
          x,
          y
        );
        y += 14;
      }
      const last = this.lastResolve[fighter.id];
      if (last) {
        y += 6;
        ctx.fillText(
          `last: ${last.shape} outcome=${last.outcome} hitChance=${last.hitChance.toFixed(2)} roll=${last.roll.toFixed(2)}`,
          x,
          y
        );
      }
    };

    renderFighter(p1, 20);
    renderFighter(p2, 620);

    if (knowledgeGraph) {
      const obs = knowledgeGraph
        .allObservations()
        .filter((o) => o.weight > 0.02)
        .sort((a, b) => b.weight - a.weight)
        .slice(0, 6)
        .map((o) => `${o.id}=${o.weight.toFixed(2)}`)
        .join('  ');
      ctx.fillText(`KG: ${obs || 'none'}`, 20, 400);
    }
  }
}
