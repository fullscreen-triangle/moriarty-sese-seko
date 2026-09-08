import type { FighterState } from '../entities/FighterState';
import { ALL_LIMB_IDS } from '../entities/LimbState';
import type { MatchPhase } from '../game/GameState';

const BAR_WIDTH = 280;
const BAR_HEIGHT = 14;
const MARGIN = 24;

const drawBar = (
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  width: number,
  fillRatio: number,
  color: string,
  flip: boolean
): void => {
  ctx.fillStyle = '#1c1f2a';
  ctx.fillRect(x, y, width, BAR_HEIGHT);
  ctx.fillStyle = color;
  const fillWidth = Math.max(0, Math.min(1, fillRatio)) * width;
  if (flip) {
    ctx.fillRect(x + width - fillWidth, y, fillWidth, BAR_HEIGHT);
  } else {
    ctx.fillRect(x, y, fillWidth, BAR_HEIGHT);
  }
  ctx.strokeStyle = '#565d73';
  ctx.lineWidth = 1;
  ctx.strokeRect(x, y, width, BAR_HEIGHT);
};

const drawInjuryDots = (ctx: CanvasRenderingContext2D, x: number, y: number, fighter: FighterState, flip: boolean): void => {
  const dotRadius = 5;
  const spacing = 16;
  ALL_LIMB_IDS.forEach((limbId, i) => {
    const limb = fighter.limbs[limbId];
    const cx = flip ? x - i * spacing : x + i * spacing;
    let color = '#3d4356';
    if (limb.injurySeverity > 0.5) color = '#ff4d4d';
    else if (limb.injurySeverity > 0.2) color = '#ffb74d';
    else if (limb.injurySeverity > 0) color = '#fff176';
    ctx.beginPath();
    ctx.arc(cx, y, dotRadius, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
  });
};

export const drawHUD = (
  ctx: CanvasRenderingContext2D,
  width: number,
  p1: FighterState,
  p2: FighterState,
  phase: MatchPhase
): void => {
  drawBar(ctx, MARGIN, MARGIN, BAR_WIDTH, p1.composure, '#5ec2ff', false);
  drawBar(ctx, MARGIN, MARGIN + BAR_HEIGHT + 4, BAR_WIDTH, p1.stamina, '#8bd17c', false);
  drawInjuryDots(ctx, MARGIN, MARGIN + (BAR_HEIGHT + 4) * 2 + 10, p1, false);

  const rightX = width - MARGIN - BAR_WIDTH;
  drawBar(ctx, rightX, MARGIN, BAR_WIDTH, p2.composure, '#ff6b6b', true);
  drawBar(ctx, rightX, MARGIN + BAR_HEIGHT + 4, BAR_WIDTH, p2.stamina, '#8bd17c', true);
  drawInjuryDots(ctx, width - MARGIN, MARGIN + (BAR_HEIGHT + 4) * 2 + 10, p2, true);

  if (phase.kind === 'FIGHTING') {
    const seconds = Math.ceil(phase.remainingMs / 1000);
    ctx.fillStyle = '#e8eaf2';
    ctx.font = '24px monospace';
    ctx.textAlign = 'center';
    ctx.fillText(String(seconds), width / 2, MARGIN + 24);
  } else if (phase.kind === 'ROUND_END') {
    ctx.fillStyle = '#e8eaf2';
    ctx.font = '32px monospace';
    ctx.textAlign = 'center';
    const label =
      phase.winner === 'draw'
        ? 'DRAW'
        : `${phase.winner.toUpperCase()} WINS (${phase.reason === 'stumble' ? 'stumble' : 'timer'})`;
    ctx.fillText(label, width / 2, width * 0 + 80);
  } else if (phase.kind === 'FLED') {
    ctx.fillStyle = '#e8eaf2';
    ctx.font = '28px monospace';
    ctx.textAlign = 'center';
    ctx.fillText(`${phase.fledPlayer.toUpperCase()} FLED THE FIGHT`, width / 2, 80);
  }
  ctx.textAlign = 'left';
};
