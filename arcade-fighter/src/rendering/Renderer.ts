import type { FighterState } from '../entities/FighterState';
import { drawFighter, FLOOR_PIXEL_MARGIN } from './StickFigureRig';
import { ROSTER } from '../entities/Character';

export class Renderer {
  private ctx: CanvasRenderingContext2D;
  private width: number;
  private height: number;

  constructor(canvas: HTMLCanvasElement) {
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('Canvas 2D context unavailable');
    this.ctx = ctx;
    this.width = canvas.width;
    this.height = canvas.height;
  }

  draw(fighters: FighterState[]): void {
    const ctx = this.ctx;
    ctx.clearRect(0, 0, this.width, this.height);

    const floorY = this.height - FLOOR_PIXEL_MARGIN;

    ctx.strokeStyle = '#3a3f50';
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(0, floorY);
    ctx.lineTo(this.width, floorY);
    ctx.stroke();

    for (const fighter of fighters) {
      drawFighter(ctx, fighter, floorY, ROSTER[fighter.characterId].color);
    }
  }
}
