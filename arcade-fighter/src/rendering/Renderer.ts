import type { FighterState } from '../entities/FighterState';
import { FLOOR_PIXEL_MARGIN } from './StickFigureRig';
import { drawFighterSprite } from './SpriteRig';
import type { ImaginedMode } from '../game/ImaginedMode';

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

  draw(fighters: FighterState[], imaginedMode?: ImaginedMode, dtMs = 16): void {
    const ctx = this.ctx;
    ctx.clearRect(0, 0, this.width, this.height);

    const floorY = this.height - FLOOR_PIXEL_MARGIN;
    const imagined = imaginedMode?.isActive() ?? false;

    if (imagined) {
      const gradient = ctx.createRadialGradient(
        this.width / 2, this.height / 2, this.height * 0.15,
        this.width / 2, this.height / 2, this.height * 0.75
      );
      gradient.addColorStop(0, 'rgba(120, 70, 200, 0.08)');
      gradient.addColorStop(1, 'rgba(20, 5, 40, 0.55)');
      ctx.fillStyle = gradient;
      ctx.fillRect(0, 0, this.width, this.height);
    }

    ctx.strokeStyle = imagined ? '#6a4fb8' : '#3a3f50';
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(0, floorY);
    ctx.lineTo(this.width, floorY);
    ctx.stroke();

    // Fighter positions are in arena meters with x=0 at stage center; the sprite
    // rig converts meters to pixels but has no notion of the canvas itself, so
    // the horizontal recenter has to happen here via a translate before drawing.
    ctx.save();
    ctx.translate(this.width / 2, 0);
    for (const fighter of fighters) {
      const persona = imagined ? imaginedMode!.personaFor(fighter.characterId) : undefined;
      drawFighterSprite(ctx, fighter, floorY, dtMs, persona);
    }
    ctx.restore();

    if (imagined) {
      ctx.fillStyle = 'rgba(210, 190, 255, 0.85)';
      ctx.font = 'italic 16px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('...imagined...', this.width / 2, 28);
      ctx.textAlign = 'left';
    }
  }
}
