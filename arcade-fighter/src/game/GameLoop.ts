import { Renderer } from '../rendering/Renderer';
import { drawHUD } from '../rendering/HUD';
import { drawPauseMenu } from '../ui/PauseMenu';
import { Simulation } from './Simulation';
import type { DossierHandle } from '../ui/Dossier';

const STEP_MS = 1000 / 120;

export const startGameLoop = (
  canvas: HTMLCanvasElement,
  renderer: Renderer,
  simulation: Simulation,
  dossier: DossierHandle
): void => {
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('Canvas 2D context unavailable');

  let accumulator = 0;
  let lastTime = performance.now();
  let dossierDestroyed = false;

  const frame = (now: number) => {
    const frameDt = Math.min(now - lastTime, 250);
    accumulator += frameDt;
    lastTime = now;

    while (accumulator >= STEP_MS) {
      simulation.tick(STEP_MS);
      accumulator -= STEP_MS;
    }

    const phase = simulation.phase;
    const fightPhase = phase.kind === 'PAUSED' ? phase.previous : phase;

    if (phase.kind === 'LOADING') {
      dossier.showLoading(phase.progress);
    } else if (phase.kind === 'INTRO' || phase.kind === 'CHARACTER_SELECT') {
      dossier.showSelect();
    } else {
      if (!dossierDestroyed) {
        dossier.destroy();
        dossierDestroyed = true;
      }
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const shake = simulation.cameraShake.offset();
      ctx.save();
      ctx.translate(shake.x, shake.y);
      renderer.draw([simulation.p1, simulation.p2], simulation.imaginedMode, frameDt);
      ctx.restore();
      drawHUD(ctx, canvas.width, simulation.p1, simulation.p2, fightPhase);
      simulation.debugOverlay.draw(ctx, simulation.p1, simulation.p2, simulation.knowledgeGraph);
      if (phase.kind === 'PAUSED') {
        drawPauseMenu(ctx, canvas.width, canvas.height);
      }
    }

    requestAnimationFrame(frame);
  };

  requestAnimationFrame(frame);
};
