import { Renderer } from '../rendering/Renderer';
import { drawHUD } from '../rendering/HUD';
import { drawLoadingScreen } from '../ui/LoadingScreen';
import { drawStartScreen } from '../ui/StartScreen';
import { drawCharacterSelect } from '../ui/CharacterSelect';
import { drawPauseMenu } from '../ui/PauseMenu';
import { Simulation } from './Simulation';

const STEP_MS = 1000 / 120;

export const startGameLoop = (
  canvas: HTMLCanvasElement,
  renderer: Renderer,
  simulation: Simulation
): void => {
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('Canvas 2D context unavailable');

  let accumulator = 0;
  let lastTime = performance.now();

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
      drawLoadingScreen(ctx, canvas.width, canvas.height, phase.progress, frameDt);
    } else if (phase.kind === 'INTRO') {
      drawStartScreen(ctx, canvas.width, canvas.height, frameDt);
    } else if (phase.kind === 'CHARACTER_SELECT') {
      drawCharacterSelect(ctx, canvas.width, canvas.height, phase);
    } else {
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
