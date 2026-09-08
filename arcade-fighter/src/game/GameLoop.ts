import { Renderer } from '../rendering/Renderer';
import { drawHUD } from '../rendering/HUD';
import { drawStartScreen } from '../ui/StartScreen';
import { drawCharacterSelect } from '../ui/CharacterSelect';
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
    accumulator += Math.min(now - lastTime, 250);
    lastTime = now;

    while (accumulator >= STEP_MS) {
      simulation.tick(STEP_MS);
      accumulator -= STEP_MS;
    }

    if (simulation.phase.kind === 'INTRO') {
      drawStartScreen(ctx, canvas.width, canvas.height);
    } else if (simulation.phase.kind === 'CHARACTER_SELECT') {
      drawCharacterSelect(ctx, canvas.width, canvas.height, simulation.phase);
    } else {
      const shake = simulation.cameraShake.offset();
      ctx.save();
      ctx.translate(shake.x, shake.y);
      renderer.draw([simulation.p1, simulation.p2], simulation.imaginedMode);
      ctx.restore();
      drawHUD(ctx, canvas.width, simulation.p1, simulation.p2, simulation.phase);
      simulation.debugOverlay.draw(ctx, simulation.p1, simulation.p2, simulation.knowledgeGraph);
    }

    requestAnimationFrame(frame);
  };

  requestAnimationFrame(frame);
};
