import { ROSTER } from '../entities/Character';
import { getImage, isImageReady } from '../rendering/ImageCache';

const bhuru = ROSTER.bhuru;
const heinrich = ROSTER.heinrich;

let animT = 0;

const drawFighterPortrait = (
  ctx: CanvasRenderingContext2D,
  def: typeof bhuru,
  cx: number,
  baseY: number,
  maxH: number,
  flip: boolean,
  bob: number
): void => {
  const img = getImage(def.portrait.src);
  if (!isImageReady(img)) return;

  const scale = maxH / def.portrait.sh;
  const destW = def.portrait.sw * scale;
  const destH = def.portrait.sh * scale;

  ctx.save();
  ctx.translate(cx, baseY + bob);
  if (flip) ctx.scale(-1, 1);
  ctx.drawImage(
    img,
    def.portrait.sx,
    def.portrait.sy,
    def.portrait.sw,
    def.portrait.sh,
    -destW / 2,
    -destH,
    destW,
    destH
  );
  ctx.restore();
};

export const drawStartScreen = (ctx: CanvasRenderingContext2D, width: number, height: number, dtMs = 16): void => {
  animT += dtMs;

  ctx.fillStyle = '#0c0d14';
  ctx.fillRect(0, 0, width, height);

  const bandY = height * 0.32;
  const bandH = height * 0.4;
  const grad = ctx.createLinearGradient(0, bandY, 0, bandY + bandH);
  grad.addColorStop(0, 'rgba(94, 194, 255, 0.12)');
  grad.addColorStop(0.5, 'rgba(232, 234, 242, 0.06)');
  grad.addColorStop(1, 'rgba(255, 107, 107, 0.12)');
  ctx.fillStyle = grad;
  ctx.fillRect(0, bandY, width, bandH);

  const bob = Math.sin(animT / 260) * 6;
  const portraitH = Math.min(height * 0.5, 420);
  const groundY = height * 0.86;

  drawFighterPortrait(ctx, bhuru, width * 0.28, groundY, portraitH, false, bob);
  drawFighterPortrait(ctx, heinrich, width * 0.72, groundY, portraitH, true, -bob);

  ctx.textAlign = 'center';
  ctx.fillStyle = '#e8eaf2';
  ctx.font = 'bold 40px monospace';
  ctx.fillText('BHURU-SUKURIN', width * 0.28, height * 0.2);
  ctx.font = 'bold 40px monospace';
  ctx.fillText('HEINRICH', width * 0.72, height * 0.2);

  ctx.font = 'bold 64px monospace';
  ctx.fillStyle = '#ffd166';
  const vsScale = 1 + Math.sin(animT / 220) * 0.05;
  ctx.save();
  ctx.translate(width / 2, height * 0.24);
  ctx.scale(vsScale, vsScale);
  ctx.fillText('VS', 0, 0);
  ctx.restore();

  ctx.font = 'bold 44px monospace';
  ctx.fillStyle = '#e8eaf2';
  ctx.fillText('CHAPTER ONE', width / 2, height * 0.1);

  ctx.font = '15px monospace';
  ctx.fillStyle = '#c8ccdc';
  ctx.fillText('No movesets. Only what you\'ve practiced.', width / 2, height * 0.1 + 30);

  ctx.font = 'bold 18px monospace';
  ctx.fillStyle = '#8bd17c';
  ctx.fillText('Press 1 for ONE PLAYER (vs. AI Heinrich)', width / 2, height * 0.93);
  ctx.fillText('Press 2 for TWO PLAYERS (same keyboard)', width / 2, height * 0.93 + 26);

  ctx.font = '12px monospace';
  ctx.fillStyle = '#8890a8';
  ctx.fillText('Esc pauses mid-fight for full controls   •   Tab: debug overlay', width / 2, height * 0.93 + 56);

  ctx.textAlign = 'left';
};
