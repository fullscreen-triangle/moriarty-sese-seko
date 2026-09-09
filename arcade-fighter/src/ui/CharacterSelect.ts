import { ROSTER } from '../entities/Character';
import type { MatchPhase } from '../game/GameState';
import { getImage, isImageReady } from '../rendering/ImageCache';

const ORDER = ['bhuru', 'heinrich'] as const;

export const drawCharacterSelect = (
  ctx: CanvasRenderingContext2D,
  width: number,
  _height: number,
  phase: Extract<MatchPhase, { kind: 'CHARACTER_SELECT' }>
): void => {
  ctx.fillStyle = '#e8eaf2';
  ctx.textAlign = 'center';

  ctx.font = 'bold 32px monospace';
  ctx.fillText('SELECT YOUR FIGHTER', width / 2, 70);

  ctx.font = '14px monospace';
  ctx.fillText(
    phase.mode === '1P_VS_AI' ? '1 PLAYER — vs. AI Heinrich' : '2 PLAYERS — same keyboard',
    width / 2,
    100
  );

  const cardW = 260;
  const cardH = 320;
  const gap = 40;
  const totalW = cardW * ORDER.length + gap * (ORDER.length - 1);
  const startX = width / 2 - totalW / 2;
  const y = 140;

  ORDER.forEach((id, i) => {
    const def = ROSTER[id];
    const x = startX + i * (cardW + gap);
    const p1Pick = phase.picks.p1 === id;
    const p2Pick = phase.picks.p2 === id;

    ctx.strokeStyle = p1Pick || p2Pick ? def.color : '#3a3f50';
    ctx.lineWidth = p1Pick || p2Pick ? 4 : 2;
    ctx.strokeRect(x, y, cardW, cardH);

    const portraitTop = y + 30;
    const portraitH = cardH - 30 - 40;
    const img = getImage(def.portrait.src);
    if (isImageReady(img)) {
      const scale = Math.min(
        (cardW - 20) / def.portrait.sw,
        portraitH / def.portrait.sh
      );
      const destW = def.portrait.sw * scale;
      const destH = def.portrait.sh * scale;
      ctx.drawImage(
        img,
        def.portrait.sx,
        def.portrait.sy,
        def.portrait.sw,
        def.portrait.sh,
        x + (cardW - destW) / 2,
        portraitTop + (portraitH - destH) / 2,
        destW,
        destH
      );
    }

    ctx.fillStyle = def.color;
    ctx.font = 'bold 18px monospace';
    ctx.fillText(def.name, x + cardW / 2, y + cardH - 58);

    ctx.fillStyle = '#c8ccdc';
    ctx.font = '10px monospace';
    wrapText(ctx, def.tagline, x + cardW / 2, y + cardH - 42, cardW - 20, 12);

    ctx.font = '13px monospace';
    ctx.fillStyle = '#8890a8';
    ctx.fillText(`Press ${i + 1}`, x + cardW / 2, y + cardH - 10);

    const tags: string[] = [];
    if (p1Pick) tags.push('P1');
    if (p2Pick) tags.push(phase.mode === '1P_VS_AI' ? 'HEINRICH (AI)' : 'P2');
    if (tags.length > 0) {
      ctx.fillStyle = def.color;
      ctx.font = 'bold 14px monospace';
      ctx.fillText(tags.join(' / '), x + cardW / 2, y + 20);
    }
  });

  ctx.fillStyle = '#e8eaf2';
  ctx.font = '13px monospace';
  const hint =
    phase.mode === '1P_VS_AI'
      ? 'Press 1 or 2 to pick your fighter — Heinrich (AI) takes the other automatically.'
      : 'P1 press 1/2 to pick — P2 press 1/2 to pick the remaining fighter.';
  ctx.fillText(hint, width / 2, y + cardH + 50);

  ctx.textAlign = 'left';
};

const wrapText = (
  ctx: CanvasRenderingContext2D,
  text: string,
  cx: number,
  y: number,
  maxWidth: number,
  lineHeight: number
): void => {
  const words = text.split(' ');
  let line = '';
  let lineY = y;
  for (const word of words) {
    const test = line.length > 0 ? `${line} ${word}` : word;
    if (ctx.measureText(test).width > maxWidth && line.length > 0) {
      ctx.fillText(line, cx, lineY);
      line = word;
      lineY += lineHeight;
    } else {
      line = test;
    }
  }
  if (line.length > 0) ctx.fillText(line, cx, lineY);
};
