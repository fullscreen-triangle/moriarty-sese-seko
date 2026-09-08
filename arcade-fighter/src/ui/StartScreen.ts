export const drawStartScreen = (ctx: CanvasRenderingContext2D, width: number, height: number): void => {
  ctx.fillStyle = '#e8eaf2';
  ctx.textAlign = 'center';

  ctx.font = 'bold 40px monospace';
  ctx.fillText('ARCADE FIGHTER', width / 2, height / 2 - 110);

  ctx.font = '16px monospace';
  ctx.fillText('No movesets. Only what you\'ve practiced.', width / 2, height / 2 - 70);

  ctx.font = '18px monospace';
  ctx.fillText('Press 1 for ONE PLAYER (vs. AI Heinrich)', width / 2, height / 2 - 20);
  ctx.fillText('Press 2 for TWO PLAYERS (same keyboard)', width / 2, height / 2 + 10);

  ctx.font = '14px monospace';
  const lines = [
    'P1: A/D move  W/S tilt  F punch  G kick  H block  Q flee',
    'P2: Arrows move/tilt  4 punch  5 kick  6 block  / flee',
    '',
    'Tab toggles debug overlay.',
  ];
  lines.forEach((line, i) => {
    ctx.fillText(line, width / 2, height / 2 + 60 + i * 22);
  });

  ctx.textAlign = 'left';
};
