export const drawPauseMenu = (ctx: CanvasRenderingContext2D, width: number, height: number): void => {
  ctx.fillStyle = 'rgba(10, 11, 16, 0.78)';
  ctx.fillRect(0, 0, width, height);

  ctx.textAlign = 'center';
  ctx.fillStyle = '#e8eaf2';
  ctx.font = 'bold 36px monospace';
  ctx.fillText('PAUSED', width / 2, height / 2 - 140);

  ctx.font = 'bold 16px monospace';
  ctx.fillStyle = '#8bd17c';
  ctx.fillText('Press Esc to resume', width / 2, height / 2 - 100);

  const columns: { title: string; color: string; lines: string[] }[] = [
    {
      title: 'PLAYER 1',
      color: '#5ec2ff',
      lines: [
        'Move       A / D',
        'Tilt       W / S',
        'Punch      F',
        'Kick       G',
        'Block      H (hold)',
        'Flee       Q',
      ],
    },
    {
      title: 'PLAYER 2',
      color: '#ff6b6b',
      lines: [
        'Move       ← / →',
        'Tilt       ↑ / ↓',
        'Punch      4',
        'Kick       5',
        'Block      6 (hold)',
        'Flee       /',
      ],
    },
  ];

  const colW = 260;
  const gap = 60;
  const totalW = colW * columns.length + gap * (columns.length - 1);
  const startX = width / 2 - totalW / 2;
  const topY = height / 2 - 60;

  columns.forEach((col, i) => {
    const x = startX + i * (colW + gap) + colW / 2;
    ctx.font = 'bold 16px monospace';
    ctx.fillStyle = col.color;
    ctx.fillText(col.title, x, topY);

    ctx.font = '14px monospace';
    ctx.fillStyle = '#c8ccdc';
    col.lines.forEach((line, li) => {
      ctx.fillText(line, x, topY + 30 + li * 22);
    });
  });

  ctx.font = '13px monospace';
  ctx.fillStyle = '#8890a8';
  ctx.fillText('Tab: debug overlay   •   I: force Imagined Mode', width / 2, topY + 30 + columns[0].lines.length * 22 + 40);

  ctx.textAlign = 'left';
};
