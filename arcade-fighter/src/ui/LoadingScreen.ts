let animT = 0;

export const drawLoadingScreen = (
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  progress: number,
  dtMs = 16
): void => {
  animT += dtMs;

  ctx.fillStyle = '#0c0d14';
  ctx.fillRect(0, 0, width, height);

  // Faint radial glow, same family as the imagined-mode wash, to keep the
  // loading screen visually related to the rest of the game rather than a
  // generic splash.
  const glow = ctx.createRadialGradient(
    width / 2, height * 0.42, height * 0.05,
    width / 2, height * 0.42, height * 0.6
  );
  glow.addColorStop(0, 'rgba(94, 194, 255, 0.10)');
  glow.addColorStop(1, 'rgba(12, 13, 20, 0)');
  ctx.fillStyle = glow;
  ctx.fillRect(0, 0, width, height);

  const pulse = 0.5 + 0.5 * Math.sin(animT / 420);

  ctx.textAlign = 'center';
  ctx.fillStyle = '#e8eaf2';
  ctx.font = 'bold 44px monospace';
  ctx.fillText('BHURU-SUKURIN', width / 2, height * 0.36);
  ctx.font = 'bold 22px monospace';
  ctx.fillStyle = '#ffd166';
  ctx.fillText('CHAPTER ONE', width / 2, height * 0.36 + 34);

  // Three-dot spinner orbiting a center point, monospace/arcade-flavored
  // rather than a spinner asset.
  const spinnerY = height * 0.52;
  const spinnerR = 14;
  const dotCount = 3;
  for (let i = 0; i < dotCount; i += 1) {
    const angle = animT / 260 + (i * (Math.PI * 2)) / dotCount;
    const dx = width / 2 + Math.cos(angle) * spinnerR * 2.2;
    const dy = spinnerY + Math.sin(angle) * spinnerR * 0.6;
    const dotPulse = 0.4 + 0.6 * (0.5 + 0.5 * Math.sin(angle));
    ctx.beginPath();
    ctx.fillStyle = `rgba(94, 194, 255, ${dotPulse.toFixed(2)})`;
    ctx.arc(dx, dy, 6, 0, Math.PI * 2);
    ctx.fill();
  }

  // Progress bar.
  const barW = Math.min(width * 0.5, 420);
  const barH = 10;
  const barX = width / 2 - barW / 2;
  const barY = height * 0.62;
  const clamped = Math.max(0, Math.min(1, progress));

  ctx.strokeStyle = '#3a3f50';
  ctx.lineWidth = 2;
  ctx.strokeRect(barX, barY, barW, barH);

  ctx.fillStyle = '#5ec2ff';
  ctx.fillRect(barX, barY, barW * clamped, barH);

  // Leading-edge highlight so the fill reads as "in motion" even while progress
  // sits still between asset settlements.
  if (clamped > 0 && clamped < 1) {
    const edgeX = barX + barW * clamped;
    ctx.fillStyle = `rgba(255, 255, 255, ${(0.3 + 0.3 * pulse).toFixed(2)})`;
    ctx.fillRect(Math.max(barX, edgeX - 3), barY, 3, barH);
  }

  ctx.font = 'bold 16px monospace';
  ctx.fillStyle = '#c8ccdc';
  ctx.fillText(`LOADING… ${Math.round(clamped * 100)}%`, width / 2, barY + barH + 30);

  ctx.font = '12px monospace';
  ctx.fillStyle = '#8890a8';
  ctx.fillText('No movesets. Only what you\'ve practiced.', width / 2, height * 0.92);

  ctx.textAlign = 'left';
};
