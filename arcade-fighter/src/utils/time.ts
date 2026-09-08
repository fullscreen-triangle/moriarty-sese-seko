export const msToSec = (ms: number): number => ms / 1000;

export const easeOutQuad = (t: number): number => 1 - (1 - t) * (1 - t);

export const lerpAngle = (a: number, b: number, t: number): number => a + (b - a) * t;
