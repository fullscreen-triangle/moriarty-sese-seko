export class CameraShake {
  private magnitude = 0;
  private decayPerMs = 0.02;

  trigger(amount: number): void {
    this.magnitude = Math.min(20, this.magnitude + amount * 40);
  }

  update(dtMs: number): void {
    this.magnitude = Math.max(0, this.magnitude - this.decayPerMs * dtMs);
  }

  offset(): { x: number; y: number } {
    if (this.magnitude <= 0) return { x: 0, y: 0 };
    const angle = Math.random() * Math.PI * 2;
    const r = Math.random() * this.magnitude;
    return { x: Math.cos(angle) * r, y: Math.sin(angle) * r };
  }
}
