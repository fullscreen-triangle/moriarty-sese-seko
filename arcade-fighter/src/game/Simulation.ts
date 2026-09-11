import type { FighterState, FighterId } from '../entities/FighterState';
import { createFighter } from '../entities/Fighter';
import { InputManager } from '../input/InputManager';
import { InputBuffer } from '../input/InputBuffer';
import type { ActionIntent } from '../input/ActionIntent';
import { RNG } from '../physics/RNG';
import { resolveAction, durationForShape } from '../physics/MoveResolver';
import { decayLimb } from '../physics/InjuryModel';
import { regenStamina } from '../physics/StaminaModel';
import { regenComposure, decayDisorientation } from '../physics/ComposureModel';
import { ALL_LIMB_IDS } from '../entities/LimbState';
import { POSES, interpolatePose } from '../rendering/Pose';
import { IDLE_POSE } from '../entities/FighterState';
import { ARENA } from '../physics/Kinematics';
import { MATCH_CONFIG } from './MatchConfig';
import type { MatchMode, MatchPhase } from './GameState';
import { createInitialPhase } from './GameState';
import { CameraShake } from '../rendering/CameraShake';
import { DebugOverlay } from '../ui/DebugOverlay';
import type { CharacterId } from '../entities/Character';
import { KnowledgeGraph } from '../ai/KnowledgeGraph';
import { PolicyResolver } from '../ai/PolicyResolver';
import { OllamaStrategist } from '../ai/OllamaStrategist';
import { ImaginedMode } from './ImaginedMode';
import type { ImaginedWindowTally } from './ImaginedMode';
import type { ActionShape } from '../entities/MoveHistory';
import { preloadImages } from '../rendering/ImageCache';
import { ALL_SPRITE_SHEETS } from '../rendering/SpriteData';

export class Simulation {
  p1: FighterState;
  p2: FighterState;
  phase: MatchPhase = createInitialPhase();
  input = new InputManager();
  buffers: Record<FighterId, InputBuffer> = { p1: new InputBuffer(), p2: new InputBuffer() };
  rng = new RNG(Date.now() >>> 0);
  cameraShake = new CameraShake();
  debugOverlay = new DebugOverlay();
  private scores: Record<FighterId, number> = { p1: 0, p2: 0 };
  private clockMs = 0;
  knowledgeGraph = new KnowledgeGraph();
  private policyResolver = new PolicyResolver(this.knowledgeGraph);
  private ollamaStrategist = new OllamaStrategist(this.knowledgeGraph);
  imaginedMode = new ImaginedMode();

  constructor() {
    this.p1 = createFighter('p1', 'bhuru', { x: -2, y: 0 }, 1);
    this.p2 = createFighter('p2', 'heinrich', { x: 2, y: 0 }, -1);

    preloadImages(ALL_SPRITE_SHEETS, (fraction) => {
      if (this.phase.kind === 'LOADING') {
        this.phase = { kind: 'LOADING', progress: fraction };
      }
    }).then(() => {
      if (this.phase.kind === 'LOADING') {
        // The Dossier overlay (ui/Dossier.ts) presents mode + fighter pick as a
        // single scroll, so there's no separate INTRO beat to wait in — go
        // straight to CHARACTER_SELECT with a default mode the overlay's own
        // mode buttons can still change before a fighter is picked.
        this.startCharacterSelect('1P_VS_AI');
      }
    });

    // Character-select input (mode + fighter pick) is owned entirely by the
    // Dossier overlay now (ui/Dossier.ts calls simulation.lockIn directly, with
    // its own keyboard fallback) — Simulation no longer listens for 1/2 here.
    window.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && (this.phase.kind === 'FIGHTING' || this.phase.kind === 'PAUSED')) {
        this.togglePause();
      }
      if (e.key === 'Tab') {
        e.preventDefault();
        this.debugOverlay.toggle();
      }
      if (e.key === 'i' || e.key === 'I') {
        this.imaginedMode.forceTrigger();
      }
    });
  }

  private togglePause(): void {
    if (this.phase.kind === 'FIGHTING') {
      this.phase = { kind: 'PAUSED', previous: this.phase };
    } else if (this.phase.kind === 'PAUSED') {
      this.phase = this.phase.previous;
    }
  }

  private startCharacterSelect(mode: MatchMode): void {
    this.phase = { kind: 'CHARACTER_SELECT', mode, picks: {} };
  }

  /**
   * Single entry point for finalizing a character pick — called by the Dossier
   * overlay's click-driven and keyboard-fallback flows alike (see ui/Dossier.ts,
   * wired in main.ts) so both input paths produce identical fighter/AI/
   * knowledge-graph setup.
   */
  lockIn(mode: MatchMode, p1: CharacterId, p2: CharacterId): void {
    this.p1 = createFighter('p1', p1, { x: -2, y: 0 }, 1);
    this.p2 = createFighter('p2', p2, { x: 2, y: 0 }, -1);
    this.aiEnabled = mode === '1P_VS_AI';
    this.knowledgeGraph = new KnowledgeGraph();
    this.policyResolver = new PolicyResolver(this.knowledgeGraph);
    this.ollamaStrategist = new OllamaStrategist(this.knowledgeGraph);
    this.phase = { kind: 'FIGHTING', remainingMs: MATCH_CONFIG.roundDurationMs };
  }

  /**
   * Milestone 2: fast per-tick KnowledgeGraph-driven policy for Heinrich, plus a
   * slow independent OllamaStrategist propagation cycle. The policy replaces
   * InputManager as p2's ActionIntent producer; Ollama runs off the hot path and
   * only ever mutates the graph, never blocks a tick.
   */
  private updateAi(dtMs: number): void {
    const fighter = this.p2;
    const opponent = this.p1;

    this.policyResolver.observe(fighter, opponent, this.clockMs);
    this.policyResolver.refillAttention(dtMs);
    this.ollamaStrategist.tick(dtMs);

    const distance = opponent.position.x - fighter.position.x;
    const desiredRange = 1.3;
    const closeEnough = Math.abs(distance) < desiredRange + 0.3;
    fighter.position.x += Math.sign(distance) * (closeEnough ? 0 : 1) * 1.6 * (dtMs / 1000);
    fighter.position.x = Math.max(-ARENA.width / 2, Math.min(ARENA.width / 2, fighter.position.x));
    fighter.facing = fighter.position.x <= opponent.position.x ? 1 : -1;

    const intent = this.policyResolver.resolve(fighter, opponent, this.rng);
    if (!intent) return;

    if (intent.type === 'block') {
      fighter.blocking = true;
      return;
    }
    fighter.blocking = false;
    this.buffers.p2.push(intent, this.clockMs, fighter.disorientation, this.rng);
  }

  private fighters(): FighterState[] {
    return [this.p1, this.p2];
  }

  private opponentOf(id: FighterId): FighterState {
    return id === 'p1' ? this.p2 : this.p1;
  }

  private aiEnabled = false;

  tick(dtMs: number): void {
    if (this.phase.kind === 'PAUSED') return;
    this.clockMs += dtMs;
    const intents = this.input.poll(this.clockMs);

    for (const fighter of this.fighters()) {
      if (fighter.id === 'p2' && this.aiEnabled) continue;
      this.updateMovement(fighter, dtMs);
      fighter.blocking = this.input.isBlocking(fighter.id);
    }
    if (this.aiEnabled) this.updateAi(dtMs);

    for (const intent of intents) {
      const fighter = intent.playerId === 'p1' ? this.p1 : this.p2;
      if (fighter.id === 'p2' && this.aiEnabled) continue;
      this.buffers[fighter.id].push(intent, this.clockMs, fighter.disorientation, this.rng);
    }

    if (this.phase.kind === 'FIGHTING') {
      for (const fighter of this.fighters()) {
        const ready = this.buffers[fighter.id].drain(this.clockMs);
        for (const readyIntent of ready) {
          this.tryStartAction(fighter, readyIntent);
        }
        if (
          this.input.isFleeing(fighter.id) &&
          fighter.actionState.kind !== 'fleeing' &&
          fighter.actionState.kind !== 'stumble'
        ) {
          fighter.actionState = { kind: 'fleeing', elapsedMs: 0, totalMs: MATCH_CONFIG.fleeWindupMs };
        }
      }

      for (const fighter of this.fighters()) {
        this.advanceActionState(fighter, dtMs);
      }

      for (const fighter of this.fighters()) {
        for (const limbId of ALL_LIMB_IDS) decayLimb(fighter.limbs[limbId], dtMs);
        regenStamina(fighter, dtMs);
        regenComposure(fighter, dtMs);
        decayDisorientation(fighter, dtMs);
      }

      this.checkStumble();
      this.checkFled();
      this.checkTimer(dtMs);

      const closedTally = this.imaginedMode.tick(dtMs, this.rng);
      if (closedTally) this.applyImaginedResidue(closedTally);
    }

    this.cameraShake.update(dtMs);
    for (const fighter of this.fighters()) {
      this.updatePose(fighter);
    }
  }

  private updateMovement(fighter: FighterState, dtMs: number): void {
    if (fighter.actionState.kind === 'stumble' || fighter.actionState.kind === 'fleeing') {
      fighter.velocity.x = 0;
      return;
    }
    const axis = this.input.movementAxis(fighter.id);
    const speed = 3;
    fighter.velocity.x = axis * speed;
    fighter.position.x += axis * speed * (dtMs / 1000);
    fighter.position.x = Math.max(-ARENA.width / 2, Math.min(ARENA.width / 2, fighter.position.x));

    const other = this.opponentOf(fighter.id);
    fighter.facing = fighter.position.x <= other.position.x ? 1 : -1;
  }

  private tryStartAction(fighter: FighterState, intent: ActionIntent): void {
    if (fighter.actionState.kind !== 'idle' && fighter.actionState.kind !== 'blocking') return;
    if (intent.type !== 'punch' && intent.type !== 'kick') return;

    const opponent = this.opponentOf(fighter.id);
    const imagined = this.imaginedMode.isActive();
    const result = resolveAction(fighter, opponent, intent, this.rng, imagined);
    if (!result) return;

    this.debugOverlay.recordResolve(fighter.id, result);
    this.imaginedMode.noteAttempt(fighter.id, result.shape);

    if (result.composureDrain > 0) {
      this.cameraShake.trigger(result.composureDrain);
      this.scores[fighter.id] += result.composureDrain;
      this.imaginedMode.noteComposureDrain(result.composureDrain);
    }

    const durations = durationForShape(result.shape, fighter.moveHistory[result.shape].proficiency);
    fighter.actionState = {
      kind: 'startup',
      shape: result.shape,
      limb: result.limb,
      elapsedMs: 0,
      totalMs: durations.startup,
    };
    (fighter as unknown as { _pendingDurations: typeof durations })._pendingDurations = durations;
  }

  private advanceActionState(fighter: FighterState, dtMs: number): void {
    const state = fighter.actionState;

    if (state.kind === 'startup' || state.kind === 'active' || state.kind === 'recovery') {
      const elapsed = state.elapsedMs + dtMs;
      if (elapsed < state.totalMs) {
        fighter.actionState = { ...state, elapsedMs: elapsed };
        return;
      }
      const durations = (fighter as unknown as { _pendingDurations: { startup: number; active: number; recovery: number } })
        ._pendingDurations;
      if (state.kind === 'startup') {
        fighter.actionState = { kind: 'active', shape: state.shape, limb: state.limb, elapsedMs: 0, totalMs: durations.active };
      } else if (state.kind === 'active') {
        fighter.actionState = { kind: 'recovery', shape: state.shape, limb: state.limb, elapsedMs: 0, totalMs: durations.recovery };
      } else {
        fighter.actionState = { kind: 'idle' };
      }
      return;
    }

    if (state.kind === 'stumble') {
      const elapsed = state.elapsedMs + dtMs;
      fighter.actionState = elapsed < state.totalMs ? { ...state, elapsedMs: elapsed } : { kind: 'idle' };
      return;
    }

    if (state.kind === 'fleeing') {
      const elapsed = state.elapsedMs + dtMs;
      fighter.actionState = { ...state, elapsedMs: elapsed };
      return;
    }

    if (state.kind === 'blocking' && !fighter.blocking) {
      fighter.actionState = { kind: 'idle' };
    } else if (state.kind === 'idle' && fighter.blocking) {
      fighter.actionState = { kind: 'blocking' };
    }
  }

  private checkStumble(): void {
    for (const fighter of this.fighters()) {
      if (fighter.composure <= 0 && fighter.actionState.kind !== 'stumble') {
        fighter.actionState = { kind: 'stumble', elapsedMs: 0, totalMs: 1500 };
        const winner = this.opponentOf(fighter.id).id;
        this.phase = { kind: 'ROUND_END', reason: 'stumble', winner };
      }
    }
  }

  private checkFled(): void {
    for (const fighter of this.fighters()) {
      if (fighter.actionState.kind === 'fleeing' && fighter.actionState.elapsedMs >= fighter.actionState.totalMs) {
        fighter.hasFled = true;
        this.phase = { kind: 'FLED', fledPlayer: fighter.id };
      }
    }
  }

  private checkTimer(dtMs: number): void {
    if (this.phase.kind !== 'FIGHTING') return;
    const remainingMs = this.phase.remainingMs - dtMs;
    if (remainingMs <= 0) {
      const winner: FighterId | 'draw' =
        this.scores.p1 === this.scores.p2 ? 'draw' : this.scores.p1 > this.scores.p2 ? 'p1' : 'p2';
      this.phase = { kind: 'ROUND_END', reason: 'timer', winner };
    } else {
      this.phase = { kind: 'FIGHTING', remainingMs };
    }
  }

  /**
   * Imagined Mode leaves no damage behind, but Heinrich — the analyst literally
   * building a case file on Bhuru — still remembers what he saw. Feed the closed
   * window's attempted shapes into the KnowledgeGraph as permanent residue, the
   * same "favors:<shape>" signal PolicyResolver already reads at rest.
   */
  private applyImaginedResidue(tally: Record<FighterId, ImaginedWindowTally>): void {
    const entries = Object.entries(tally.p1.shapesAttempted) as Array<[ActionShape, number]>;
    for (const [shape, count] of entries) {
      this.knowledgeGraph.observe(`favors:${shape}`, 0.03 * count, this.clockMs);
    }
  }

  private updatePose(fighter: FighterState): void {
    const state = fighter.actionState;

    if (state.kind === 'idle') {
      fighter.jointAngles = this.applyDebuffOverlay(fighter, { ...IDLE_POSE });
      return;
    }
    if (state.kind === 'blocking') {
      fighter.jointAngles = this.applyDebuffOverlay(fighter, { ...POSES.block });
      return;
    }
    if (state.kind === 'stumble') {
      fighter.jointAngles = { ...POSES.stumble };
      return;
    }
    if (state.kind === 'fleeing') {
      fighter.jointAngles = this.applyDebuffOverlay(fighter, { ...POSES.fleeStance });
      return;
    }

    const phaseKey = state.kind === 'startup' ? 'Startup' : state.kind === 'active' ? 'Active' : 'Recovery';
    const key = `${state.shape}${phaseKey}` as keyof typeof POSES;
    const pose = POSES[key] ?? POSES.idle;
    const t = Math.min(1, state.elapsedMs / state.totalMs);
    const fromKey = `${state.shape}${phaseKey === 'Active' ? 'Startup' : 'Active'}` as keyof typeof POSES;
    const from = state.kind === 'startup' ? IDLE_POSE : POSES[fromKey] ?? IDLE_POSE;
    fighter.jointAngles = this.applyDebuffOverlay(fighter, interpolatePose(from, pose, t));
  }

  private applyDebuffOverlay(
    fighter: FighterState,
    pose: Record<string, number>
  ): Record<string, number> {
    const leftArm = fighter.limbs.leftArm;
    const rightArm = fighter.limbs.rightArm;
    if (leftArm.injurySeverity > 0.2) {
      pose.leftShoulder += 0.3 * leftArm.injurySeverity;
    }
    if (rightArm.injurySeverity > 0.2) {
      pose.rightShoulder -= 0.3 * rightArm.injurySeverity;
    }
    if (fighter.disorientation > 0) {
      const wobble = Math.sin(this.clockMs / 120) * 0.05 * fighter.disorientation;
      pose.neck += wobble;
    }
    return pose;
  }
}
