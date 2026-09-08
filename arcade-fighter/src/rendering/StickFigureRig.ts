import type { FighterState } from '../entities/FighterState';
import { SEGMENT_LENGTHS } from '../physics/Kinematics';

// Canvas pixels per simulation meter.
const PPM = 90;

type Point = { x: number; y: number };

const project = (origin: Point, angle: number, length: number, facing: number): Point => ({
  x: origin.x + Math.cos(angle) * length * facing,
  y: origin.y + Math.sin(angle) * length,
});

export const drawFighter = (
  ctx: CanvasRenderingContext2D,
  state: FighterState,
  floorPixelY: number,
  color: string
): void => {
  const facing = state.facing;
  const rootX = state.position.x * PPM;
  const rootY = floorPixelY - state.position.y * PPM;

  const pelvis: Point = { x: rootX, y: rootY - SEGMENT_LENGTHS.thigh * PPM * 1.6 };
  const neckBase: Point = { x: pelvis.x, y: pelvis.y - SEGMENT_LENGTHS.torso * PPM };
  const angles = state.jointAngles;

  const head = project(neckBase, angles.neck, SEGMENT_LENGTHS.headRadius * 1.4 * PPM, facing);

  const lShoulder = neckBase;
  const lElbow = project(lShoulder, angles.leftShoulder - Math.PI / 2, SEGMENT_LENGTHS.upperArm * PPM, facing);
  const lHand = project(lElbow, angles.leftShoulder + angles.leftElbow - Math.PI / 2, SEGMENT_LENGTHS.forearm * PPM, facing);

  const rShoulder = neckBase;
  const rElbow = project(rShoulder, angles.rightShoulder - Math.PI / 2, SEGMENT_LENGTHS.upperArm * PPM, facing);
  const rHand = project(rElbow, angles.rightShoulder + angles.rightElbow - Math.PI / 2, SEGMENT_LENGTHS.forearm * PPM, facing);

  const lHip = pelvis;
  const lKnee = project(lHip, angles.leftHip + Math.PI / 2, SEGMENT_LENGTHS.thigh * PPM, facing);
  const lAnkle = project(lKnee, angles.leftHip + angles.leftKnee + Math.PI / 2, SEGMENT_LENGTHS.shank * PPM, facing);
  const lFoot = project(lAnkle, Math.PI / 2 + angles.leftAnkle, SEGMENT_LENGTHS.foot * PPM, facing);

  const rHip = pelvis;
  const rKnee = project(rHip, angles.rightHip + Math.PI / 2, SEGMENT_LENGTHS.thigh * PPM, facing);
  const rAnkle = project(rKnee, angles.rightHip + angles.rightKnee + Math.PI / 2, SEGMENT_LENGTHS.shank * PPM, facing);
  const rFoot = project(rAnkle, Math.PI / 2 + angles.rightAnkle, SEGMENT_LENGTHS.foot * PPM, facing);

  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = 4;
  ctx.lineCap = 'round';

  const segment = (a: Point, b: Point) => {
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
  };

  segment(pelvis, neckBase);
  segment(lShoulder, lElbow);
  segment(lElbow, lHand);
  segment(rShoulder, rElbow);
  segment(rElbow, rHand);
  segment(lHip, lKnee);
  segment(lKnee, lAnkle);
  segment(lAnkle, lFoot);
  segment(rHip, rKnee);
  segment(rKnee, rAnkle);
  segment(rAnkle, rFoot);

  ctx.beginPath();
  ctx.arc(head.x, head.y, SEGMENT_LENGTHS.headRadius * PPM, 0, Math.PI * 2);
  ctx.fill();
};

export const FLOOR_PIXEL_MARGIN = 80;
export const PIXELS_PER_METER = PPM;
