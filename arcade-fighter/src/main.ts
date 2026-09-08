import { Renderer } from './rendering/Renderer';
import { Simulation } from './game/Simulation';
import { startGameLoop } from './game/GameLoop';

const canvas = document.getElementById('game-canvas') as HTMLCanvasElement | null;
if (!canvas) throw new Error('Canvas element #game-canvas not found');

const renderer = new Renderer(canvas);
const simulation = new Simulation();

startGameLoop(canvas, renderer, simulation);
