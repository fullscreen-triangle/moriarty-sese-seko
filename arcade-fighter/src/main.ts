import { Renderer } from './rendering/Renderer';
import { Simulation } from './game/Simulation';
import { startGameLoop } from './game/GameLoop';
import { mountDossier } from './ui/Dossier';

const canvas = document.getElementById('game-canvas') as HTMLCanvasElement | null;
if (!canvas) throw new Error('Canvas element #game-canvas not found');

const renderer = new Renderer(canvas);
const simulation = new Simulation();
const dossier = mountDossier((mode, p1, p2) => simulation.lockIn(mode, p1, p2));

startGameLoop(canvas, renderer, simulation, dossier);
