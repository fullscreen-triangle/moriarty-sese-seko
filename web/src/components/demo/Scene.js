// =====================================================================
//  Scene.js — the R3F canvas holding the character.
//
//  Client-only (imported via next/dynamic with ssr:false from the page).
//  The GLB is Mixamo-scale (~160 units tall, cm), so we scale it down and
//  frame it. Theme-aware: background follows the app's dark/light class.
// =====================================================================

import React, { Suspense, useEffect, useState } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls, ContactShadows, Html } from "@react-three/drei";
import Character from "./Character";

function Loader() {
  return (
    <Html center>
      <div className="text-dark/70 dark:text-light/70 text-sm font-medium animate-pulse">
        loading character…
      </div>
    </Html>
  );
}

// If anything in the 3D subtree throws (bad GLB, WebGL init failure, a drei
// API mismatch), show the reason instead of a silently blank canvas.
class GLErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { error: null };
  }
  static getDerivedStateFromError(error) {
    return { error };
  }
  componentDidCatch(error, info) {
    // Surface it in the console too, so the exact stack is visible.
    // eslint-disable-next-line no-console
    console.error("3D scene failed:", error, info);
  }
  render() {
    if (this.state.error) {
      return (
        <div className="flex h-full w-full items-center justify-center p-6 text-center text-sm text-red-500">
          3D scene failed to load: {String(this.state.error.message || this.state.error)}
        </div>
      );
    }
    return this.props.children;
  }
}

function useIsDark() {
  const [dark, setDark] = useState(false);
  useEffect(() => {
    const root = document.documentElement;
    const update = () => setDark(root.classList.contains("dark"));
    update();
    const obs = new MutationObserver(update);
    obs.observe(root, { attributes: true, attributeFilter: ["class"] });
    return () => obs.disconnect();
  }, []);
  return dark;
}

export default function Scene({ activeClip }) {
  const dark = useIsDark();

  return (
    <GLErrorBoundary>
      {/* Absolute fill guarantees the Canvas has real pixel dimensions.
          R3F sizes to its parent, and a bare Canvas in an auto-height grid
          cell can collapse to 0×0 (invisible). The parent <section> is
          position:relative, so this fills it. */}
      <div className="absolute inset-0">
        <Canvas
          shadows
          dpr={[1, 2]}
          camera={{ position: [0, 1.4, 8.4], fov: 42 }}
          className="rounded-2xl"
        >
          <color attach="background" args={[dark ? "#1b1b1b" : "#f5f5f5"]} />
      {/* Self-contained lighting — no CDN environment map, so nothing
          external has to load before the character can render. */}
      <hemisphereLight
        intensity={dark ? 0.6 : 0.9}
        color={dark ? "#9fb2ff" : "#ffffff"}
        groundColor={dark ? "#1b1b1b" : "#d8d8d8"}
      />
      <ambientLight intensity={dark ? 0.4 : 0.6} />
      <directionalLight
        position={[3, 6, 4]}
        intensity={dark ? 1.3 : 1.6}
        castShadow
        shadow-mapSize={[1024, 1024]}
      />
      <directionalLight position={[-4, 3, -3]} intensity={dark ? 0.5 : 0.7} />
      <Suspense fallback={<Loader />}>
        {/* Character normalises itself to ~1.7m with feet at y=0, so it
            sits directly on the ground plane — no manual scale/offset. */}
        <Character activeClip={activeClip} />
        <ContactShadows
          position={[0, 0, 0]}
          opacity={dark ? 0.5 : 0.35}
          scale={8}
          blur={2.4}
          far={4}
        />
      </Suspense>
      <OrbitControls
        enablePan={false}
        minDistance={4}
        maxDistance={12}
        minPolarAngle={Math.PI / 6}
        maxPolarAngle={Math.PI / 1.9}
        target={[0, 0.9, 0]}
          />
        </Canvas>
      </div>
    </GLErrorBoundary>
  );
}
