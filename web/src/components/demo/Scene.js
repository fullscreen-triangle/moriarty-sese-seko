// =====================================================================
//  Scene.js — the R3F canvas holding the character.
//
//  Client-only (imported via next/dynamic with ssr:false from the page).
//  The GLB is Mixamo-scale (~160 units tall, cm), so we scale it down and
//  frame it. Theme-aware: background follows the app's dark/light class.
// =====================================================================

import React, { Suspense, useEffect, useState } from "react";
import { Canvas } from "@react-three/fiber";
import {
  OrbitControls,
  Environment,
  ContactShadows,
  Html,
} from "@react-three/drei";
import Character from "./Character";

// Mixamo models are authored in centimetres; scale to a ~1.8 unit avatar.
const MODEL_SCALE = 0.011;

function Loader() {
  return (
    <Html center>
      <div className="text-dark/70 dark:text-light/70 text-sm font-medium animate-pulse">
        loading character…
      </div>
    </Html>
  );
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
    <Canvas
      shadows
      dpr={[1, 2]}
      camera={{ position: [0, 1.1, 4.2], fov: 42 }}
      className="rounded-2xl"
    >
      <color attach="background" args={[dark ? "#1b1b1b" : "#f5f5f5"]} />
      <ambientLight intensity={dark ? 0.5 : 0.8} />
      <directionalLight
        position={[3, 6, 4]}
        intensity={dark ? 1.1 : 1.4}
        castShadow
        shadow-mapSize={[1024, 1024]}
      />
      <Suspense fallback={<Loader />}>
        <group position={[0, -0.9, 0]}>
          <Character activeClip={activeClip} scale={MODEL_SCALE} />
        </group>
        <ContactShadows
          position={[0, -0.9, 0]}
          opacity={dark ? 0.5 : 0.35}
          scale={8}
          blur={2.4}
          far={4}
        />
        <Environment preset="city" />
      </Suspense>
      <OrbitControls
        enablePan={false}
        minDistance={2.5}
        maxDistance={7}
        minPolarAngle={Math.PI / 6}
        maxPolarAngle={Math.PI / 1.9}
        target={[0, 0.1, 0]}
      />
    </Canvas>
  );
}
