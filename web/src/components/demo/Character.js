// =====================================================================
//  Character.js — the GLB avatar and its animation mixer.
//
//  Loads /xbot_multiple_animations.glb and cross-fades between clips
//  whenever `activeClip` changes. The GLB carries a base "mixamo.com"
//  track that we never play. Clips: Idle, Jump, Running, Walking.
// =====================================================================

import React, { useEffect, useRef } from "react";
import { useGLTF, useAnimations } from "@react-three/drei";

const MODEL_URL = "/xbot_multiple_animations.glb";
const IGNORED = new Set(["mixamo.com"]);
const FADE = 0.35;

export default function Character({ activeClip = "Running", ...props }) {
  const group = useRef();
  const { scene, animations } = useGLTF(MODEL_URL);
  const { actions, names } = useAnimations(animations, group);
  const prevClip = useRef(null);

  useEffect(() => {
    if (!actions) return;

    // Resolve the requested clip against the real action names,
    // case-insensitively, skipping the ignored base track.
    const playable = names.filter((n) => !IGNORED.has(n));
    const target =
      playable.find((n) => n.toLowerCase() === String(activeClip).toLowerCase()) ||
      playable[0];

    const next = actions[target];
    if (!next) return;

    const prev = prevClip.current ? actions[prevClip.current] : null;
    if (prev && prev !== next) prev.fadeOut(FADE);

    next.reset().fadeIn(FADE).play();
    prevClip.current = target;

    return () => {
      // Do not stop on cleanup — let the next effect cross-fade.
    };
  }, [actions, names, activeClip]);

  return (
    <group ref={group} {...props} dispose={null}>
      <primitive object={scene} />
    </group>
  );
}

useGLTF.preload(MODEL_URL);
