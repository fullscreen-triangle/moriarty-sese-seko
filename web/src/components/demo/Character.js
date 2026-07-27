// =====================================================================
//  Character.js — the GLB avatar and its animation mixer.
//
//  Loads /xbot_multiple_animations.glb and cross-fades between clips
//  whenever `activeClip` changes. The GLB carries a base "mixamo.com"
//  track that we never play. Clips: Idle, Jump, Running, Walking.
//
//  Two Mixamo gotchas we handle:
//   1. Export scale/offset — the model is in centimetres and not centred on
//      the origin, so we measure its bounding box and apply a corrective
//      transform on an OUTER wrapper group (never on the animated root, whose
//      own transform the mixer may drive).
//   2. Skinned-mesh cloning — a plain Object3D.clone() leaves cloned meshes
//      bound to the ORIGINAL skeleton, so animations play but nothing
//      deforms. We clone with SkeletonUtils, which rebinds the skeleton, and
//      run the mixer on that same clone.
// =====================================================================

import React, { useEffect, useMemo, useRef } from "react";
import { useGLTF, useAnimations } from "@react-three/drei";
import { clone as skeletonClone } from "three/examples/jsm/utils/SkeletonUtils.js";
import * as THREE from "three";

const MODEL_URL = "/xbot_multiple_animations.glb";
const IGNORED = new Set(["mixamo.com"]);
const FADE = 0.35;
const TARGET_HEIGHT = 1.7; // metres — a human-scale avatar in the scene.

export default function Character({ activeClip = "Running", ...props }) {
  const { scene, animations } = useGLTF(MODEL_URL);
  const prevClip = useRef(null);

  // Skeleton-aware clone: cloned SkinnedMeshes get their own rebound skeleton,
  // so the mixer actually deforms THIS instance (not the shared original).
  const model = useMemo(() => skeletonClone(scene), [scene]);

  // Corrective transform for the wrapper group. Mixamo GLBs are authored in
  // centimetres, so ~0.01 yields a ~1.7 m avatar. We measure the STATIC
  // (rest-pose) bounds to derive the exact factor, but skinned-mesh bounds
  // can read as zero before the first render — so we fall back to the known
  // centimetre scale and only trust a measured value that is sane.
  const fit = useMemo(() => {
    const CM_SCALE = 0.01; // fallback: 1 unit = 1 cm
    const box = new THREE.Box3().setFromObject(model);
    const size = new THREE.Vector3();
    const center = new THREE.Vector3();
    box.getSize(size);
    box.getCenter(center);

    // Trust the measurement only if it looks like a real, tall model
    // (Mixamo rest height ~150–200 cm). Otherwise use the cm fallback.
    const measured = size.y > 50 ? TARGET_HEIGHT / size.y : CM_SCALE;
    const scale = measured;

    // Recentre using measured center when valid; else assume origin-ish.
    const cx = Number.isFinite(center.x) ? center.x : 0;
    const cz = Number.isFinite(center.z) ? center.z : 0;
    const minY = Number.isFinite(box.min.y) ? box.min.y : 0;
    return {
      scale,
      position: [-cx * scale, -minY * scale, -cz * scale],
    };
  }, [model]);

  useEffect(() => {
    model.traverse((o) => {
      if (o.isMesh) {
        o.castShadow = true;
        o.receiveShadow = true;
        o.frustumCulled = false; // animated skinned meshes can be mis-culled
      }
    });
  }, [model]);

  // Run the mixer on the clone itself (not an outer ref), so actions bind to
  // the same skeleton we render.
  const { actions, names } = useAnimations(animations, model);

  useEffect(() => {
    if (!actions) return;

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
  }, [actions, names, activeClip]);

  return (
    <group {...props} position={fit.position} scale={fit.scale} dispose={null}>
      <primitive object={model} />
    </group>
  );
}

useGLTF.preload(MODEL_URL);
