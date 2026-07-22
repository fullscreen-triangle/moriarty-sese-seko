// =====================================================================
//  index.js — the demo home.
//
//  Left: the 3D character (client-only). Right: purpose card, chat, and
//  the research-mode instrument panel. The page owns the two pieces of
//  shared state — the active animation clip and the committed count —
//  which the chat drives via onCommit / onVerdict.
// =====================================================================

import Head from "next/head";
import dynamic from "next/dynamic";
import { useState } from "react";
import { AnimatePresence } from "framer-motion";

import TransitionEffect from "@/components/TransitionEffect";
import PurposeCard from "@/components/demo/PurposeCard";
import ChatPanel from "@/components/demo/ChatPanel";
import InstrumentPanel from "@/components/demo/InstrumentPanel";
import ResearchToggle from "@/components/demo/ResearchToggle";
import { INITIAL_CLIP } from "@/lib/persona";
import { DEFAULT_P_STAR } from "@/lib/scorer";

// R3F must be client-only — no SSR.
const Scene = dynamic(() => import("@/components/demo/Scene"), {
  ssr: false,
  loading: () => (
    <div className="flex h-full w-full items-center justify-center text-dark/50 dark:text-light/50">
      loading 3D…
    </div>
  ),
});

export default function Home() {
  const [activeClip, setActiveClip] = useState(INITIAL_CLIP);
  const [committedCount, setCommittedCount] = useState(0);
  const [verdict, setVerdict] = useState(null);
  const [research, setResearch] = useState(false);
  const [pStar] = useState(DEFAULT_P_STAR);

  // A persuaded verdict: commit the change (irreversible: count only grows).
  function handleCommit(nextClip, v) {
    setActiveClip(nextClip);
    setCommittedCount((c) => c + 1);
    setVerdict(v);
  }

  // Every turn reports its verdict so the instrument panel stays fresh
  // even when the argument fails.
  function handleVerdict(v) {
    setVerdict(v);
  }

  return (
    <>
      <Head>
        <title>NPC Persuasion Demo — argue with a character that has a purpose</title>
        <meta
          name="description"
          content="A 3D character runs with a purpose. Chat with it and try to make a compelling case to change what it does — judged by a two-factor relevance gate from the NPC papers."
        />
      </Head>
      <TransitionEffect />
      <main className="flex min-h-screen w-full flex-col bg-light px-8 pb-8 pt-4 text-dark dark:bg-dark dark:text-light lg:px-6 md:px-4">
        <div className="grid min-h-[calc(100vh-9rem)] flex-1 grid-cols-2 gap-6 lg:grid-cols-1">
          {/* 3D stage */}
          <section className="relative min-h-[50vh] overflow-hidden rounded-2xl border border-dark/10 dark:border-light/10">
            <Scene activeClip={activeClip} />
            <div className="pointer-events-none absolute left-4 top-4 rounded-full bg-dark/70 px-3 py-1 text-xs font-semibold text-light dark:bg-light/70 dark:text-dark">
              {activeClip}
            </div>
          </section>

          {/* control column */}
          <section className="flex min-h-0 flex-col gap-4">
            <PurposeCard />

            <div className="flex items-center justify-between">
              <span className="text-xs text-dark/50 dark:text-light/50">
                committed changes: <b>{committedCount}</b>
              </span>
              <ResearchToggle on={research} onChange={setResearch} />
            </div>

            <AnimatePresence>
              {research && (
                <InstrumentPanel
                  verdict={verdict}
                  committedCount={committedCount}
                  pStar={pStar}
                />
              )}
            </AnimatePresence>

            <div className="min-h-[320px] flex-1">
              <ChatPanel
                currentAnimation={activeClip}
                pStar={pStar}
                onCommit={handleCommit}
                onVerdict={handleVerdict}
              />
            </div>
          </section>
        </div>
      </main>
    </>
  );
}
