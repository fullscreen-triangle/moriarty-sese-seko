// =====================================================================
//  InstrumentPanel.js — the "research mode" readout.
//
//  Purely presentational. Visualises the two-factor gate of "The
//  Physiology of Response": the purpose-gain meter (with the p* price
//  line), the coherence meter (with the neutral line), the committed
//  count (irreversibility), and the latest verdict's "why".
// =====================================================================

import React from "react";
import { motion } from "framer-motion";

function Meter({ label, value, threshold, thresholdLabel, cleared }) {
  const pct = Math.round(value * 100);
  const tPct = Math.round(threshold * 100);
  return (
    <div>
      <div className="flex items-baseline justify-between text-xs font-semibold">
        <span className="text-dark/70 dark:text-light/70">{label}</span>
        <span
          className={
            cleared
              ? "text-primary dark:text-primaryDark"
              : "text-dark/50 dark:text-light/50"
          }
        >
          {value.toFixed(2)}
        </span>
      </div>
      <div className="relative mt-1 h-2.5 w-full rounded-full bg-dark/10 dark:bg-light/10">
        <motion.div
          className={`h-2.5 rounded-full ${
            cleared ? "bg-primary dark:bg-primaryDark" : "bg-dark/40 dark:bg-light/40"
          }`}
          animate={{ width: `${pct}%` }}
          transition={{ type: "spring", stiffness: 120, damping: 18 }}
        />
        {/* threshold line */}
        <div
          className="absolute top-[-3px] h-[16px] w-[2px] bg-red-500"
          style={{ left: `${tPct}%` }}
          title={`${thresholdLabel} = ${threshold.toFixed(2)}`}
        />
      </div>
      <div className="mt-1 text-[10px] text-dark/50 dark:text-light/50">
        {thresholdLabel} = {threshold.toFixed(2)}
      </div>
    </div>
  );
}

export default function InstrumentPanel({ verdict, committedCount, pStar }) {
  const gain = verdict?.purposeGain ?? 0;
  const coh = verdict?.coherence ?? 0;
  const clearsPurpose = verdict?.clearsPurpose ?? false;
  const preservesCoherence = verdict?.preservesCoherence ?? false;

  return (
    <motion.div
      initial={{ opacity: 0, height: 0 }}
      animate={{ opacity: 1, height: "auto" }}
      exit={{ opacity: 0, height: 0 }}
      className="overflow-hidden rounded-2xl border border-dark/10 dark:border-light/10 bg-light dark:bg-dark p-5"
    >
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-bold text-dark dark:text-light">
          Two-factor relevance
        </h3>
        <span className="text-[10px] uppercase tracking-wide text-dark/50 dark:text-light/50">
          Physiology of Response
        </span>
      </div>

      <div className="mt-4 space-y-4">
        <Meter
          label="purpose gain"
          value={gain}
          threshold={pStar}
          thresholdLabel="price p*"
          cleared={clearsPurpose}
        />
        <Meter
          label="coherence"
          value={coh}
          threshold={0.5}
          thresholdLabel="neutral"
          cleared={preservesCoherence}
        />
      </div>

      <div className="mt-4 grid grid-cols-2 gap-3 border-t border-dark/10 dark:border-light/10 pt-3">
        <div>
          <div className="text-[10px] uppercase tracking-wide text-dark/50 dark:text-light/50">
            committed count
          </div>
          <div className="text-2xl font-bold text-dark dark:text-light">
            {committedCount}
          </div>
          <div className="text-[10px] text-dark/50 dark:text-light/50">
            irreversible — a change never un-happens
          </div>
        </div>
        <div>
          <div className="text-[10px] uppercase tracking-wide text-dark/50 dark:text-light/50">
            last verdict
          </div>
          <div
            className={`text-sm font-bold ${
              verdict?.persuaded
                ? "text-primary dark:text-primaryDark"
                : "text-dark/60 dark:text-light/60"
            }`}
          >
            {verdict ? (verdict.persuaded ? "PERSUADED" : "NOT PERSUADED") : "—"}
          </div>
        </div>
      </div>

      {verdict?.reason && (
        <p className="mt-3 rounded-lg bg-dark/5 dark:bg-light/5 px-3 py-2 text-[11px] leading-relaxed text-dark/70 dark:text-light/70">
          {verdict.reason}
        </p>
      )}
    </motion.div>
  );
}
