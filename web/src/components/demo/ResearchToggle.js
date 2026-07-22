// =====================================================================
//  ResearchToggle.js — switch between chat-first and research mode.
// =====================================================================

import React from "react";

export default function ResearchToggle({ on, onChange }) {
  return (
    <button
      type="button"
      onClick={() => onChange(!on)}
      className="flex items-center gap-2 text-xs font-semibold text-dark/70 dark:text-light/70"
      aria-pressed={on}
    >
      <span
        className={`relative h-5 w-9 rounded-full transition-colors ${
          on ? "bg-primary dark:bg-primaryDark" : "bg-dark/20 dark:bg-light/20"
        }`}
      >
        <span
          className={`absolute top-0.5 h-4 w-4 rounded-full bg-light dark:bg-dark transition-transform ${
            on ? "translate-x-4" : "translate-x-0.5"
          }`}
        />
      </span>
      Research mode
    </button>
  );
}
