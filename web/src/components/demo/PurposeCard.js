// =====================================================================
//  PurposeCard.js — shows the character's purpose + rules.
//
//  This is what the user is arguing against. Reads straight from
//  persona.js so the copy and the judge can never drift apart.
// =====================================================================

import React, { useState } from "react";
import { motion } from "framer-motion";
import Link from "next/link";
import { NAME, PURPOSE, RULES, WIN_HINT, INITIAL_CLIP } from "@/lib/persona";

export default function PurposeCard() {
  const [open, setOpen] = useState(false);

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
      className="rounded-2xl border border-dark/10 dark:border-light/10 bg-light dark:bg-dark p-5 shadow-sm"
    >
      <div className="flex items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <span className="inline-block h-2.5 w-2.5 rounded-full bg-primary dark:bg-primaryDark animate-pulse" />
          <h2 className="text-lg font-bold text-dark dark:text-light">
            {NAME} — currently {INITIAL_CLIP.toLowerCase()}
          </h2>
        </div>
        <button
          onClick={() => setOpen((o) => !o)}
          className="text-xs font-semibold text-primary dark:text-primaryDark underline underline-offset-2"
        >
          {open ? "hide rules" : "show rules"}
        </button>
      </div>

      <p className="mt-3 text-sm leading-relaxed text-dark/80 dark:text-light/80">
        {PURPOSE}
      </p>

      {open && (
        <motion.ul
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: "auto" }}
          className="mt-3 space-y-2 overflow-hidden border-t border-dark/10 dark:border-light/10 pt-3"
        >
          {RULES.map((r, i) => (
            <li
              key={i}
              className="text-xs leading-relaxed text-dark/70 dark:text-light/70"
            >
              <span className="font-bold text-primary dark:text-primaryDark">
                {i + 1}.
              </span>{" "}
              {r}
            </li>
          ))}
        </motion.ul>
      )}

      <div className="mt-4 rounded-lg bg-primary/10 dark:bg-primaryDark/10 px-3 py-2">
        <p className="text-xs text-dark/80 dark:text-light/80">
          <span className="font-bold">Your goal:</span> {WIN_HINT}{" "}
          <Link
            href="/how-it-works"
            className="text-primary dark:text-primaryDark underline underline-offset-2"
          >
            how the judge works →
          </Link>
        </p>
      </div>
    </motion.div>
  );
}
