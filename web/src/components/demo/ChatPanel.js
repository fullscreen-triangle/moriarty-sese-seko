// =====================================================================
//  ChatPanel.js — the conversation with the character.
//
//  Owns the message history, posts each user turn to /api/persuade, shows
//  the character's reply and a verdict badge, and on a PERSUADED verdict
//  calls onCommit() with the new animation + the verdict (the page uses
//  this to cross-fade the model and increment the committed count).
// =====================================================================

import React, { useEffect, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";

const GREETING = {
  role: "assistant",
  content:
    "I'm running the final leg — I don't intend to stop. Make your case if you think I should.",
  verdict: null,
};

export default function ChatPanel({ currentAnimation, pStar, onCommit, onVerdict }) {
  const [messages, setMessages] = useState([GREETING]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const scrollRef = useRef(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, busy]);

  async function send(e) {
    e?.preventDefault();
    const text = input.trim();
    if (!text || busy) return;

    setError(null);
    setInput("");
    const userMsg = { role: "user", content: text };
    const history = [...messages, userMsg];
    setMessages(history);
    setBusy(true);

    try {
      // Send only role/content pairs; strip UI-only fields.
      const payload = history
        .filter((m) => m.role === "user" || m.role === "assistant")
        .map((m) => ({ role: m.role, content: m.content }));

      const res = await fetch("/api/persuade", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          messages: payload,
          pStar,
          currentAnimation,
        }),
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data?.error || "The judge is unavailable.");
      }

      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: data.reply, verdict: data },
      ]);

      // Report every verdict (keeps the instrument panel fresh); commit
      // only on a persuaded verdict that actually changes the clip.
      onVerdict?.(data);
      if (data.persuaded && data.targetAnimation !== currentAnimation) {
        onCommit?.(data.targetAnimation, data);
      }
    } catch (err) {
      setError(err.message);
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: "…(the judge could not be reached — I keep running.)",
          verdict: null,
        },
      ]);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="flex h-full min-h-0 flex-col rounded-2xl border border-dark/10 dark:border-light/10 bg-light dark:bg-dark">
      <div
        ref={scrollRef}
        className="flex-1 min-h-0 space-y-3 overflow-y-auto p-4"
      >
        <AnimatePresence initial={false}>
          {messages.map((m, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}
            >
              <div
                className={`max-w-[85%] rounded-2xl px-3.5 py-2 text-sm leading-relaxed ${
                  m.role === "user"
                    ? "bg-primary text-light dark:bg-primaryDark dark:text-dark"
                    : "bg-dark/5 text-dark dark:bg-light/10 dark:text-light"
                }`}
              >
                {m.content}
                {m.verdict && (
                  <div
                    className={`mt-1.5 text-[10px] font-bold uppercase tracking-wide ${
                      m.verdict.persuaded
                        ? "text-primary dark:text-primaryDark"
                        : "text-dark/40 dark:text-light/40"
                    }`}
                  >
                    {m.verdict.persuaded
                      ? `persuaded → ${m.verdict.targetAnimation}`
                      : "not persuaded"}
                  </div>
                )}
              </div>
            </motion.div>
          ))}
        </AnimatePresence>

        {busy && (
          <div className="flex justify-start">
            <div className="rounded-2xl bg-dark/5 px-3.5 py-2 text-sm text-dark/50 dark:bg-light/10 dark:text-light/50">
              <span className="animate-pulse">weighing your argument…</span>
            </div>
          </div>
        )}
      </div>

      {error && (
        <div className="px-4 pb-1 text-[11px] text-red-500">{error}</div>
      )}

      <form
        onSubmit={send}
        className="flex items-center gap-2 border-t border-dark/10 dark:border-light/10 p-3"
      >
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Argue your case…"
          disabled={busy}
          className="flex-1 rounded-xl bg-dark/5 dark:bg-light/10 px-3.5 py-2 text-sm text-dark dark:text-light placeholder:text-dark/40 dark:placeholder:text-light/40 outline-none focus:ring-2 focus:ring-primary dark:focus:ring-primaryDark disabled:opacity-50"
        />
        <button
          type="submit"
          disabled={busy || !input.trim()}
          className="rounded-xl bg-primary dark:bg-primaryDark px-4 py-2 text-sm font-bold text-light dark:text-dark disabled:opacity-40"
        >
          Send
        </button>
      </form>
    </div>
  );
}
