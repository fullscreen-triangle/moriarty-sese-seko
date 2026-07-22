// =====================================================================
//  /api/persuade — the judge endpoint.
//
//  POST { messages: [{role,content}], pStar?, currentAnimation? }
//   ->  { purposeGain, coherence, requestedAnimation, reply,
//         persuaded, targetAnimation, reason, provider, parseError }
//
//  Flow: build the system prompt -> call the configured LLM (Ollama or
//  HF) -> parse the model's two-factor scores -> run the DETERMINISTIC
//  two-factor gate (scorer.js) server-side. The model never decides the
//  verdict; it only scores. A soft model failure degrades to
//  "not persuaded", never a 500 for the user.
// =====================================================================

import { buildMessages, parseJudgeReply } from "@/lib/prompt";
import { decide, pickAnimation, DEFAULT_P_STAR } from "@/lib/scorer";
import { INITIAL_CLIP } from "@/lib/persona";
import { callOllama } from "@/lib/llm/ollama";
import { callHF } from "@/lib/llm/hf";

export default async function handler(req, res) {
  if (req.method !== "POST") {
    res.setHeader("Allow", "POST");
    return res.status(405).json({ error: "Method not allowed" });
  }

  const {
    messages = [],
    pStar = DEFAULT_P_STAR,
    currentAnimation = INITIAL_CLIP,
  } = req.body || {};

  if (!Array.isArray(messages) || messages.length === 0) {
    return res.status(400).json({ error: "messages must be a non-empty array" });
  }

  const provider = (process.env.LLM_PROVIDER || "ollama").toLowerCase();
  const chat = buildMessages(messages);

  let rawReply;
  try {
    rawReply = provider === "hf" ? await callHF(chat) : await callOllama(chat);
  } catch (err) {
    // Backend unreachable/misconfigured: surface a clear, actionable error.
    return res.status(502).json({
      error: err?.message || "LLM backend error",
      provider,
    });
  }

  const parsed = parseJudgeReply(rawReply);
  const verdict = decide({
    purposeGain: parsed.purposeGain,
    coherence: parsed.coherence,
    pStar,
  });
  const targetAnimation = pickAnimation(
    verdict.persuaded,
    parsed.requestedAnimation,
    currentAnimation
  );

  return res.status(200).json({
    provider,
    purposeGain: verdict.gain,
    coherence: verdict.coherence,
    pStar: verdict.pStar,
    persuaded: verdict.persuaded,
    clearsPurpose: verdict.clearsPurpose,
    preservesCoherence: verdict.preservesCoherence,
    requestedAnimation: parsed.requestedAnimation,
    targetAnimation,
    reply: parsed.reply,
    reason: verdict.reason,
    parseError: parsed.parseError,
  });
}
