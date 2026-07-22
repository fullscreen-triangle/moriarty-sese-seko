// =====================================================================
//  how-it-works.js — explains the judge and the LLM plumbing.
// =====================================================================

import Head from "next/head";
import Link from "next/link";
import { motion } from "framer-motion";
import AnimatedText from "@/components/AnimatedText";
import Layout from "@/components/Layout";
import TransitionEffect from "@/components/TransitionEffect";

function Step({ n, title, children }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      className="rounded-2xl border border-dark/10 bg-light p-6 dark:border-light/10 dark:bg-dark"
    >
      <div className="flex items-center gap-3">
        <span className="flex h-8 w-8 items-center justify-center rounded-full bg-primary text-sm font-bold text-light dark:bg-primaryDark dark:text-dark">
          {n}
        </span>
        <h2 className="text-xl font-bold">{title}</h2>
      </div>
      <div className="mt-3 text-base leading-relaxed text-dark/80 dark:text-light/80">
        {children}
      </div>
    </motion.div>
  );
}

export default function HowItWorks() {
  return (
    <>
      <Head>
        <title>How it works — NPC Persuasion Demo</title>
        <meta
          name="description"
          content="The two-factor relevance judge, the p* threshold, irreversibility, and the switchable Ollama / Hugging Face backend."
        />
      </Head>
      <TransitionEffect />
      <main className="mb-16 flex w-full flex-col items-center justify-center dark:text-light">
        <Layout className="pt-16">
          <AnimatedText
            text="How the judge decides."
            className="mb-12 !text-6xl !leading-tight lg:!text-5xl sm:!text-3xl xs:!text-2xl sm:mb-8"
          />

          <div className="grid grid-cols-1 gap-6">
            <Step n="1" title="The character has a purpose">
              XBOT is running the final leg of an endurance trial. Running is
              not a mood — it is what it is <em>for</em>. Every message you send
              is an <b>interaction</b>; most interactions are simply irrelevant
              to a purpose, and the correct behaviour is to let them pass. This
              is the setup from{" "}
              <Link href="/papers" className="text-primary underline underline-offset-2 dark:text-primaryDark">
                The Purpose of a Character
              </Link>
              .
            </Step>

            <Step n="2" title="Two-factor relevance — both bars, or nothing">
              A local or hosted LLM reads your argument and scores it on two
              independent factors, each in [0, 1]:
              <ul className="mt-2 list-disc pl-6">
                <li>
                  <b>purpose gain</b> — does a response advance the purpose, or
                  show that continuing defeats it?
                </li>
                <li>
                  <b>coherence</b> — would changing behaviour keep the character
                  consistent, or force it to contradict itself?
                </li>
              </ul>
              The character is persuaded <b>only if both bars clear</b>: purpose
              gain must exceed the price <code>p*</code>, <em>and</em> coherence
              must hold. Neither factor alone is enough — the central result of{" "}
              <Link href="/papers" className="text-primary underline underline-offset-2 dark:text-primaryDark">
                The Physiology of Response
              </Link>
              .
            </Step>

            <Step n="3" title="The price p* — why a busy agent is hard to reach">
              <code>p*</code> is the attention price: the gain a response must
              clear to be worth committing. It is the red line on the
              purpose-gain meter in <b>Research mode</b>. A weak or convenient
              argument lands below it and is declined — not because the character
              is broken, but because the argument fell below its price of
              attention.
            </Step>

            <Step n="4" title="Commitment is irreversible">
              When the character is persuaded, it changes what it does and the{" "}
              <b>committed count</b> ticks up by one. That count never
              decreases: a committed change is a one-way deposit, so the
              character you meet after a change is in a strictly later state than
              before. Turn on Research mode to watch the count grow.
            </Step>

            <Step n="5" title="The verdict is ours, not the character's">
              The character never represents &ldquo;I have been
              persuaded.&rdquo; It answers by search, and a deterministic gate
              (not the LLM) computes the verdict from the two scores. The label
              is the observer&rsquo;s — the framing from{" "}
              <Link href="/papers" className="text-primary underline underline-offset-2 dark:text-primaryDark">
                Propagation Mechanics of an Embedded Agent
              </Link>
              .
            </Step>

            <Step n="6" title="The LLM backend is switchable">
              The judge runs through one API route,{" "}
              <code>/api/persuade</code>, behind a provider switch. Set{" "}
              <code>LLM_PROVIDER=ollama</code> to use a local Ollama model
              (offline, no cost), or <code>LLM_PROVIDER=hf</code> with an{" "}
              <code>HF_TOKEN</code> to use the Hugging Face Inference API when
              deployed. The model only <em>scores</em>; the gate decides.
            </Step>
          </div>
        </Layout>
      </main>
    </>
  );
}
