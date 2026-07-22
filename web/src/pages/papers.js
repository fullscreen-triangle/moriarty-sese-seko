// =====================================================================
//  papers.js — the three NPC manuscripts the demo instantiates.
// =====================================================================

import Head from "next/head";
import Link from "next/link";
import { motion } from "framer-motion";
import AnimatedText from "@/components/AnimatedText";
import Layout from "@/components/Layout";
import TransitionEffect from "@/components/TransitionEffect";

const PAPERS = [
  {
    title: "The Purpose of a Character",
    subtitle:
      "Identity as a Conserved Invariant and Purpose as a Fixed Point",
    blurb:
      "An agent is a finite weighted self-graph. From three axioms it carries a conserved identity and, under a single-basin drive, a unique purpose that attracts — the thing the character is drawn back toward. In the demo, this is why XBOT has a purpose to argue against.",
    maps: "→ the character's purpose",
    pdf: "/papers/purpose.pdf",
  },
  {
    title: "The Physiology of Response",
    subtitle: "When an Interaction Is Relevant, and What a Response Costs",
    blurb:
      "Relevance is two-factor: a response is admissible only if it both advances the agent's purpose and preserves its coherence — neither alone suffices. Attention is priced by a threshold p*, and a committed response is a floored, irreversible deposit. This is the judge behind the demo.",
    maps: "→ the two-factor judge, p*, and irreversibility",
    pdf: "/papers/physiology.pdf",
  },
  {
    title: "Propagation Mechanics of an Embedded Agent",
    subtitle: "How Testimony, Divergence, and Inquiry Arise Without Representation",
    blurb:
      "The character answers by search, not by fetching a script; its replies are freshly produced and it holds no internal concept of 'being persuaded'. The verdict is the observer's computation, never a state the agent represents.",
    maps: "→ replies are searched, verdicts are ours",
    pdf: "/papers/embedded-agent.pdf",
  },
];

export default function Papers() {
  return (
    <>
      <Head>
        <title>The Papers — NPC Persuasion Demo</title>
        <meta
          name="description"
          content="The three manuscripts the demo instantiates: purpose, two-factor response, and propagation mechanics."
        />
      </Head>
      <TransitionEffect />
      <main className="mb-16 flex w-full flex-col items-center justify-center dark:text-light">
        <Layout className="pt-16">
          <AnimatedText
            text="Three papers, one running character."
            className="mb-16 !text-6xl !leading-tight lg:!text-5xl sm:!text-3xl xs:!text-2xl sm:mb-8"
          />
          <div className="grid grid-cols-1 gap-8">
            {PAPERS.map((p, i) => (
              <motion.article
                key={i}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.4, delay: i * 0.05 }}
                className="rounded-2xl border border-dark/10 bg-light p-6 dark:border-light/10 dark:bg-dark"
              >
                <div className="text-xs font-semibold uppercase tracking-wide text-primary dark:text-primaryDark">
                  {p.maps}
                </div>
                <h2 className="mt-2 text-2xl font-bold">{p.title}</h2>
                <p className="text-sm italic text-dark/60 dark:text-light/60">
                  {p.subtitle}
                </p>
                <p className="mt-3 text-base leading-relaxed text-dark/80 dark:text-light/80">
                  {p.blurb}
                </p>
                <Link
                  href={p.pdf}
                  target="_blank"
                  className="mt-4 inline-block rounded-lg bg-dark px-4 py-2 text-sm font-bold text-light dark:bg-light dark:text-dark"
                >
                  Read the PDF →
                </Link>
              </motion.article>
            ))}
          </div>
          <p className="mt-10 text-sm text-dark/50 dark:text-light/50">
            PDFs are built from the LaTeX sources in{" "}
            <code>publications/</code>. If a link 404s, the PDF has not been
            copied into <code>web/public/papers/</code> yet.
          </p>
        </Layout>
      </main>
    </>
  );
}
