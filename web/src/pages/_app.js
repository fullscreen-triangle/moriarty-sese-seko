import Footer from "@/components/Footer";
import Navbar from "@/components/Navbar";
import "@/styles/globals.css";
import { AnimatePresence } from "framer-motion";
// pages/_app.js
import Head from "next/head";
import { useRouter } from "next/router";

// We deliberately do NOT use next/font/google: it fetches the font from
// Google at build/dev time, which stalls (AbortError) on a slow or blocked
// network and can hang the whole compile. Instead we define --font-mont as
// a self-contained system font stack (Tailwind's font-mont reads it), so
// the app renders fully offline.
export default function App({ Component, pageProps }) {
  const router = useRouter();

  return (
    <>
      <Head>
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/favicon.ico" />
        <style>{`:root{--font-mont:'Segoe UI',system-ui,-apple-system,'Helvetica Neue',Arial,sans-serif;}`}</style>
      </Head>
      <main className="font-mont bg-light dark:bg-dark w-full min-h-screen h-full">
        <Navbar />
        <AnimatePresence initial={false} mode="wait">
          <Component key={router.asPath} {...pageProps} />
        </AnimatePresence>
        <Footer />
      </main>
    </>
  );
}
