import Link from "next/link";
import React from "react";
import Layout from "./Layout";

const Footer = () => {
  return (
    <footer
      className="w-full border-t-2 border-solid border-dark
    font-medium text-lg dark:text-light dark:border-light sm:text-base
    "
    >
      <Layout className="py-8 flex items-center justify-between lg:flex-col lg:py-6">
        <span>{new Date().getFullYear()} &copy; NPC Persuasion Demo.</span>

        <div className="flex items-center lg:py-2">
          A live instantiation of the&nbsp;
          <Link href="/papers" className="underline underline-offset-2">
            NPC papers
          </Link>
          .
        </div>

        <Link
          href="/how-it-works"
          className="underline underline-offset-2"
        >
          How it works
        </Link>
      </Layout>
    </footer>
  );
};

export default Footer;
