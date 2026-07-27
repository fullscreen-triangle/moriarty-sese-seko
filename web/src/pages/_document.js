import { Html, Head, Main, NextScript } from "next/document";

export default function Document() {
  return (
    // This demo is dark-mode only — the `dark` class is fixed on <html>
    // and there is no theme toggle. Set it here (before hydration) so there
    // is never a light flash.
    <Html lang="en" className="dark">
      <Head />
      <body>
        <Main />
        <NextScript />
      </body>
    </Html>
  );
}
