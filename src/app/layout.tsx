import type { Metadata } from "next";
import localFont from "next/font/local";
import "./globals.css";

const clashDisplay = localFont({
  src: "../fonts/ClashDisplay-Variable.woff2",
  display: "swap",
  variable: "--font-clash",
  weight: "200 700",
});

export const metadata: Metadata = {
  title: "ProductName — AI Video Editor",
  description:
    "Edit your videos with just an idea. AI-powered editing, no complexity.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={`${clashDisplay.variable} antialiased`}>
      <body className="font-[family-name:var(--font-clash)]">{children}</body>
    </html>
  );
}
