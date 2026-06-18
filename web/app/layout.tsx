import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Finance AI",
  description: "RAG chat over market data and SEC filings"
};

export default function RootLayout({
  children
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
