import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Finance AI",
  description: "Daily market close emails with a news-grounded finance assistant"
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
