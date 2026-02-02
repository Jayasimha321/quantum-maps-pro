
import type { Metadata } from "next";
import { DM_Sans, Space_Grotesk } from "next/font/google";
import "./globals.css";
import { ThemeProvider } from "@/components/theme-provider";

const dmSans = DM_Sans({
  subsets: ["latin"],
  weight: ['400', '500', '700'],
  variable: '--font-dm-sans',
});

const spaceGrotesk = Space_Grotesk({
  subsets: ["latin"],
  weight: ['400', '500', '600', '700'],
  variable: '--font-space-grotesk',
});

export const metadata: Metadata = {
  title: "Quantum Maps Pro - Navigate the Future",
  description: "Experience the next generation of routing intelligence.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={`${dmSans.variable} ${spaceGrotesk.variable} antialiased bg-light-bg text-light-text dark:bg-dark-bg dark:text-dark-text transition-colors duration-300 overflow-x-hidden selection:bg-primary selection:text-black`}>
        <ThemeProvider
          attribute="class"
          defaultTheme="system"
          enableSystem
          disableTransitionOnChange
        >
          {children}
        </ThemeProvider>
      </body>
    </html>
  );
}
