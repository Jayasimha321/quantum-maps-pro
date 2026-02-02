'use client';

import Link from 'next/link';
import { SplineScene } from "@/components/ui/splite";
import { Navbar } from "@/components/landing/Navbar";
import { Features } from "@/components/landing/Features";
import { GlobeSection } from "@/components/landing/GlobeSection";
import { Testimonials } from "@/components/landing/Testimonials";
import { Footer } from "@/components/landing/Footer";
import { PlayCircle, Navigation } from 'lucide-react';

export default function LandingPage() {
  return (
    <div className="min-h-screen relative">
      {/* Ambient Background Effects */}
      {/* Ambient Background Effects - Limited to Hero Section */}
      <div className="absolute top-0 left-0 w-full h-full z-0 pointer-events-none overflow-hidden">

        {/* Light Mode blobs - Deep Blue/Indigo for professional look */}
        <div className="absolute -top-[10%] -left-[10%] w-[40vw] h-[40vw] rounded-full blur-[100px] animate-pulse-slow bg-blue-500/20 dark:bg-primary/10"></div>
        <div
          className="absolute top-[20%] -right-[10%] w-[35vw] h-[35vw] rounded-full blur-[120px] animate-pulse-slow object-right-bottom bg-indigo-500/20 dark:bg-secondary/10"
          style={{ animationDelay: '2s' }}
        ></div>

        {/* Grid Overlay - Moved after blobs to be on top */}
        <div className="absolute inset-0 bg-grid opacity-80"></div>

        {/* Fade out removed to show grid on whole hero section */}
        {/* <div className="absolute bottom-0 left-0 w-full h-48 bg-gradient-to-t from-background to-transparent"></div> */}
      </div>

      <Navbar />

      {/* Hero Section */}
      <main className="relative z-10 pt-44 pb-20 lg:pt-56 lg:pb-32 px-6">
        <div className="max-w-7xl mx-auto grid lg:grid-cols-2 gap-16 items-start">

          {/* Hero Content - Top Left Aligned */}
          <div className="space-y-8 text-left">



            <h1 className="text-4xl sm:text-5xl lg:text-7xl font-heading font-bold leading-[1.1] tracking-tight w-fit text-center">
              <span className="whitespace-nowrap">Quantum&#8209;Powered</span> <br />
              <span className="text-gradient">Navigation</span>
            </h1>

            <p className="text-lg text-gray-600 dark:text-gray-400 max-w-xl leading-relaxed">
              22% more efficient routes. 4x faster computation. Experience the next generation of fleet optimization.
            </p>

            <div className="flex flex-col sm:flex-row items-center gap-4 justify-start pt-4">
              <a href="/q1.html" className="w-full sm:w-auto bg-cta text-white font-bold h-12 px-8 rounded-lg flex items-center justify-center gap-2 shadow-cta hover:scale-105 transition-transform duration-200">
                Start Navigating
                <Navigation className="w-5 h-5" />
              </a>

              <Link href="/robot-demo" className="w-full sm:w-auto h-12 px-8 rounded-lg border-2 border-violet-500 dark:border-cyan-400 font-bold hover:bg-violet-500/10 dark:hover:bg-cyan-400/10 transition-colors flex items-center justify-center gap-2">
                <span className="bg-gradient-to-r from-violet-600 to-fuchsia-600 dark:from-cyan-400 dark:to-purple-400 bg-clip-text text-transparent">View Demo</span>
                <PlayCircle className="w-5 h-5 text-violet-600 dark:text-cyan-400" />
              </Link>
            </div>

            <div className="pt-8 flex items-center justify-start gap-8 opacity-60 grayscale hover:grayscale-0 transition-all duration-500">
              <span className="font-heading font-bold text-xl">TRUSTED BY</span>
              <div className="flex gap-6">
                <svg className="h-6 w-auto" viewBox="0 0 100 30" fill="currentColor">
                  <path d="M10,15 L20,5 L30,15 L20,25 Z M40,5 H50 V25 H40 Z M60,5 H90 V10 H65 V12 H85 V17 H65 V25 H60 Z"></path>
                </svg>
                <svg className="h-6 w-auto" viewBox="0 0 100 30" fill="currentColor">
                  <circle cx="15" cy="15" r="10"></circle>
                  <rect x="35" y="5" width="20" height="20"></rect>
                  <rect x="65" y="5" width="20" height="20"></rect>
                </svg>
              </div>
            </div>
          </div>

          {/* Hero Visual - Robot */}
          <div className="relative lg:h-[750px] w-full flex items-center justify-center">
            {/* 1. Container Frame: 175% (Massive 'Safe Zone' to absolutely prevent clipping) */}
            <div className="absolute w-[175%] h-[175%] -top-[37.5%] -left-[37.5%] z-0 flex items-center justify-center">
              {/* 2. Robot Visual: Scaled to ~71% of the 175% container = ~125% of original size */}
              <SplineScene
                scene="https://prod.spline.design/kZDDjO5HuC9GJUM2/scene.splinecode"
                className="w-full h-full scale-[0.714]"
              />
            </div>
          </div>
        </div>
      </main>

      <Features />
      <GlobeSection />
      <Testimonials />
      <Footer />
    </div>
  );
}
