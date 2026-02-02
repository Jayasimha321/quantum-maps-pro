'use client';

import { Globe } from "@/components/ui/globe";
import { Server, Zap, Shield } from "lucide-react";

export function GlobeSection() {
    return (
        <section className="relative py-24 overflow-hidden">
            <div className="max-w-7xl mx-auto px-6 relative z-10">
                <div className="grid lg:grid-cols-2 gap-16 items-center">
                    {/* Left Content: Productive/Dashboard Look */}
                    <div className="space-y-8">
                        <div>
                            <span className="bg-gradient-to-r from-violet-600 to-fuchsia-600 dark:from-cyan-400 dark:to-purple-400 bg-clip-text text-transparent font-bold tracking-wider uppercase text-sm mb-2 block animate-pulse">
                                <span className="inline-block w-2 h-2 rounded-full bg-green-500 mr-2"></span>
                                System Status: Operational
                            </span>
                            <h2 className="text-3xl lg:text-5xl font-heading font-bold mb-6">
                                Quantum Optimization <br />
                                <span className="text-gradient">Network</span>
                            </h2>
                            <p className="text-lg text-gray-500 dark:text-gray-400 leading-relaxed">
                                Our edge network spans 180+ countries, ensuring sub-50ms latency for 99% of global devices. Real-time routing intelligence updates every millisecond.
                            </p>
                        </div>

                        {/* Metrics Grid */}
                        <div className="grid grid-cols-2 gap-4">
                            <div className="p-4 rounded-xl bg-white/80 dark:bg-white/5 shadow-sm backdrop-blur-sm transition-all hover:bg-white hover:shadow-md dark:hover:bg-white/10">
                                <div className="flex items-center gap-2 mb-2 text-gray-500 dark:text-gray-400">
                                    <Server className="w-4 h-4" />
                                    <span className="text-xs font-bold uppercase tracking-wide">Route Efficiency</span>
                                </div>
                                <div className="text-2xl font-bold font-mono text-foreground">
                                    +22%
                                </div>
                            </div>

                            <div className="p-4 rounded-xl bg-white/80 dark:bg-white/5 shadow-sm backdrop-blur-sm transition-all hover:bg-white hover:shadow-md dark:hover:bg-white/10">
                                <div className="flex items-center gap-2 mb-2 text-gray-500 dark:text-gray-400">
                                    <Zap className="w-4 h-4" />
                                    <span className="text-xs font-bold uppercase tracking-wide">Speed</span>
                                </div>
                                <div className="text-2xl font-bold font-mono text-foreground">
                                    4-8x Faster
                                </div>
                            </div>

                            <div className="p-4 rounded-xl bg-white/80 dark:bg-white/5 shadow-sm backdrop-blur-sm transition-all hover:bg-white hover:shadow-md dark:hover:bg-white/10">
                                <div className="flex items-center gap-2 mb-2 text-gray-500 dark:text-gray-400">
                                    <Shield className="w-4 h-4" />
                                    <span className="text-xs font-bold uppercase tracking-wide">Fuel Savings</span>
                                </div>
                                <div className="text-2xl font-bold font-mono text-green-500">
                                    14.8%
                                </div>
                            </div>

                            <div className="p-4 rounded-xl bg-white/80 dark:bg-white/5 shadow-sm backdrop-blur-sm transition-all hover:bg-white hover:shadow-md dark:hover:bg-white/10">
                                <div className="flex items-center gap-2 mb-2 text-gray-500 dark:text-gray-400">
                                    <span className="relative flex h-3 w-3">
                                        <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-violet-500 dark:bg-cyan-400 opacity-75"></span>
                                        <span className="relative inline-flex rounded-full h-3 w-3 bg-violet-500 dark:bg-cyan-400"></span>
                                    </span>
                                    <span className="text-xs font-bold uppercase tracking-wide">CO₂ Reduction</span>
                                </div>
                                <div className="text-2xl font-bold font-mono text-violet-600 dark:text-cyan-400">
                                    15%
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* Right Content: Large Interactive Globe */}
                    <div className="relative h-[600px] w-full flex items-center justify-center lg:justify-end">
                        <div className="absolute inset-0 bg-gradient-to-tr from-primary/20 to-secondary/20 rounded-full blur-[100px] opacity-20"></div>

                        {/* Productive look container with tight frame */}
                        <div className="relative w-full max-w-[800px] aspect-square border-2 border-slate-200/50 dark:border-white/10 rounded-full shadow-2xl bg-white/5 backdrop-blur-sm overflow-hidden">
                            <Globe className="w-full h-full opacity-100 max-w-none" />
                            {/* Overlay Gradient to blend bottom */}
                            <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_50%_120%,rgba(0,0,0,0.5),transparent)]" />
                        </div>
                    </div>
                </div>
            </div>
        </section>
    );
}
