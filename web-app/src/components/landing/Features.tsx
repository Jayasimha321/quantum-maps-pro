'use client';

import { Activity, Zap, MapPin } from 'lucide-react';

export function Features() {
    return (
        <section id="features" className="py-20 bg-light-card/50 dark:bg-dark-card/50 backdrop-blur-sm border-t border-light-border dark:border-dark-border">
            <div className="max-w-7xl mx-auto px-6">
                <div className="text-center max-w-2xl mx-auto mb-16">
                    <span className="bg-gradient-to-r from-violet-600 to-fuchsia-600 dark:from-cyan-400 dark:to-purple-400 bg-clip-text text-transparent font-bold tracking-wider uppercase text-sm mb-2 block">Core Capabilities</span>
                    <h2 className="text-3xl lg:text-4xl font-heading font-bold mb-4">Engineered for <span className="text-gradient">Performance</span></h2>
                    <p className="text-gray-500 dark:text-gray-400">Everything you need to manage complex routes and track assets in real-time.</p>
                </div>

                <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
                    {/* Feature 1 */}
                    <div className="p-6 rounded-2xl bg-light-card dark:bg-dark-card border border-light-border dark:border-dark-border card-hover group cursor-pointer z-10 relative overflow-hidden">
                        <div className="absolute top-0 right-0 w-24 h-24 bg-secondary/5 rounded-full blur-[40px] group-hover:bg-secondary/10 transition-colors"></div>
                        <div className="w-12 h-12 rounded-xl bg-secondary/10 flex items-center justify-center text-secondary mb-4 group-hover:scale-110 transition-transform duration-300">
                            <Zap className="w-6 h-6" />
                        </div>
                        <h3 className="text-lg font-bold font-heading mb-2 group-hover:text-secondary transition-colors">QAOA Optimization</h3>
                        <p className="text-sm text-gray-500 dark:text-gray-400 leading-relaxed">Routes optimized by quantum algorithms for peak efficiency.</p>
                    </div>

                    {/* Feature 2 */}
                    <div className="p-6 rounded-2xl bg-light-card dark:bg-dark-card border border-light-border dark:border-dark-border card-hover group cursor-pointer z-10 relative overflow-hidden">
                        <div className="absolute top-0 right-0 w-24 h-24 bg-primary/5 rounded-full blur-[40px] group-hover:bg-primary/10 transition-colors"></div>
                        <div className="w-12 h-12 rounded-xl bg-primary/10 flex items-center justify-center text-primary mb-4 group-hover:scale-110 transition-transform duration-300">
                            <Activity className="w-6 h-6" />
                        </div>
                        <h3 className="text-lg font-bold font-heading mb-2 group-hover:text-primary transition-colors">Fleet Management</h3>
                        <p className="text-sm text-gray-500 dark:text-gray-400 leading-relaxed">Coordinate multi-vehicle fleets with seamless precision.</p>
                    </div>

                    {/* Feature 3 */}
                    <div className="p-6 rounded-2xl bg-light-card dark:bg-dark-card border border-light-border dark:border-dark-border card-hover group cursor-pointer z-10 relative overflow-hidden">
                        <div className="absolute top-0 right-0 w-24 h-24 bg-violet-500/5 dark:bg-cyan-400/5 rounded-full blur-[40px] group-hover:bg-violet-500/10 dark:group-hover:bg-cyan-400/10 transition-colors"></div>
                        <div className="w-12 h-12 rounded-xl bg-violet-500/10 dark:bg-cyan-400/10 flex items-center justify-center text-violet-600 dark:text-cyan-400 mb-4 group-hover:scale-110 transition-transform duration-300">
                            <MapPin className="w-6 h-6" />
                        </div>
                        <h3 className="text-lg font-bold font-heading mb-2 group-hover:text-violet-600 dark:group-hover:text-cyan-400 transition-colors">Live Tracking</h3>
                        <p className="text-sm text-gray-500 dark:text-gray-400 leading-relaxed">Real-time monitoring with millisecond-latency updates.</p>
                    </div>

                    {/* Feature 4 - New Smart POI */}
                    <div className="p-6 rounded-2xl bg-light-card dark:bg-dark-card border border-light-border dark:border-dark-border card-hover group cursor-pointer z-10 relative overflow-hidden">
                        <div className="absolute top-0 right-0 w-24 h-24 bg-cta/5 rounded-full blur-[40px] group-hover:bg-cta/10 transition-colors"></div>
                        <div className="w-12 h-12 rounded-xl bg-cta/10 flex items-center justify-center text-cta mb-4 group-hover:scale-110 transition-transform duration-300">
                            <Activity className="w-6 h-6" />
                        </div>
                        <h3 className="text-lg font-bold font-heading mb-2 group-hover:text-cta transition-colors">Smart POI Query</h3>
                        <p className="text-sm text-gray-500 dark:text-gray-400 leading-relaxed">Context-aware search for Fuel (Petrol/EV), Hotels, and more.</p>
                    </div>
                </div>
            </div>
        </section>
    );
}
