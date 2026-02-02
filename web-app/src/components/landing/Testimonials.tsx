'use client';

import { Star } from 'lucide-react';

export function Testimonials() {
    return (
        <section id="reviews" className="py-20">
            <div className="max-w-7xl mx-auto px-6">
                <div className="flex flex-col lg:flex-row gap-12 items-center">
                    <div className="lg:w-1/3">
                        <h2 className="text-3xl font-heading font-bold mb-6">Loved by<br /><span className="text-gradient">Innovators</span></h2>
                        <p className="text-gray-500 dark:text-gray-400 mb-8">Join thousands of users who have optimized their navigation workflow.</p>
                        <div className="flex items-center gap-4">
                            <div className="flex -space-x-4">
                                <div className="w-12 h-12 rounded-full border-2 border-dark-bg bg-gray-600"></div>
                                <div className="w-12 h-12 rounded-full border-2 border-dark-bg bg-gray-500"></div>
                                <div className="w-12 h-12 rounded-full border-2 border-dark-bg bg-gray-400"></div>
                            </div>
                            <div>
                                <div className="font-bold text-lg">2,000+</div>
                                <div className="text-xs text-gray-500">Active Users</div>
                            </div>
                        </div>
                    </div>

                    <div className="lg:w-2/3 grid md:grid-cols-2 gap-6">
                        <div className="p-6 rounded-2xl bg-light-card dark:bg-dark-card border border-light-border dark:border-dark-border shadow-sm">
                            <div className="flex text-yellow-400 mb-4">
                                {[...Array(5)].map((_, i) => (
                                    <Star key={i} className="w-4 h-4 fill-current" />
                                ))}
                            </div>
                            <p className="text-sm leading-relaxed italic mb-4">"The quantum routing algorithms cut our delivery times by 18%. The efficiency gains are absolutely real."</p>
                            <div className="font-bold text-sm text-violet-600 dark:text-cyan-400">Sarah J.</div>
                            <div className="text-xs text-gray-500">Logistics Manager</div>
                        </div>

                        <div className="p-6 rounded-2xl bg-light-card dark:bg-dark-card border border-light-border dark:border-dark-border shadow-sm">
                            <div className="flex text-yellow-400 mb-4">
                                {[...Array(5)].map((_, i) => (
                                    <Star key={i} className="w-4 h-4 fill-current" />
                                ))}
                            </div>
                            <p className="text-sm leading-relaxed italic mb-4">"Smart POI Query is a game changer. Finding specific fuel types for our hybrid fleet is now instant."</p>
                            <div className="font-bold text-sm text-secondary">Mike R.</div>
                            <div className="text-xs text-gray-500">Fleet Coordinator</div>
                        </div>
                    </div>
                </div>
            </div>
        </section>
    );
}
