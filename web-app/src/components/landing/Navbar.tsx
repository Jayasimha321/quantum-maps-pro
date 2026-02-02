'use client';

import { useTheme } from "next-themes";
import Link from 'next/link';
import { Moon, Sun, Map, Check } from 'lucide-react';
import { useEffect, useState } from "react";

export function Navbar() {
    const { theme, setTheme } = useTheme();
    const [mounted, setMounted] = useState(false);

    useEffect(() => {
        setMounted(true);
    }, []);

    return (
        <nav className="fixed top-0 w-full z-50 transition-all duration-300 backdrop-blur-sm bg-white/5 dark:bg-black/5" id="navbar">
            <div className="max-w-7xl mx-auto px-6 h-20 flex items-center justify-between">
                {/* Logo */}
                <div className="flex items-center gap-3 group cursor-pointer">
                    <div className="relative w-10 h-10 rounded-xl bg-gradient-to-br from-primary to-secondary flex items-center justify-center text-white font-bold text-xl shadow-lg group-hover:shadow-neon transition-shadow duration-300">
                        Q
                        <div className="absolute inset-0 rounded-xl bg-white/20 opacity-0 group-hover:opacity-100 transition-opacity"></div>
                    </div>
                    <span className="font-heading font-bold text-xl tracking-tight group-hover:text-violet-600 dark:group-hover:text-cyan-400 transition-colors">Quantum Maps Pro</span>
                </div>

                {/* Desktop Menu */}
                <div className="hidden md:flex items-center gap-8">
                    <a href="#features" className="font-medium text-sm hover:text-violet-600 dark:hover:text-cyan-400 transition-colors">Features</a>
                    <a href="#reviews" className="font-medium text-sm hover:text-violet-600 dark:hover:text-cyan-400 transition-colors">Reviews</a>

                    <div className="h-6 w-px bg-gray-200 dark:bg-white/10"></div>

                    {/* Theme Toggle Button */}
                    <button
                        onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
                        className="p-2 rounded-full hover:bg-black/5 dark:hover:bg-white/10 transition-colors focus:outline-none focus:ring-2 focus:ring-primary"
                        aria-label="Toggle Theme"
                    >
                        {mounted && theme === 'dark' ? (
                            <Moon className="w-5 h-5 text-yellow-400" />
                        ) : (
                            <Sun className="w-5 h-5 text-slate-700" />
                        )}
                    </button>

                    <a href="https://quantum-maps-pro-frontend.onrender.com" className="relative group overflow-hidden bg-cta hover:bg-cta/90 text-white font-bold py-2.5 px-6 rounded-lg transition-all hover:bg-opacity-90 shadow-cta hover:shadow-lg hover:-translate-y-0.5">
                        <span className="relative z-10 flex items-center gap-2">
                            Launch App
                            <Map className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
                        </span>
                        <div className="absolute inset-0 bg-white/20 translate-y-full group-hover:translate-y-0 transition-transform duration-300"></div>
                    </a>
                </div>
            </div>
        </nav>
    );
}
