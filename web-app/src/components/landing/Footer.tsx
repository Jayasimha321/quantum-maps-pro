'use client';

import { Twitter, Github } from 'lucide-react';

export function Footer() {
    return (
        <footer className="border-t border-light-border dark:border-dark-border py-12 bg-light-card/50 dark:bg-dark-card/50">
            <div className="max-w-7xl mx-auto px-6 flex flex-col md:flex-row justify-between items-center gap-6">
                <div className="flex items-center gap-2 grayscale opacity-70 hover:grayscale-0 hover:opacity-100 transition-all">
                    <div className="w-8 h-8 rounded-lg bg-violet-500/20 dark:bg-cyan-400/20 flex items-center justify-center text-violet-600 dark:text-cyan-400 font-bold">Q</div>
                    <span className="font-heading font-bold text-lg">Quantum Maps Pro</span>
                </div>
                <div className="text-sm text-gray-500">
                    &copy; 2026 Quantum Maps. All rights reserved.
                </div>
                <div className="flex gap-6 text-gray-400">
                    <a href="#" className="hover:text-violet-600 dark:hover:text-cyan-400 transition-colors">
                        <span className="sr-only">Twitter</span>
                        <Twitter className="h-5 w-5" />
                    </a>
                    <a href="#" className="hover:text-violet-600 dark:hover:text-cyan-400 transition-colors">
                        <span className="sr-only">GitHub</span>
                        <Github className="h-5 w-5" />
                    </a>
                </div>
            </div>
        </footer>
    );
}
