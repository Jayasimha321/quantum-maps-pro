import { Config } from "tailwindcss";

const config: Config = {
    darkMode: ["class"],
    content: [
        "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
        "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
        "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
    ],
    theme: {
        extend: {
            colors: {
                border: "hsl(var(--border))",
                input: "hsl(var(--input))",
                ring: "hsl(var(--ring))",
                background: "hsl(var(--background))",
                foreground: "hsl(var(--foreground))",
                primary: {
                    DEFAULT: '#00FFFF', // Custom Cyan (overrides Shadcn primary)
                    foreground: "hsl(var(--primary-foreground))",
                },
                secondary: {
                    DEFAULT: '#7B61FF', // Custom Purple (overrides Shadcn secondary)
                    foreground: "hsl(var(--secondary-foreground))",
                },
                destructive: {
                    DEFAULT: "hsl(var(--destructive))",
                    foreground: "hsl(var(--destructive-foreground))",
                },
                muted: {
                    DEFAULT: "hsl(var(--muted))",
                    foreground: "hsl(var(--muted-foreground))",
                },
                accent: {
                    DEFAULT: "hsl(var(--accent))",
                    foreground: "hsl(var(--accent-foreground))",
                },
                popover: {
                    DEFAULT: "hsl(var(--popover))",
                    foreground: "hsl(var(--popover-foreground))",
                },
                card: {
                    DEFAULT: "hsl(var(--card))",
                    foreground: "hsl(var(--card-foreground))",
                },
                // Custom Legacy Colors
                cta: '#FF00FF', // Magenta
                dark: {
                    bg: '#050510',
                    card: '#0A0A18',
                    text: '#E0E0FF',
                    border: 'rgba(255, 255, 255, 0.1)'
                },
                light: {
                    bg: '#F8FAFC',
                    card: '#FFFFFF',
                    text: '#0F172A',
                    border: '#E2E8F0'
                }
            },
            fontFamily: {
                sans: ['var(--font-dm-sans)', 'sans-serif'],
                heading: ['var(--font-space-grotesk)', 'sans-serif'],
            },
            boxShadow: {
                'neon': '0 0 20px rgba(0, 255, 255, 0.2)',
                'cta': '0 0 20px rgba(255, 0, 255, 0.4)',
                'glass': '0 8px 32px 0 rgba(0, 0, 0, 0.3)',
                'soft-light': '0 20px 40px -10px rgba(0, 0, 0, 0.05)', // New soft shadow for light mode
            },
            animation: {
                'float': 'float 6s ease-in-out infinite',
                'pulse-slow': 'pulse 8s cubic-bezier(0.4, 0, 0.6, 1) infinite', // Slower pulse
            },
            keyframes: {
                float: {
                    '0%, 100%': { transform: 'translateY(0)' },
                    '50%': { transform: 'translateY(-20px)' },
                }
            },
            borderRadius: {
                lg: 'var(--radius)',
                md: 'calc(var(--radius) - 2px)',
                sm: 'calc(var(--radius) - 4px)'
            }
        }
    },
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    plugins: [require("tailwindcss-animate")],
};
export default config;
