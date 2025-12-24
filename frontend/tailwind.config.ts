import type { Config } from 'tailwindcss'

const config: Config = {
  content: [
    './pages/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
    './app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        // Dark theme colors matching Streamlit app
        slate: {
          900: '#0F172A',
          800: '#1E293B',
          700: '#334155',
          600: '#475569',
          500: '#64748B',
          400: '#94A3B8',
          300: '#CBD5E1',
          200: '#E2E8F0',
          100: '#F1F5F9',
        },
        emerald: {
          500: '#10B981',
          400: '#34D399',
        },
        amber: {
          500: '#F59E0B',
          400: '#FBBF24',
        },
        red: {
          500: '#EF4444',
          400: '#F87171',
        },
        blue: {
          500: '#3B82F6',
          400: '#60A5FA',
        },
      },
    },
  },
  plugins: [],
}
export default config
