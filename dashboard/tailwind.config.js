/** @type {import('tailwindcss').Config} */
export default {
  content: ['./src/**/*.{js,ts,jsx,tsx,mdx}'],
  theme: {
    extend: {
      colors: {
        // X-Ray theme colors
        void: '#000000',
        primary: {
          DEFAULT: '#0a0a0f',
          50: '#f0f0f5',
          100: '#e0e0e8',
          200: '#c0c0d0',
          300: '#a0a0b0',
          400: '#808090',
          500: '#606070',
          600: '#404050',
          700: '#2a2a35',
          800: '#1a1a25',
          900: '#12121a',
          950: '#0a0a0f',
        },
        accent: {
          DEFAULT: '#6366f1',
          hover: '#818cf8',
          glow: 'rgba(99, 102, 241, 0.4)',
        },
        success: '#22c55e',
        warning: '#eab308',
        error: '#ef4444',
        border: {
          DEFAULT: '#2a2a35',
          light: '#e5e5e5',
        },
      },
      fontFamily: {
        sans: ['Inter Variable', 'Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono Variable', 'JetBrains Mono', 'Consolas', 'monospace'],
        serif: ['Georgia', 'Cambria', 'Times New Roman', 'serif'],
      },
      fontSize: {
        'massive': ['clamp(4rem, 15vw, 12rem)', { lineHeight: '0.75', letterSpacing: '-0.04em' }],
      },
      animation: {
        'xray-pulse': 'xray-pulse 3s ease-in-out infinite',
      },
      keyframes: {
        'xray-pulse': {
          '0%, 100%': {
            filter: 'invert(1) brightness(1.2) contrast(1.1) drop-shadow(0 0 10px rgba(99, 102, 241, 0.6))',
          },
          '50%': {
            filter: 'invert(1) brightness(1.5) contrast(1.2) drop-shadow(0 0 30px rgba(99, 102, 241, 0.6))',
          },
        },
      },
    },
  },
  plugins: [],
};
