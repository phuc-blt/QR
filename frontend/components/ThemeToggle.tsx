"use client";

import { useState, useEffect } from "react";
import { useTheme } from "next-themes";

export function ThemeToggle() {
  const [mounted, setMounted] = useState(false);
  const { theme, setTheme } = useTheme();

  useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) {
    return <div className="w-10 h-10" />;
  }

  return (
    <button
      onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
      // Thêm transition mượt cho nền và viền nút toggle
      className="fixed top-6 right-6 z-50 p-2 rounded-full bg-white/20 dark:bg-white/10 backdrop-blur-md border border-slate-300 dark:border-white/20 shadow-lg hover:scale-110 transition-all duration-500 group"
      aria-label="Toggle Theme"
    >
      {theme === "dark" ? (
        <svg className="w-6 h-6 text-yellow-300 transition-transform duration-500 rotate-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z" />
        </svg>
      ) : (
        // Thêm hiệu ứng xoay khi đổi icon
        <svg className="w-6 h-6 text-orange-500 transition-transform duration-500 rotate-90" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364-6.364l-.707.707M6.343 17.657l-.707.707M12 5a7 7 0 100 14 7 7 0 000-14z" />
        </svg>
      )}
    </button>
  );
}