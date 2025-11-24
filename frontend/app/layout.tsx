import type { Metadata } from "next";
import { Be_Vietnam_Pro } from "next/font/google";
import "./globals.css";
// Import Provider vừa tạo
import { ThemeProvider } from "./providers";

const beVietnamPro = Be_Vietnam_Pro({
  variable: "--font-geist-sans",
  subsets: ["latin", "vietnamese"],
  weight: ["100", "200", "300", "400", "500", "600", "700", "800", "900"],
  display: 'swap',
});

export const metadata: Metadata = {
  title: "QR-AILAB",
  description: "Nền tảng tạo mã QR thông minh bằng AI",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    // suppressHydrationWarning cần thiết cho next-themes để tránh lỗi hydrate
    <html lang="vi" suppressHydrationWarning>
      <head />
      <body className={`${beVietnamPro.variable} antialiased`}>
        {/* Bọc ThemeProvider ở đây */}
        <ThemeProvider
          attribute="class"
          defaultTheme="dark" // Mặc định là Dark Mode
          enableSystem
          disableTransitionOnChange
        >
          {children}
        </ThemeProvider>
      </body>
    </html>
  );
}