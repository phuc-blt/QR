import { Link, asset } from '@/lib/link';
import { ThemeToggle } from '@/components/ThemeToggle';

export const metadata = {
  title: "QR-AILAB - DiffQRCoder",
  description: "Nền tảng tạo mã QR thông minh bằng AI",
};

export default function Home() {
  // Class gradient cho chữ
  const gradientTextClass = "bg-gradient-to-r from-[#3B82F6] via-[#8B5CF6] to-[#EC4899] bg-clip-text text-transparent";
  
  // Class chung cho transition mượt (áp dụng cho mọi thành phần)
  const smoothTransition = "transition-all duration-500 ease-in-out";

  return (
    <main className={`min-h-screen flex items-center justify-center px-4 sm:px-6 relative overflow-hidden bg-background text-foreground ${smoothTransition}`}>
      
      <ThemeToggle />

      {/* Hiệu ứng nền Gradient Orb (Chỉ hiện rõ ở Dark Mode) */}
      <div className={`absolute inset-0 overflow-hidden pointer-events-none ${smoothTransition}`}>
        <div className="absolute top-[-10%] right-[-5%] w-[600px] h-[600px] bg-blue-600/10 rounded-full blur-[120px] opacity-0 dark:opacity-40 mix-blend-normal transition-opacity duration-700"></div>
        <div className="absolute bottom-[-10%] left-[-5%] w-[600px] h-[600px] bg-purple-600/10 rounded-full blur-[120px] opacity-0 dark:opacity-40 mix-blend-normal transition-opacity duration-700"></div>
      </div>

      <div className="relative z-10 text-center max-w-4xl w-full p-4">
        
        {/* Logo */}
        <div className="mb-10 transition-transform duration-300 hover:scale-105">
          <img
            src={asset('/logo.png')} 
            alt="QR-AILAB"
            className="mx-auto h-32 w-auto object-contain drop-shadow-2xl"
            loading="eager"
          />
        </div>

        {/* Heading */}
        <h1 className={`text-6xl sm:text-7xl font-black mb-8 leading-tight tracking-tight pb-2 ${gradientTextClass}`}>
          Chào mừng đến với<br/>
          QR-AILAB
        </h1>

        {/* Mô tả */}
        <p className={`text-slate-700 dark:text-blue-100 text-xl sm:text-2xl font-medium leading-relaxed max-w-3xl mx-auto mb-14 ${smoothTransition}`}>
          <strong className={gradientTextClass}>DiffQRCoder</strong> – Nền tảng tạo mã QR thông minh 
          và độc đáo bằng câu lệnh.
        </p>

        {/* Buttons */}
        <div className="flex flex-col sm:flex-row justify-center gap-6 mb-20">
          {/* NÚT BẮT ĐẦU NGAY: 
              - Light Mode: Nền đen (hoặc gradient), chữ trắng.
              - Dark Mode: Nền trắng, chữ đen.
          */}
          <Link
            href="/chat"
            className={`group relative inline-flex items-center justify-center px-12 py-5 rounded-full font-bold text-xl shadow-xl hover:scale-105 active:scale-95 ${smoothTransition}
              bg-slate-900 text-white dark:bg-white dark:text-slate-900
            `}
          >
            <span className="flex items-center gap-2">
              Bắt đầu ngay
              <svg className="w-6 h-6 transition-transform duration-300 group-hover:translate-x-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M13 7l5 5m0 0l-5 5m5-5H6" />
              </svg>
            </span>
          </Link>
          
          {/* NÚT XEM THÊM: (Sửa lỗi chữ trắng ở nền sáng)
              - Light Mode: Viền xám đậm, Chữ xám đậm (slate-800).
              - Dark Mode: Viền trắng mờ, Chữ trắng.
          */}
          <Link
            href="/xemthem"
            className={`group inline-flex items-center justify-center px-12 py-5 rounded-full font-bold text-xl bg-transparent border-2 hover:scale-105 active:scale-95 ${smoothTransition}
              border-slate-400 text-slate-800 hover:bg-slate-200/50 
              dark:border-slate-500 dark:text-white dark:hover:bg-white/10 dark:hover:border-white
            `}
          >
            <span className="flex items-center gap-2">
              Xem thêm
              <svg className="w-6 h-6 transition-transform duration-300 group-hover:rotate-12" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
              </svg>
            </span>
          </Link>
        </div>

        {/* Feature badges */}
        <div className={`grid grid-cols-3 gap-8 pt-8 border-t border-slate-300 dark:border-gray-800/60 ${smoothTransition}`}>
          
          {/* Feature 1 */}
          <div className="group text-center transition-transform duration-300 hover:scale-110 cursor-default">
            {/* SỬA LỖI ICON: Light mode dùng bg-white để nổi bật trên nền xám, Dark mode dùng bg tối */}
            <div className={`mx-auto w-16 h-16 rounded-full flex items-center justify-center mb-4 shadow-md group-hover:border-blue-500 border border-transparent ${smoothTransition}
              bg-white dark:bg-[#1e293b] dark:border-slate-700
            `}>
                <svg className={`w-8 h-8 ${smoothTransition} text-blue-600 dark:text-blue-400`} fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" /></svg>
            </div>
            <p className={`text-lg font-black uppercase tracking-widest ${gradientTextClass}`}>
              Nhanh chóng
            </p>
          </div>

          {/* Feature 2 */}
          <div className="group text-center transition-transform duration-300 hover:scale-110 cursor-default">
            <div className={`mx-auto w-16 h-16 rounded-full flex items-center justify-center mb-4 shadow-md group-hover:border-purple-500 border border-transparent ${smoothTransition}
              bg-white dark:bg-[#1e293b] dark:border-slate-700
            `}>
                <svg className={`w-8 h-8 ${smoothTransition} text-purple-600 dark:text-purple-400`} fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 21a4 4 0 01-4-4V5a2 2 0 012-2h4a2 2 0 012 2v12a4 4 0 01-4 4zm0 0h12a2 2 0 002-2v-4a2 2 0 00-2-2h-2.343M11 7.343l1.657-1.657a2 2 0 012.828 0l2.829 2.829a2 2 0 010 2.828l-8.486 8.485M7 17h.01" /></svg>
            </div>
            <p className={`text-lg font-black uppercase tracking-widest ${gradientTextClass}`}>
              Sáng tạo
            </p>
          </div>

          {/* Feature 3 */}
          <div className="group text-center transition-transform duration-300 hover:scale-110 cursor-default">
            <div className={`mx-auto w-16 h-16 rounded-full flex items-center justify-center mb-4 shadow-md group-hover:border-pink-500 border border-transparent ${smoothTransition}
              bg-white dark:bg-[#1e293b] dark:border-slate-700
            `}>
                <svg className={`w-8 h-8 ${smoothTransition} text-pink-600 dark:text-pink-400`} fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" /></svg>
            </div>
            <p className={`text-lg font-black uppercase tracking-widest ${gradientTextClass}`}>
              AI Core
            </p>
          </div>

        </div>
      </div>
    </main>
  );
}