import { Link, asset } from '@/lib/link';
export const metadata = {
  title: "QR-AILAB - DiffQRCoder",
  description: "Nền tảng tạo mã QR thông minh bằng AI",
};
export default function XemThem() {
    const galleryExamples = [
    {
        title: "Magical Forest",
        prompt: "A magical forest glowing with fireflies at night",
        image: asset('/qr/examples/forest.png'),  // ✅ Add this
        category: "Fantasy",
        color: "from-green-400 to-emerald-600"
    },
    {
        title: "Neon City",
        prompt: "Futuristic neon city with glowing signs and mist",
        image: asset('/qr/examples/city.png'),    // ✅ Add this
        category: "Sci-Fi",
        color: "from-cyan-400 to-blue-600"
    },
    {
        title: "Vintage Poster",
        prompt: "Retro newspaper illustration with subtle grain texture",
        image: asset('/qr/examples/vintage.png'), // ✅ Add this
        category: "Vintage",
        color: "from-amber-400 to-orange-600"
    },
    {
        title: "Zen Garden",
        prompt: "Clean Japanese zen garden in soft daylight",
        image: asset('/qr/examples/zen.png'),     // ✅ Add this
        category: "Simplicity",
        color: "from-slate-400 to-gray-600"
    }
    ];

  const useCases = [
    {
      icon: "🏢",
      title: "Business",
      items: ["Danh thiếp", "Menu nhà hàng", "Poster"]
    },
    {
      icon: "🎉",
      title: "Sự kiện",
      items: ["Wedding", "Conference", "Tickets"]
    },
    {
      icon: "🎓",
      title: "Giáo dục",
      items: ["Tài liệu", "Bài tập", "Forms"]
    },
    {
      icon: "🛍️",
      title: "E-commerce",
      items: ["Products", "Payments", "Reviews"]
    }
  ];

  const faqs = [
    {
      q: "Mã QR có scan được không?",
      a: "Có! AI đảm bảo QR code vẫn scannable với độ chính xác cao."
    },
    {
      q: "Mất bao lâu để tạo?",
      a: "Thường 30-60 giây tùy độ phức tạp của prompt."
    },
    {
      q: "File output format?",
      a: "PNG độ phân giải cao (1024x1024)."
    },
    {
      q: "Có giới hạn số lượng?",
      a: "Hiện tại miễn phí không giới hạn."
    },
    {
      q: "Làm sao để QR đẹp?",
      a: "Dùng prompt chi tiết, tham khảo Gallery!"
    }
  ];

  return (
    <main
      className="min-h-screen py-12 px-4 sm:px-6 relative overflow-hidden"
      style={{
        background: "linear-gradient(90deg, #C9EBF7, #A0D3E6, #75B6D5, #b57ef5ff, #c29df1ff, #cdb5ecff)",
      }}
    >
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-1/4 right-1/4 w-96 h-96 bg-white/10 rounded-full blur-3xl animate-pulse"></div>
        <div className="absolute bottom-1/3 left-1/3 w-80 h-80 bg-purple-300/10 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '1s' }}></div>
      </div>

      <div className="relative max-w-7xl mx-auto">
        <div className="text-center mb-16">
          <Link href="/" className="inline-block mb-6 transition-transform hover:scale-110">
            <img
              src={asset('/qr/image.png')}
              alt="AI Lab Logo"
              className="h-16 w-auto drop-shadow-lg"
              loading="eager"
            />
          </Link>
          <h1 className="text-4xl sm:text-5xl font-black text-[#013A5A] mb-4">
            Khám phá
            <span className="block text-3xl sm:text-4xl bg-gradient-to-r from-[#013A5A] via-[#0F5F8C] to-[#b57ef5] bg-clip-text text-transparent">
              QR-AILAB
            </span>
          </h1>
          <p className="text-[#0F5F8C] text-lg font-medium max-w-2xl mx-auto">
            Tạo mã QR nghệ thuật với AI – Từ ý tưởng đến hiện thực
          </p>
        </div>

        <section className="mb-20">
          <div className="text-center mb-10">
            <h2 className="text-3xl font-black text-[#013A5A] mb-3">🎨 Gallery</h2>
            <p className="text-[#0F5F8C]">Các mã QR độc đáo được tạo bởi AI</p>
          </div>
          
          <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-6">
            {galleryExamples.map((item, i) => (
              <div
                key={i}
                className="group bg-white/30 backdrop-blur-xl border border-white/40 rounded-2xl overflow-hidden shadow-lg hover:shadow-2xl transition-all duration-300 hover:-translate-y-2"
              >
                <div className="aspect-square bg-gradient-to-br from-gray-100 to-gray-200 flex items-center justify-center overflow-hidden">
  {/* ✅ Replace emoji with real image */}
  <img 
    src={item.image} 
    alt={item.title}
    className="w-full h-full object-cover"
  />
</div>

                <div className="p-5">
                  <div className={`inline-block px-3 py-1 rounded-full text-xs font-bold text-white bg-gradient-to-r ${item.color} mb-3`}>
                    {item.category}
                  </div>
                  <h3 className="font-bold text-[#013A5A] text-lg mb-2">{item.title}</h3>
                  <p className="text-sm text-[#0F5F8C] mb-4 line-clamp-2">{item.prompt}</p>
                  <Link
                    href="/qr/chat"
                    className="block w-full text-center bg-[#013A5A] text-white px-4 py-2 rounded-lg font-semibold hover:bg-[#0F5F8C] transition-colors"
                  >
                    Dùng prompt này
                  </Link>
                </div>
              </div>
            ))}
          </div>
        </section>

        <section className="mb-20">
          <div className="text-center mb-10">
            <h2 className="text-3xl font-black text-[#013A5A] mb-3">💼 Ứng dụng</h2>
            <p className="text-[#0F5F8C]">QR Codes cho mọi nhu cầu</p>
          </div>
          
          <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-6">
            {useCases.map((uc, i) => (
              <div
                key={i}
                className="bg-white/30 backdrop-blur-xl border border-white/40 rounded-2xl p-6 text-center hover:shadow-xl transition-all duration-300 hover:-translate-y-1"
              >
                <div className="text-5xl mb-4">{uc.icon}</div>
                <h3 className="font-bold text-[#013A5A] text-lg mb-4">{uc.title}</h3>
                <ul className="space-y-2 text-sm text-[#0F5F8C]">
                  {uc.items.map((item, j) => (
                    <li key={j} className="flex items-center gap-2 justify-center">
                      <span className="text-green-500">✓</span>
                      {item}
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        </section>

        <section className="mb-20">
          <div className="bg-white/30 backdrop-blur-xl border border-white/40 rounded-3xl p-10 shadow-xl">
            <h2 className="text-3xl font-black text-[#013A5A] text-center mb-10">
              ✨ Tính năng
            </h2>
            <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-8">
              {[
                { emoji: "⚡", title: "Xử lý nhanh", desc: "30-60 giây" },
                { emoji: "🎨", title: "Tùy chỉnh", desc: "Vô số phong cách" },
                { emoji: "🤖", title: "AI thông minh", desc: "DiffQRCoder" },
                { emoji: "🔄", title: "Auto transfer", desc: "Tự động chuyển" },
                { emoji: "🌙", title: "Dark mode", desc: "Bảo vệ mắt" },
                { emoji: "📱", title: "Responsive", desc: "Mọi thiết bị" }
              ].map((f, i) => (
                <div key={i} className="text-center">
                  <div className="text-5xl mb-4">{f.emoji}</div>
                  <h3 className="font-bold text-[#013A5A] mb-2">{f.title}</h3>
                  <p className="text-sm text-[#0F5F8C]">{f.desc}</p>
                </div>
              ))}
            </div>
          </div>
        </section>

        <section className="mb-20">
          <div className="text-center mb-10">
            <h2 className="text-3xl font-black text-[#013A5A] mb-3">❓ FAQs</h2>
            <p className="text-[#0F5F8C]">Câu hỏi thường gặp</p>
          </div>
          
          <div className="max-w-3xl mx-auto space-y-4">
            {faqs.map((faq, i) => (
              <details
                key={i}
                className="bg-white/30 backdrop-blur-xl border border-white/40 rounded-2xl p-6 group hover:shadow-lg transition-all"
              >
                <summary className="font-bold text-[#013A5A] cursor-pointer flex items-center justify-between">
                  <span>{faq.q}</span>
                  <svg className="w-5 h-5 transition-transform group-open:rotate-180" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                  </svg>
                </summary>
                <p className="mt-4 text-[#0F5F8C] leading-relaxed">{faq.a}</p>
              </details>
            ))}
          </div>
        </section>

        <section className="text-center">
          <div className="bg-gradient-to-r from-[#F67A78] via-[#E85D5B] to-[#C63D3D] rounded-3xl p-12 shadow-2xl">
            <h2 className="text-3xl sm:text-4xl font-black text-white mb-4">
              Sẵn sàng tạo mã QR?
            </h2>
            <p className="text-white/90 text-lg mb-8">
              Miễn phí • Không giới hạn • Không cần đăng ký
            </p>
            <Link
              href="/qr/chat"
              className="inline-flex items-center justify-center bg-white text-[#C63D3D] px-10 py-5 rounded-full font-bold text-xl shadow-lg hover:shadow-2xl transition-all duration-300 hover:scale-110"
            >
              <span className="flex items-center gap-3">
                Bắt đầu ngay
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M13 7l5 5m0 0l-5 5m5-5H6" />
                </svg>
              </span>
            </Link>
          </div>
          
          <div className="mt-10">
            <Link
              href="/"
              className="inline-flex items-center gap-2 text-[#013A5A] hover:text-[#0F5F8C] font-semibold transition-colors"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
              </svg>
              Quay lại trang chủ
            </Link>
          </div>
        </section>
      </div>
    </main>
  );
}
