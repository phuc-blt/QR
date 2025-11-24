"use client";
import { useState, useEffect } from "react";

/* Icons */
const ImagePlus = ({ size = 18, className = "", ...props }: any) => (
  <svg viewBox="0 0 24 24" width={size} height={size} className={className} fill="none" {...props}>
    <rect x="3" y="3" width="18" height="18" rx="3" stroke="currentColor" strokeWidth="1.5" />
    <path d="M12 8v8M8 12h8" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
  </svg>
);


const SendHorizontal = ({ size = 18, className = "", ...props }: any) => (
  <svg viewBox="0 0 24 24" width={size} height={size} className={className} fill="none" {...props}>
    <path d="M2 12h18" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
    <path d="M12 5l7 7-7 7" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
  </svg>
);


const LinkIcon = ({ size = 18, className = "", ...props }: any) => (
  <svg viewBox="0 0 24 24" width={size} height={size} className={className} fill="none" {...props}>
    <path d="M10.5 13.5a3.5 3.5 0 010-4.95l2.1-2.1a3.5 3.5 0 014.95 4.95l-1.4 1.4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
    <path d="M13.5 10.5a3.5 3.5 0 010 4.95l-2.1 2.1a3.5 3.5 0 01-4.95-4.95l1.4-1.4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
  </svg>
);


const Sparkles = ({ size = 18, className = "", ...props }: any) => (
  <svg viewBox="0 0 24 24" width={size} height={size} className={className} fill="none" {...props}>
    <path d="M12 3v3m0 12v3m9-9h-3M6 12H3m15.364-6.364l-2.121 2.121M8.757 15.243l-2.121 2.121m12.728 0l-2.121-2.121M8.757 8.757L6.636 6.636" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
  </svg>
);


const QrCode = ({ size = 18, className = "", ...props }: any) => (
  <svg viewBox="0 0 24 24" width={size} height={size} className={className} fill="none" {...props}>
    <rect x="3" y="3" width="7" height="7" rx="1" stroke="currentColor" strokeWidth="1.5" />
    <rect x="14" y="3" width="7" height="7" rx="1" stroke="currentColor" strokeWidth="1.5" />
    <rect x="3" y="14" width="7" height="7" rx="1" stroke="currentColor" strokeWidth="1.5" />
    <path d="M14 14h2m0 0h2m-2 0v2m0-2v-2M14 19h7M19 14v7" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
  </svg>
);


const Moon = ({ size = 20, className = "" }: any) => (
  <svg viewBox="0 0 24 24" width={size} height={size} className={className} fill="none" stroke="currentColor">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12.79A9 9 0 1111.21 3 7 7 0 0021 12.79z" />
  </svg>
);


const Sun = ({ size = 20, className = "" }: any) => (
  <svg viewBox="0 0 24 24" width={size} height={size} className={className} fill="none" stroke="currentColor">
    <circle cx="12" cy="12" r="5" strokeWidth={2} />
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 1v2m0 18v2M4.22 4.22l1.42 1.42m12.72 12.72l1.42 1.42M1 12h2m18 0h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42" />
  </svg>
);


const ArrowRight = ({ size = 18, className = "" }: any) => (
  <svg viewBox="0 0 24 24" width={size} height={size} className={className} fill="none" stroke="currentColor">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
  </svg>
);

export default function ChatPage() {
  const [darkMode, setDarkMode] = useState(false);
  
  // AI QR states
  const [prompt, setPrompt] = useState("");
  const [image, setImage] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [qrResult, setQrResult] = useState<string | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [imageUrl, setImageUrl] = useState("");


  // Simple QR states
  const [simpleQrText, setSimpleQrText] = useState("");
  const [simpleQrType, setSimpleQrType] = useState("url");
  const [simpleQrResult, setSimpleQrResult] = useState<string | null>(null);
  const [simpleQrLoading, setSimpleQrLoading] = useState(false);

  // ⭐ NEW: State để hiển thị thông báo transfer
  const [showTransferNotice, setShowTransferNotice] = useState(false);


  const PROMPT_CATEGORIES = {
    cosy: [
      "A warm wooden cabin in snowy forest with soft lighting",
      "Sunset view from a cozy coffee shop window",
    ],
    fiction: [
      "A steampunk airship flying above the clouds",
      "Futuristic neon city with glowing signs and mist",
    ],
    fantasy: [
      "Winter wonderland, fresh snowfall, evergreen trees, cozy log cabin, smoke rising from chimney, aurora borealis in night sky",
      "Photorealistic mountain, high view, sunset, moving clouds, scenery.",
    ],
    simplicity: [
      "Minimal black and white geometric pattern",
      "Clean Japanese zen garden in soft daylight",
    ],
    vintage: [
      "Retro newspaper illustration with subtle grain texture",
      "Classic oil painting style portrait of a cat in a suit",
    ],
  };


  useEffect(() => {
    document.body.classList.toggle('dark-mode', darkMode);
  }, [darkMode]);


  const handleUpload = (e: any) => {
    const f = e.target.files?.[0];
    if (!f) return;
    setImage(f);
    setImageUrl("");
    setPreviewUrl(URL.createObjectURL(f));
    setErrorMsg(null);
  };


  const handleImageUrl = () => {
    if (!imageUrl.trim()) return setErrorMsg("Vui lòng nhập URL hình hợp lệ.");
    setImage(null);
    setPreviewUrl(imageUrl);
    setErrorMsg(null);
  };


  const encodeImageToBase64 = (file: File): Promise<string> =>
    new Promise((resolve, reject) => {
      const r = new FileReader();
      r.onloadend = () => resolve(r.result as string);
      r.onerror = reject;
      r.readAsDataURL(file);
    });


const handleGenerate = async () => {
  if (!prompt.trim()) return setErrorMsg("Vui lòng nhập prompt.");
  if (!image && !previewUrl) return setErrorMsg("Vui lòng tải ảnh hoặc URL.");

  setLoading(true);
  setQrResult(null);
  setErrorMsg(null);

  try {
    let base64Image = "";

    if (image) {
      base64Image = await encodeImageToBase64(image);
    } else {
      const res = await fetch("https://b400b1cb73b1.ngrok-free.app/url_to_base64", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url: previewUrl }),
      });
      base64Image = (await res.json()).image_base64;
    }

    // ⭐ BỎ AbortController và timeout
    const r = await fetch("https://b400b1cb73b1.ngrok-free.app/generate_base64", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt, image_base64: base64Image }),
      // ← KHÔNG CÓ signal
    });

    if (!r.ok) {
      const errorData = await r.json();
      throw new Error(errorData.detail || "API error");
    }

    const data = await r.json();
    setQrResult(data.output_base64 || null);
  } catch (error: any) {
    console.error("Generate error:", error);
    setErrorMsg(error.message || "Không thể kết nối API.");
  }
  setLoading(false);
};



  const handleGenerateSimpleQR = async () => {
    if (!simpleQrText.trim()) {
      setErrorMsg("Vui lòng nhập nội dung cho mã QR.");
      return;
    }


    setSimpleQrLoading(true);
    setSimpleQrResult(null);
    setErrorMsg(null);


    try {
      const qrApiUrl = `https://api.qrserver.com/v1/create-qr-code/?size=400x400&data=${encodeURIComponent(simpleQrText)}&color=000000&bgcolor=ffffff`;
      setSimpleQrResult(qrApiUrl);

      // ⭐ TỰ ĐỘNG CHUYỂN MÃ QR SANG AI PANEL
      setTimeout(() => {
        setImage(null);
        setImageUrl("");
        setPreviewUrl(qrApiUrl);
        
        // Hiển thị thông báo
        setShowTransferNotice(true);
        setTimeout(() => setShowTransferNotice(false), 3000);
      }, 500);

    } catch {
      setErrorMsg("Không thể tạo mã QR.");
    }
    setSimpleQrLoading(false);
  };


  return (
    <div className="app-container">
      {/* Dark Mode Toggle */}
      <button 
        onClick={() => setDarkMode(!darkMode)} 
        className="dark-mode-toggle"
        aria-label="Toggle dark mode"
      >
        {darkMode ? <Sun size={22} /> : <Moon size={22} />}
      </button>


      {/* Background */}
      <div className="bg-layer"></div>


      {/* Floating Orbs */}
      <div className="orbs-container">
        <div className="floating-orb orb-1"></div>
        <div className="floating-orb orb-2"></div>
        <div className="floating-orb orb-3"></div>
      </div>

      {/* ⭐ TRANSFER NOTIFICATION */}
      {showTransferNotice && (
        <div className="transfer-notification">
          <ArrowRight size={20} />
          <span>Mã QR đã được chuyển sang AI Generator!</span>
        </div>
      )}


      {/* Main Container - 3 Column Layout */}
      <div className="main-grid">
        
        {/* LEFT PANEL - AI QR Generator */}
        <div className="glass-card">
          <div className="header-section">
            <div className="icon-wrapper">
              <Sparkles size={28} className="icon-primary" />
            </div>
            <h2 className="section-title">AI QR Generator</h2>
            <p className="section-subtitle">Tạo mã QR độc đáo với AI</p>
          </div>


          <div className="card-content">
            <label className="upload-area">
              <ImagePlus className="upload-icon" />
              <span className="upload-text">Tải hình ảnh từ máy</span>
              <span className="upload-hint">PNG, JPG up to 10MB</span>
              <input type="file" accept="image/*" className="hidden" onChange={handleUpload} />
            </label>


            <div className="url-input-group">
              <input
                className="modern-input"
                placeholder="Hoặc dán URL hình ảnh..."
                value={imageUrl}
                onChange={(e) => setImageUrl(e.target.value)}
              />
              <button onClick={handleImageUrl} className="url-btn">
                <LinkIcon size={20} />
              </button>
            </div>


            {previewUrl && (
              <div className="preview-wrapper">
                <img src={previewUrl} alt="Preview" className="preview-image" />
              </div>
            )}


            <div>
              <label className="input-label">Mô tả phong cách</label>
              <textarea
                className="modern-input modern-textarea"
                placeholder="Ví dụ: A magical forest glowing with fireflies..."
                rows={5}
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
              />
            </div>


            <button onClick={handleGenerate} className="generate-btn" disabled={loading}>
              {loading ? (
                <>
                  <div className="btn-spinner"></div>
                  <span>Đang xử lý...</span>
                </>
              ) : (
                <>
                  <SendHorizontal size={20} />
                  <span>Tạo mã QR với AI</span>
                </>
              )}
            </button>


            {errorMsg && (
              <div className="error-message">
                <span className="error-icon">⚠️</span>
                {errorMsg}
              </div>
            )}


            {loading && (
              <div className="loading-state">
                <div className="loading-spinner"></div>
                <p className="loading-text">Đang xử lý hình ảnh của bạn...</p>
                <p className="loading-subtext">
                  ⏱️ Có thể mất 30-60 giây
                </p>
              </div>
            )}



            {qrResult && (
              <div className="result-card">
                <h3 className="result-title">Mã QR của bạn</h3>
                <img src={qrResult} alt="QR Result" className="result-image" />
                <button
                  onClick={() => {
                    const a = document.createElement("a");
                    a.href = qrResult;
                    a.download = `AI-QR-${Date.now()}.png`;
                    a.click();
                  }}
                  className="download-btn"
                >
                  Tải xuống QR Code
                </button>
              </div>
            )}
          </div>
        </div>


        {/* MIDDLE PANEL - Simple QR Generator */}
        <div className="glass-card">
          <div className="header-section">
            <div className="icon-wrapper">
              <QrCode size={28} className="icon-primary" />
            </div>
            <h2 className="section-title">QR Đơn Giản</h2>
            <p className="section-subtitle">Tạo QR trắng đen cơ bản</p>
          </div>


          <div className="card-content">
            <div>
              <label className="input-label">Loại nội dung</label>
              <select 
                className="modern-input" 
                value={simpleQrType} 
                onChange={(e) => setSimpleQrType(e.target.value)}
              >
                <option value="url">🌐 URL / Website</option>
                <option value="text">📝 Văn bản</option>
                <option value="email">✉️ Email</option>
                <option value="phone">📱 Số điện thoại</option>
              </select>
            </div>


            <div>
              <label className="input-label">
                {simpleQrType === "url" && "Nhập URL"}
                {simpleQrType === "text" && "Nhập văn bản"}
                {simpleQrType === "email" && "Nhập email"}
                {simpleQrType === "phone" && "Nhập số điện thoại"}
              </label>
              <textarea
                className="modern-input modern-textarea"
                placeholder={
                  simpleQrType === "url" ? "https://example.com" :
                  simpleQrType === "text" ? "Nội dung của bạn..." :
                  simpleQrType === "email" ? "email@example.com" :
                  "+84 123 456 789"
                }
                rows={4}
                value={simpleQrText}
                onChange={(e) => setSimpleQrText(e.target.value)}
              />
            </div>


            {/* Ví dụ nhanh */}
            <div className="examples-section">
              <p className="examples-title">Ví dụ nhanh:</p>
              <div className="examples-grid">
                {[
                  { label: "Website", value: "https://ailab.com", type: "url" },
                  { label: "Email", value: "contact@ailab.com", type: "email" },
                  { label: "Phone", value: "+84 912 345 678", type: "phone" },
                  { label: "Text", value: "Phúc đẹp trai", type: "text" },
                ].map((example) => (
                  <button
                    key={example.label}
                    onClick={() => {
                      setSimpleQrText(example.value);
                      setSimpleQrType(example.type);
                      setErrorMsg(null);
                    }}
                    className="example-btn"
                  >
                    {example.label}
                  </button>
                ))}
              </div>
            </div>


            <button 
              onClick={handleGenerateSimpleQR} 
              className="generate-btn" 
              disabled={simpleQrLoading}
            >
              {simpleQrLoading ? (
                <>
                  <div className="btn-spinner"></div>
                  <span>Đang tạo...</span>
                </>
              ) : (
                <>
                  <QrCode size={20} />
                  <span>Tạo mã QR</span>
                </>
              )}
            </button>


            {simpleQrLoading && (
              <div className="loading-state">
                <div className="loading-spinner"></div>
                <p className="loading-text">Đang tạo mã QR...</p>
              </div>
            )}


            {simpleQrResult && (
              <div className="result-card">
                <h3 className="result-title">Mã QR của bạn</h3>
                <img src={simpleQrResult} alt="Simple QR" className="result-image" />
                <button
                  onClick={() => {
                    const a = document.createElement("a");
                    a.href = simpleQrResult;
                    a.download = `Simple-QR-${Date.now()}.png`;
                    a.click();
                  }}
                  className="download-btn"
                >
                  Tải xuống QR Code
                </button>
              </div>
            )}
          </div>
        </div>


        {/* RIGHT PANEL - Prompt Suggestions */}
        <div className="glass-card scrollable">
          <div className="header-section">
            <div className="icon-wrapper">
              <Sparkles size={28} className="icon-primary" />
            </div>
            <h2 className="section-title">Gợi Ý Phong Cách</h2>
            <p className="section-subtitle">Chọn hoặc tùy chỉnh theo ý bạn</p>
          </div>


          <div className="card-content">
            {Object.entries(PROMPT_CATEGORIES).map(([category, prompts]) => (
              <div key={category} className="category-section">
                <h3 className="category-header">{category}</h3>
                <div className="prompts-list">
                  {prompts.map((text, i) => (
                    <button
                      key={i}
                      className="prompt-card"
                      onClick={() => setPrompt(text)}
                    >
                      <span className="prompt-emoji">✨</span>
                      <span className="prompt-content">{text}</span>
                    </button>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>


      {/* CSS Styles */}
      <style>{`
      /* Reorder panels */
      .main-grid > .glass-card:nth-child(1) {
        order: 2; /* AI QR Generator ở giữa */
      }

      .main-grid > .glass-card:nth-child(2) {
        order: 1; /* QR Đơn Giản ở trái */
      }

      .main-grid > .glass-card:nth-child(3) {
        order: 3; /* Gợi Ý Phong Cách ở phải */
      }

        * {
          box-sizing: border-box;
        }


        body {
          margin: 0;
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica', 'Arial', sans-serif;
          transition: background-color 0.3s ease, color 0.3s ease;
        }


        .hidden {
          display: none;
        }


        /* App Container */
        .app-container {
          position: relative;
          min-height: 100vh;
          padding: 24px;
          overflow: hidden;
        }


        /* Dark Mode Toggle */
        .dark-mode-toggle {
          position: fixed;
          top: 24px;
          right: 24px;
          z-index: 1000;
          width: 56px;
          height: 56px;
          display: flex;
          align-items: center;
          justify-content: center;
          background: rgba(255, 255, 255, 0.9);
          border: 2px solid rgba(226, 232, 240, 0.8);
          border-radius: 50%;
          cursor: pointer;
          transition: all 0.3s ease;
          box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
          color: #475569;
        }


        .dark-mode-toggle:hover {
          transform: scale(1.1) rotate(10deg);
          box-shadow: 0 6px 20px rgba(0, 0, 0, 0.15);
        }

        /* ⭐ TRANSFER NOTIFICATION */
        .transfer-notification {
          position: fixed;
          top: 100px;
          left: 50%;
          transform: translateX(-50%);
          z-index: 2000;
          display: flex;
          align-items: center;
          gap: 12px;
          padding: 16px 28px;
          background: linear-gradient(135deg, #10b981 0%, #059669 100%);
          color: white;
          font-weight: 600;
          font-size: 15px;
          border-radius: 12px;
          box-shadow: 0 8px 24px rgba(16, 185, 129, 0.4);
          animation: slideInDown 0.4s ease, slideOutUp 0.4s ease 2.6s;
        }

        @keyframes slideInDown {
          from {
            opacity: 0;
            transform: translateX(-50%) translateY(-20px);
          }
          to {
            opacity: 1;
            transform: translateX(-50%) translateY(0);
          }
        }

        @keyframes slideOutUp {
          from {
            opacity: 1;
            transform: translateX(-50%) translateY(0);
          }
          to {
            opacity: 0;
            transform: translateX(-50%) translateY(-20px);
          }
        }

        body.dark-mode .transfer-notification {
          background: linear-gradient(135deg, #059669 0%, #047857 100%);
        }


        /* Light Mode Background */
        .bg-layer {
          position: fixed;
          inset: 0;
          background: 
            radial-gradient(circle at 15% 20%, rgba(99, 102, 241, 0.08), transparent 50%),
            radial-gradient(circle at 85% 80%, rgba(59, 130, 246, 0.08), transparent 50%),
            linear-gradient(135deg, #f8fafc 0%, #e2e8f0 50%, #f1f5f9 100%);
          background-attachment: fixed;
          transition: background 0.3s ease;
          z-index: -2;
        }


        /* Dark Mode Styles */
        body.dark-mode .bg-layer {
          background: 
            radial-gradient(circle at 15% 20%, rgba(99, 102, 241, 0.15), transparent 50%),
            radial-gradient(circle at 85% 80%, rgba(139, 92, 246, 0.15), transparent 50%),
            linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
        }


        body.dark-mode .dark-mode-toggle {
          background: rgba(30, 41, 59, 0.9);
          border-color: rgba(71, 85, 105, 0.8);
          color: #f1f5f9;
        }


        body.dark-mode .glass-card {
          background: rgba(30, 41, 59, 0.8);
          border-color: rgba(71, 85, 105, 0.4);
        }


        body.dark-mode .section-title {
          color: #f1f5f9;
        }


        body.dark-mode .section-subtitle {
          color: #94a3b8;
        }


        body.dark-mode .modern-input {
          background: rgba(15, 23, 42, 0.6);
          border-color: rgba(71, 85, 105, 0.6);
          color: #f1f5f9;
        }


        body.dark-mode .modern-input::placeholder {
          color: #64748b;
        }


        body.dark-mode .modern-input:focus {
          border-color: #818cf8;
          background: rgba(15, 23, 42, 0.8);
        }


        body.dark-mode .upload-area {
          background: linear-gradient(135deg, rgba(30, 41, 59, 0.6) 0%, rgba(51, 65, 85, 0.6) 100%);
          border-color: rgba(71, 85, 105, 0.6);
        }


        body.dark-mode .upload-text {
          color: #e2e8f0;
        }


        body.dark-mode .upload-hint {
          color: #94a3b8;
        }


        body.dark-mode .input-label {
          color: #cbd5e1;
        }


        body.dark-mode .category-section {
          background: rgba(15, 23, 42, 0.4);
          border-color: rgba(71, 85, 105, 0.4);
        }


        body.dark-mode .category-header {
          color: #94a3b8;
        }


        body.dark-mode .prompt-card {
          background: rgba(30, 41, 59, 0.6);
          border-color: rgba(71, 85, 105, 0.4);
        }


        body.dark-mode .prompt-card:hover {
          background: rgba(51, 65, 85, 0.6);
          border-color: rgba(100, 116, 139, 0.6);
        }


        body.dark-mode .prompt-content {
          color: #cbd5e1;
        }


        body.dark-mode .preview-wrapper {
          background: rgba(15, 23, 42, 0.6);
          border-color: rgba(71, 85, 105, 0.6);
        }


        body.dark-mode .example-btn {
          background: rgba(30, 41, 59, 0.6);
          border-color: rgba(71, 85, 105, 0.4);
          color: #cbd5e1;
        }


        body.dark-mode .example-btn:hover {
          background: rgba(51, 65, 85, 0.6);
          border-color: rgba(100, 116, 139, 0.6);
        }


        body.dark-mode .examples-title {
          color: #cbd5e1;
        }


        /* Floating Orbs */
        .orbs-container {
          position: fixed;
          inset: 0;
          pointer-events: none;
          overflow: hidden;
          z-index: -1;
        }


        .floating-orb {
          position: absolute;
          border-radius: 50%;
          filter: blur(60px);
          opacity: 0.3;
          animation: float-gentle 20s ease-in-out infinite;
          transition: opacity 0.3s ease;
        }


        body.dark-mode .floating-orb {
          opacity: 0.2;
        }


        .orb-1 {
          width: 400px;
          height: 400px;
          background: linear-gradient(135deg, #60a5fa, #818cf8);
          top: -100px;
          left: -100px;
        }


        .orb-2 {
          width: 300px;
          height: 300px;
          background: linear-gradient(135deg, #a78bfa, #c084fc);
          bottom: -50px;
          right: -50px;
          animation-delay: -10s;
        }


        .orb-3 {
          width: 250px;
          height: 250px;
          background: linear-gradient(135deg, #7dd3fc, #93c5fd);
          top: 50%;
          right: 30%;
          animation-delay: -15s;
        }


        @keyframes float-gentle {
          0%, 100% { transform: translate(0, 0) scale(1); }
          33% { transform: translate(30px, -30px) scale(1.1); }
          66% { transform: translate(-20px, 20px) scale(0.9); }
        }


        /* Main Grid */
        .main-grid {
          position: relative;
          z-index: 10;
          max-width: 1800px;
          margin: 0 auto;
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(380px, 1fr));
          gap: 24px;
        }


        @media (min-width: 1400px) {
          .main-grid {
            grid-template-columns: repeat(3, 1fr);
          }
        }


        /* Glass Card */
        .glass-card {
          background: rgba(255, 255, 255, 0.85);
          backdrop-filter: blur(24px);
          -webkit-backdrop-filter: blur(24px);
          border-radius: 28px;
          border: 1px solid rgba(255, 255, 255, 0.6);
          box-shadow: 
            0 20px 60px rgba(0, 0, 0, 0.08),
            0 0 1px rgba(0, 0, 0, 0.05),
            inset 0 1px 0 rgba(255, 255, 255, 0.9);
          transition: all 0.4s ease;
          overflow: hidden;
        }


        .glass-card.scrollable {
          max-height: calc(100vh - 48px);
          overflow-y: auto;
        }


        .glass-card::-webkit-scrollbar {
          width: 8px;
        }


        .glass-card::-webkit-scrollbar-track {
          background: rgba(226, 232, 240, 0.3);
          border-radius: 10px;
        }


        .glass-card::-webkit-scrollbar-thumb {
          background: linear-gradient(180deg, #94a3b8, #64748b);
          border-radius: 10px;
        }


        body.dark-mode .glass-card::-webkit-scrollbar-track {
          background: rgba(51, 65, 85, 0.3);
        }


        body.dark-mode .glass-card::-webkit-scrollbar-thumb {
          background: linear-gradient(180deg, #64748b, #475569);
        }


        /* Card Content */
        .card-content {
          padding: 24px;
          display: flex;
          flex-direction: column;
          gap: 20px;
        }


        /* Header Section */
        .header-section {
          text-align: center;
          padding: 28px 24px 24px;
          border-bottom: 2px solid rgba(148, 163, 184, 0.15);
        }


        body.dark-mode .header-section {
          border-bottom-color: rgba(71, 85, 105, 0.3);
        }


        .icon-wrapper {
          display: inline-flex;
          align-items: center;
          justify-content: center;
          width: 64px;
          height: 64px;
          background: linear-gradient(135deg, #dbeafe 0%, #e0e7ff 100%);
          border-radius: 18px;
          margin-bottom: 16px;
          box-shadow: 0 4px 12px rgba(99, 102, 241, 0.15);
          transition: all 0.3s ease;
        }


        body.dark-mode .icon-wrapper {
          background: linear-gradient(135deg, rgba(99, 102, 241, 0.3) 0%, rgba(129, 140, 248, 0.3) 100%);
        }


        .icon-primary {
          color: #6366f1;
        }


        body.dark-mode .icon-primary {
          color: #a5b4fc;
        }


        .section-title {
          font-size: 26px;
          font-weight: 800;
          color: #1e293b;
          margin: 0 0 8px 0;
          letter-spacing: -0.5px;
        }


        .section-subtitle {
          font-size: 15px;
          color: #64748b;
          font-weight: 500;
          margin: 0;
        }


        /* Upload Area */
        .upload-area {
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          gap: 8px;
          padding: 32px 24px;
          background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
          border: 2px dashed #cbd5e1;
          border-radius: 16px;
          cursor: pointer;
          transition: all 0.3s ease;
        }


        .upload-area:hover {
          background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%);
          border-color: #94a3b8;
          transform: translateY(-2px);
          box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        }


        .upload-icon {
          width: 24px;
          height: 24px;
          color: #64748b;
        }


        .upload-text {
          font-size: 15px;
          font-weight: 600;
          color: #334155;
        }


        .upload-hint {
          font-size: 13px;
          color: #94a3b8;
        }


        /* URL Input Group */
        .url-input-group {
          display: flex;
          gap: 12px;
        }


        /* Modern Input */
        .modern-input {
          width: 100%;
          padding: 14px 18px;
          font-size: 15px;
          color: #1e293b;
          background: #ffffff;
          border: 2px solid #e2e8f0;
          border-radius: 12px;
          outline: none;
          transition: all 0.3s ease;
          font-family: inherit;
        }


        .modern-input:focus {
          border-color: #60a5fa;
          box-shadow: 0 0 0 4px rgba(96, 165, 250, 0.1);
        }


        .modern-input::placeholder {
          color: #94a3b8;
        }


        .modern-textarea {
          resize: none;
          line-height: 1.6;
        }


        /* URL Button */
        .url-btn {
          flex-shrink: 0;
          padding: 14px 18px;
          background: linear-gradient(135deg, #60a5fa 0%, #3b82f6 100%);
          border: none;
          border-radius: 12px;
          color: white;
          cursor: pointer;
          transition: all 0.3s ease;
          box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
        }


        .url-btn:hover {
          transform: translateY(-2px);
          box-shadow: 0 6px 16px rgba(59, 130, 246, 0.4);
        }


        /* Input Label */
        .input-label {
          display: block;
          font-size: 14px;
          font-weight: 600;
          color: #475569;
          margin-bottom: 10px;
        }


        /* ⭐ PREVIEW - CENTERED WITH FLEXBOX */
        .preview-wrapper {
          /* ⭐ Flexbox centering */
          display: flex;
          align-items: center;
          justify-content: center;
          min-height: 260px;
          padding: 20px;
          background: #f8fafc;
          border: 2px solid #e2e8f0;
          border-radius: 16px;
        }

        .preview-image {
          /* ⭐ Responsive image */
          display: block;
          max-width: 100%;
          max-height: 240px;
          width: auto;
          height: auto;
          border-radius: 12px;
          box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
          /* ⭐ Additional centering with margin */
          margin: 0 auto;
        }



        /* Generate Button */
        .generate-btn {
          width: 100%;
          height: 52px;
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 10px;
          font-size: 15px;
          font-weight: 700;
          color: white;
          background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
          border: none;
          border-radius: 12px;
          cursor: pointer;
          transition: all 0.3s ease;
          box-shadow: 0 6px 20px rgba(99, 102, 241, 0.3);
        }


        .generate-btn:hover:not(:disabled) {
          transform: translateY(-2px);
          box-shadow: 0 8px 24px rgba(99, 102, 241, 0.4);
        }


        .generate-btn:disabled {
          opacity: 0.7;
          cursor: not-allowed;
        }


        .btn-spinner {
          width: 20px;
          height: 20px;
          border: 3px solid rgba(255, 255, 255, 0.3);
          border-top-color: white;
          border-radius: 50%;
          animation: spin 0.6s linear infinite;
        }


        @keyframes spin {
          to { transform: rotate(360deg); }
        }


        /* Error Message */
        .error-message {
          display: flex;
          align-items: center;
          gap: 10px;
          padding: 14px 18px;
          background: #fef2f2;
          border: 1px solid #fecaca;
          border-radius: 12px;
          color: #dc2626;
          font-size: 14px;
          font-weight: 500;
        }


        body.dark-mode .error-message {
          background: rgba(239, 68, 68, 0.1);
          border-color: rgba(239, 68, 68, 0.3);
          color: #fca5a5;
        }


        .error-icon {
          font-size: 18px;
        }


        /* Loading State */
        .loading-state {
          text-align: center;
          padding: 32px 20px;
          background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
          border-radius: 16px;
          border: 1px solid #bae6fd;
        }


        body.dark-mode .loading-state {
          background: linear-gradient(135deg, rgba(14, 165, 233, 0.1) 0%, rgba(56, 189, 248, 0.1) 100%);
          border-color: rgba(56, 189, 248, 0.2);
        }


        .loading-spinner {
          width: 48px;
          height: 48px;
          margin: 0 auto 16px;
          border: 4px solid #e0f2fe;
          border-top-color: #0ea5e9;
          border-radius: 50%;
          animation: spin 0.8s linear infinite;
        }


        body.dark-mode .loading-spinner {
          border-color: rgba(56, 189, 248, 0.2);
          border-top-color: #38bdf8;
        }


        .loading-text {
          font-size: 15px;
          font-weight: 600;
          color: #0c4a6e;
          margin: 0 0 4px 0;
        }


        body.dark-mode .loading-text {
          color: #7dd3fc;
        }


        .loading-subtext {
          font-size: 13px;
          color: #0369a1;
          margin: 0;
        }


        body.dark-mode .loading-subtext {
          color: #38bdf8;
        }


        /* Result Card */
        .result-card {
          text-align: center;
          padding: 24px;
          background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
          border: 2px solid #bbf7d0;
          border-radius: 20px;
        }


        body.dark-mode .result-card {
          background: linear-gradient(135deg, rgba(34, 197, 94, 0.1) 0%, rgba(74, 222, 128, 0.1) 100%);
          border-color: rgba(74, 222, 128, 0.2);
        }


        .result-title {
          font-size: 16px;
          font-weight: 700;
          color: #166534;
          margin: 0 0 16px 0;
        }


        body.dark-mode .result-title {
          color: #86efac;
        }


        .result-image {
          width: 100%;
          max-width: 300px;
          margin: 0 auto 20px;
          border-radius: 16px;
          box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12);
          border: 3px solid #86efac;
        }


        body.dark-mode .result-image {
          border-color: rgba(134, 239, 172, 0.3);
        }


        /* Download Button */
        .download-btn {
          padding: 14px 32px;
          font-size: 15px;
          font-weight: 700;
          color: white;
          background: linear-gradient(135deg, #10b981 0%, #059669 100%);
          border: none;
          border-radius: 12px;
          cursor: pointer;
          transition: all 0.3s ease;
          box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
        }


        .download-btn:hover {
          transform: translateY(-2px);
          box-shadow: 0 6px 16px rgba(16, 185, 129, 0.4);
        }


        /* Examples Section */
        .examples-section {
          padding: 16px;
          background: rgba(248, 250, 252, 0.6);
          border-radius: 12px;
          border: 1px solid rgba(226, 232, 240, 0.6);
        }


        body.dark-mode .examples-section {
          background: rgba(15, 23, 42, 0.4);
          border-color: rgba(71, 85, 105, 0.4);
        }


        .examples-title {
          font-size: 13px;
          font-weight: 600;
          color: #475569;
          margin: 0 0 12px 0;
        }


        .examples-grid {
          display: grid;
          grid-template-columns: repeat(2, 1fr);
          gap: 8px;
        }


        .example-btn {
          padding: 10px 14px;
          background: white;
          border: 1px solid #e2e8f0;
          border-radius: 8px;
          font-size: 13px;
          font-weight: 600;
          color: #475569;
          cursor: pointer;
          transition: all 0.3s ease;
        }


        .example-btn:hover {
          background: #f8fafc;
          border-color: #cbd5e1;
          transform: translateY(-2px);
          box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        }


        /* Category Section */
        .category-section {
          padding: 20px;
          background: rgba(248, 250, 252, 0.6);
          border-radius: 16px;
          border: 1px solid rgba(226, 232, 240, 0.6);
        }


        .category-header {
          font-size: 13px;
          font-weight: 700;
          color: #64748b;
          text-transform: uppercase;
          letter-spacing: 1px;
          margin: 0 0 12px 0;
        }


        .prompts-list {
          display: flex;
          flex-direction: column;
          gap: 12px;
        }


        /* Prompt Card */
        .prompt-card {
          width: 100%;
          display: flex;
          align-items: flex-start;
          gap: 12px;
          padding: 14px 16px;
          background: white;
          border: 1px solid #e2e8f0;
          border-radius: 12px;
          text-align: left;
          cursor: pointer;
          transition: all 0.3s ease;
        }


        .prompt-card:hover {
          background: #f8fafc;
          border-color: #cbd5e1;
          transform: translateX(6px);
          box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        }


        .prompt-emoji {
          font-size: 18px;
          flex-shrink: 0;
        }


        .prompt-content {
          color: #475569;
          font-size: 14px;
          line-height: 1.6;
          font-weight: 500;
        }


        /* Responsive */
        
        @media (max-width: 768px) {
          .app-container {
            padding: 16px;
          }


          .dark-mode-toggle {
            top: 16px;
            right: 16px;
            width: 48px;
            height: 48px;
          }

          .transfer-notification {
            font-size: 13px;
            padding: 12px 20px;
          }
        }
      `}</style>
    </div>
  );
}
