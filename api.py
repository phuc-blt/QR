from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import base64
import uuid
import subprocess
import os
import logging
import requests
import glob
import io
import segno
import cv2

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories
OUTPUT_DIR = "outputs"
QRCODE_DIR = "qrcodes"
UPLOAD_DIR = "uploads"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(QRCODE_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Request models
class Base64Request(BaseModel):
    prompt: str
    image_base64: str

class URLRequest(BaseModel):
    url: str

# Helper: Check if image is QR code
def is_qr_image(image_path: str) -> bool:
    """Check if image contains a valid QR code"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            logger.warning(f"Cannot read image: {image_path}")
            return False
        
        qr = cv2.QRCodeDetector()
        data, _, _ = qr.detectAndDecode(img)
        return data != ""
    except Exception as e:
        logger.error(f"Error checking QR: {e}")
        return False

# Endpoints
@app.get("/")
@app.get("/home")
def home():
    """Health check endpoint"""
    return {
        "message": "API hoạt động ✅",
        "endpoints": ["/generate_base64", "/url_to_base64"],
        "status": "ready"
    }

@app.post("/url_to_base64")
async def url_to_base64(req: URLRequest):
    """Download image from URL and convert to base64"""
    try:
        logger.info(f"📥 Downloading image from: {req.url}")
        
        # Download with timeout
        response = requests.get(req.url, timeout=10)
        response.raise_for_status()
        
        image_bytes = response.content
        
        # Verify it's a valid image
        try:
            img = Image.open(io.BytesIO(image_bytes))
            img.verify()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")
        
        image_base64 = "data:image/png;base64," + base64.b64encode(image_bytes).decode()
        logger.info("✅ Image downloaded and encoded")
        
        return {"image_base64": image_base64}
    
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Download failed: {e}")
        raise HTTPException(status_code=400, detail=f"Cannot download image: {str(e)}")
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
@app.post("/generate_base64")
def generate_base64(req: Base64Request):
    saved_user_image = None
    new_qr_temp = None
    qr_input = None
    output_path = None  # ⭐ THÊM DÒNG NÀY

    try:
        logger.info("🎨 Starting QR generation...")
        
        if not req.prompt or not req.prompt.strip():
            raise HTTPException(status_code=400, detail="Prompt cannot be empty")
        
        logger.info(f"Prompt: {req.prompt[:100]}...")
        
        if not req.image_base64:
            raise HTTPException(status_code=400, detail="Missing image_base64")

        encoded = req.image_base64.split(",", 1)[-1]
        
        try:
            image_bytes = base64.b64decode(encoded)
        except Exception as e:
            logger.error(f"❌ Base64 decode failed: {e}")
            raise HTTPException(status_code=400, detail="Invalid base64 format")

        saved_user_image = f"{UPLOAD_DIR}/{uuid.uuid4()}.png"
        with open(saved_user_image, "wb") as f:
            f.write(image_bytes)
        logger.info(f"💾 Saved image: {saved_user_image}")

        # Resize if needed
        try:
            img = Image.open(saved_user_image)
            original_size = (img.width, img.height)
            
            if img.width < 256 or img.height < 256:
                img = img.resize((256, 256))
                img.save(saved_user_image)
                logger.info(f"📐 Resized from {original_size} to (256, 256)")
        except Exception as e:
            logger.warning(f"⚠️ Resize failed: {e}")

        # Check if image is QR code
        is_qr = False
        try:
            is_qr = is_qr_image(saved_user_image)
        except Exception as e:
            logger.warning(f"⚠️ QR detection failed: {e}")
            is_qr = False

        if not is_qr:
            logger.info("❌ Not a QR code - generating QR from image URL")
            
            try:
                image_url = f"{BASE_URL}/uploads/{os.path.basename(saved_user_image)}"
                
                new_qr_temp = f"{QRCODE_DIR}/{uuid.uuid4()}.png"
                qrcode_obj = segno.make_qr(image_url)
                qrcode_obj.save(new_qr_temp, scale=20)
                logger.info(f"📱 Generated QR: {new_qr_temp}")
                
                # Resize QR if needed
                img = Image.open(new_qr_temp)
                if img.width < 256 or img.height < 256:
                    img = img.resize((256, 256))
                    img.save(new_qr_temp)
                
                qr_input = new_qr_temp
            except Exception as e:
                logger.error(f"❌ Failed to generate QR code: {e}")
                raise HTTPException(status_code=500, detail=f"QR generation failed: {str(e)}")
        else:
            logger.info("✅ Valid QR code detected")
            qr_input = saved_user_image

        # Validate QR input
        if not qr_input or not os.path.exists(qr_input):
            logger.error(f"❌ Invalid qr_input: {qr_input}")
            raise HTTPException(status_code=500, detail="Failed to prepare QR code input")

        # Check if run_diffqrcoder.py exists
        if not os.path.exists("run_diffqrcoder.py"):
            raise HTTPException(status_code=500, detail="run_diffqrcoder.py not found")

        # ⭐ TẠO OUTPUT PATH - PHẢI Ở ĐÂY, TRƯỚC KHI DÙNG
        output_filename = f"qr_{uuid.uuid4()}.png"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        logger.info("▶ Running DiffQRCoder...")
        logger.info(f"  QR Input: {qr_input}")
        logger.info(f"  Output: {output_path}")
        
        cmd = [
            "python", "-u", "run_diffqrcoder.py",
            "--prompt", req.prompt,
            "--qrcode_path", qr_input,
            "--output_path", output_path,
        ]

        # Log command for debugging
        logger.info(f"Command: {' '.join(cmd)}")

        env = os.environ.copy()
        cuda_device = os.environ.get("CUDA_VISIBLE_DEVICES", "2")
        env["CUDA_VISIBLE_DEVICES"] = cuda_device
        logger.info(f"Using CUDA device: {cuda_device}")

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            env=env
        )

        # Print realtime output
        for line in process.stdout:
            print(line, end="")

        process.wait()

        if process.returncode != 0:
            logger.error(f"❌ DiffQRCoder failed with return code: {process.returncode}")
            raise HTTPException(status_code=500, detail="DiffQRCoder failed")

        logger.info("✅ DiffQRCoder completed!")

        # ⭐ BÂY GIỜ MỚI CHECK output_path (SAU KHI ĐÃ DEFINE)
        if not os.path.exists(output_path):
            logger.error(f"❌ Output file not found: {output_path}")
            raise HTTPException(status_code=500, detail="No output file generated")

        logger.info(f"📸 Output file: {output_path}")
        logger.info(f"📏 File size: {os.path.getsize(output_path)} bytes")

        # Read and encode
        with open(output_path, "rb") as f:
            output_bytes = f.read()
            output_base64 = "data:image/png;base64," + base64.b64encode(output_bytes).decode()

        # Log response
        logger.info(f"📦 Response size: {len(output_base64)} characters")
        logger.info(f"📦 Base64 preview: {output_base64[:100]}...")

        # Delete output file after reading
        try:
            os.remove(output_path)
            logger.info(f"🗑️ Deleted output: {output_path}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to delete output: {e}")

        logger.info("✅ QR generation completed successfully!")
        
        response_data = {"output_base64": output_base64}
        logger.info(f"📤 Returning response")
        
        return response_data

    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup temporary files
        for path in [saved_user_image, new_qr_temp]:
            try:
                if path and os.path.exists(path):
                    os.remove(path)
                    logger.debug(f"🗑️ Deleted temp: {path}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to delete {path}: {e}")


# Run with: python -m uvicorn api:app --host 0.0.0.0 --port 8600 --workers 1
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8600)
