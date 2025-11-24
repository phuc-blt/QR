import os
from pathlib import Path
from argparse import ArgumentParser, Namespace

import torch
from diffusers import ControlNetModel, DDIMScheduler
from diffusers.utils import load_image

from diffqrcoder import DiffQRCoderPipeline
import qrcode


def generate_qr_from_text(text: str, output_path: str = "qrcodes/generated_qr.png"):
    """Sinh mã QR từ text hoặc URL"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=10,
        border=4,
    )
    qr.add_data(text)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    img.save(output_path)
    return output_path


def parse_arguments() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--controlnet_ckpt",
        type=str,
        default="monster-labs/control_v1p_sd15_qrcode_monster" ## MODEL 1
    )
    parser.add_argument(
        "--pipe_ckpt",
        type=str,
        default="https://huggingface.co/fp16-guy/Cetus-Mix_Whalefall_fp16_cleaned/blob/main/cetusMix_Whalefall2_fp16.safetensors"     #MODEL 2 
    )
    parser.add_argument(
        "--qrcode_path",
        type=str,
        default="qrcodes/thanks_reviewer.png"
    )
    parser.add_argument(
        "--qrcode_module_size",
        type=int,
        default=20,
    )
    
    parser.add_argument(
        "--qrcode_padding",
        type=int,
        default=78,
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=80,   #default 40, lên 80 để có chất lượng tốt hơn nhưng chậm hơn
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Winter wonderland, fresh snowfall, evergreen trees, cozy log cabin, smoke rising from chimney, aurora borealis in night sky.",
    )
    parser.add_argument(
        "--neg_prompt",
        type=str,
        default="easynegative"
    )
    parser.add_argument(
        "--controlnet_conditioning_scale",
        type=float,
        default=1.25,
    )
    parser.add_argument(
        "-srg",
        "--scanning_robust_guidance_scale",
        type=float,
        default=100, # giảm từ 500 default xuống 50 để chạy nhanh hơn
    )
    parser.add_argument(
        "-pg",
        "--perceptual_guidance_scale",
        type=float,
        default=3.1,
    )
    parser.add_argument(
        "--srmpgd_num_iteration",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--srmpgd_lr",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="results/pycon.png",
    ) 
    return parser.parse_args()                      #trả về toàn bộ args để sử dụng







if __name__ == "__main__":
    
    args = parse_arguments()
    
    
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)  #tạo thư mục nếu chưa tồn tại
    
    
    if args.qrcode_path == "auto":
        if not args.qr_text:
            args.qr_text = input("Enter text or URL to generate QR code: ")
        args.qrcode_path = generate_qr_from_text(args.qr_text)
        print(f"✅ QR code generated: {args.qrcode_path}")

    qrcode_img = load_image(args.qrcode_path)


    qrcode = load_image(args.qrcode_path)
    controlnet = ControlNetModel.from_pretrained(
        args.controlnet_ckpt,                           #checkpoint of controlnet
        torch_dtype=torch.bfloat16,                     #dfault float 32, change to bfloat16 or float16 for less memory usage
        attn_implementation="flash_attention_2",        # THÊM FLASH ATTENTION 2
    )
    pipe = DiffQRCoderPipeline.from_single_file(
        args.pipe_ckpt,
        controlnet=controlnet,
        torch_dtype=torch.bfloat16,                      #dfault float 32, change to bfloat16 or float16 for less memory usage
        attn_implementation="flash_attention_2",         #THÊM FLASH ATTENTION 2
        # use_auth_token=True,                           # Đã cũ và không cần thiết cho link public này nên bỏ ra
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(args.device)
    
    
    
    ## gradient checkpointing (GC) chủ yếu dùng cho TRAINING
    ## Nó không có tác dụng (hoặc rất ít) khi inference và có thể làm chậm đi
    # pipe.unet.enable_gradient_checkpointing()
    # pipe.controlnet.enable_gradient_checkpointing()
    
    ## Giữ VAE slicing 
    pipe.enable_vae_slicing()
    
    ## XÓA attention_slicing, FlashAttention 2 thay vào ok hơn
    # pipe.enable_attention_slicing()
    
    
    #cpu offloading 
    ##pipe.enable_sequential_cpu_offload()

    
    result = pipe(
        prompt=args.prompt,
        qrcode=qrcode,
        qrcode_module_size=args.qrcode_module_size,
        qrcode_padding=args.qrcode_padding,
        negative_prompt=args.neg_prompt,
        num_inference_steps=args.num_inference_steps,
        generator=torch.Generator(device=args.device), # <-- XÓA manual_seed(1) để có kết quả ngẫu nhiên
        controlnet_conditioning_scale=args.controlnet_conditioning_scale,
        scanning_robust_guidance_scale=args.scanning_robust_guidance_scale,
        perceptual_guidance_scale=args.perceptual_guidance_scale,
        srmpgd_num_iteration=args.srmpgd_num_iteration,
        srmpgd_lr=args.srmpgd_lr,
    )
    result.images[0].save(args.output_path)
