#!/bin/bash

python run_diffqrcoder.py \
    --prompt "A beautiful peach blossom tree in full bloom, pink petals falling, spring scenery, soft sunlight, detailed and realistic" \
    --qrcode_path "qrcodes/thanks_reviewer.png" \
    --qrcode_module_size 20 \
    --controlnet_conditioning_scale 1.05 \
    --scanning_robust_guidance_scale 50 \
    --perceptual_guidance_scale 20 \
    --output_path "outputs/demo2_pycon.png"
