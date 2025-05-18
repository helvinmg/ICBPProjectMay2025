import os
import numpy as np
from PIL import Image
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

def upscale_image(input_path, output_dir, model_path=None, target_size=(3510, 4965)):
    """
    Upscales an image toward A1 @150 DPI size using Real-ESRGAN, then resizes to exact target.

    Args:
        input_path (str): Path to input image.
        output_dir (str): Directory to save upscaled image.
        model_path (str): Path to model weights (x4 by default).
        target_size (tuple): Final (width, height) in pixels. Default is A1 @150 DPI.

    Returns:
        str or None: Path to saved image or None on error.
    """
    # Set default model path
    if model_path is None:
        base_dir = os.path.dirname(__file__)
        model_path = os.path.join(base_dir, "Real-ESRGAN", "weights", "RealESRGAN_x4plus.pth")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found at: {model_path}")

    os.makedirs(output_dir, exist_ok=True)

    # Setup Real-ESRGAN model
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                    num_block=23, num_grow_ch=32, scale=4)

    upsampler = RealESRGANer(
        scale=4,
        model_path=model_path,
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=torch.cuda.is_available()
    )

    try:
        img = Image.open(input_path).convert("RGB")
        img_np = np.array(img)

        # First upscale using Real-ESRGAN
        output_np, _ = upsampler.enhance(img_np, outscale=4)
        upscaled_img = Image.fromarray(output_np)

        # Resize precisely to A1 @150 DPI
        final_img = upscaled_img.resize(target_size, Image.LANCZOS)

        # Save output
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_filename = f"{base_name}_upscaled_A1_150dpi.png"
        output_path = os.path.join(output_dir, output_filename)
        final_img.save(output_path, dpi=(150, 150))

        return output_path

    except Exception as e:
        print(f"[Real-ESRGAN Wrapper] Error: {e}")
        return None
