# create_icon.py - Creates a simple WhisperJAV icon
from PIL import Image, ImageDraw, ImageFont
import os

def create_icon():
    # Create a simple icon with gradient background
    img = Image.new('RGBA', (256, 256), color=(0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Draw gradient background
    for i in range(256):
        color = int(73 + (109 - 73) * i / 256)
        draw.rectangle([0, i, 256, i+1], fill=(color, 109, 137, 255))
    
    # Add "WJ" text
    try:
        # Try to use Arial Bold
        font = ImageFont.truetype("arialbd.ttf", 120)
    except:
        try:
            # Fallback to Arial
            font = ImageFont.truetype("arial.ttf", 120)
        except:
            # Last resort - default font
            font = ImageFont.load_default()
    
    # Center the text
    text = "WJ"
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = (256 - text_width) // 2
    y = (256 - text_height) // 2 - 10
    
    # Draw text with shadow
    draw.text((x+3, y+3), text, fill=(0, 0, 0, 128), font=font)  # Shadow
    draw.text((x, y), text, fill=(255, 255, 255, 255), font=font)  # Main text
    
    # Save as ICO with multiple sizes
    img.save('whisperjav_icon.ico', format='ICO', 
             sizes=[(256, 256), (128, 128), (64, 64), (32, 32), (16, 16)])
    print("Icon created: whisperjav_icon.ico")

if __name__ == "__main__":
    create_icon()
