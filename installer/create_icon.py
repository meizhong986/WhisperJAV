# create_icon.py - Creates a multi-size WhisperJAV icon from existing 256x256 icon
from PIL import Image
import os
import sys

def create_multi_size_icon():
    """
    Creates a multi-size ICO file from the hardcoded 256x256 icon
    """
    input_icon_path = 'whisperjav_icon_big.ico'
    output_path = 'whisperjav_icon.ico'  # Original output filename
    
    try:
        # Check if input file exists
        if not os.path.exists(input_icon_path):
            print(f"Error: Input file not found: {input_icon_path}")
            print("Please make sure 'whisperjav_icon_big.ico' is in the same folder as this script.")
            sys.exit(1)
        
        # Open the source icon
        with Image.open(input_icon_path) as img:
            # Convert to RGBA to ensure transparency support
            img = img.convert('RGBA')
            
            # Verify the source is 256x256 or larger
            if img.size[0] < 256 or img.size[1] < 256:
                print(f"Warning: Source image is smaller than 256x256. Actual size: {img.size}")
                # We'll still proceed, but quality may suffer
                
            # Extract or create the 256x256 version
            if img.size != (256, 256):
                base_icon = img.resize((256, 256), Image.Resampling.LANCZOS)
            else:
                base_icon = img.copy()
            
            # Create all required sizes (same as original)
            sizes = [(256, 256), (128, 128), (64, 64), (32, 32), (16, 16)]
            icons = []
            
            for size in sizes:
                if size == (256, 256):
                    # Use the base 256x256 directly
                    icons.append(base_icon)
                else:
                    # Resize for other sizes using high-quality resampling
                    resized_icon = base_icon.resize(size, Image.Resampling.LANCZOS)
                    icons.append(resized_icon)
            
            # Save as multi-size ICO with the original output filename
            icons[0].save(
                output_path, 
                format='ICO',
                append_images=icons[1:],  # Include all other sizes
                sizes=sizes
            )
            
            print(f"Multi-size icon created: {output_path}")
            print(f"Source: {input_icon_path}")
            print(f"Sizes included: {', '.join(f'{w}x{h}' for w, h in sizes)}")
            
    except Exception as e:
        print(f"Error processing icon: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    create_multi_size_icon()