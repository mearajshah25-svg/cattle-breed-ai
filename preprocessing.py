"""
preprocessing.py
Image preprocessing for animal classification
"""

import cv2
import numpy as np
from PIL import Image
import os

class ImagePreprocessor:
    def __init__(self, target_size=(640, 480)):
        """
        Initialize preprocessor
        Args:
            target_size: (width, height) for resizing images
        """
        self.target_size = target_size
        
    def load_image(self, image_path):
        """Load image from file"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    
    def resize_image(self, image, maintain_aspect=True):
        """Resize image to target size"""
        if maintain_aspect:
            h, w = image.shape[:2]
            target_w, target_h = self.target_size
            
            # Calculate scaling factor
            scale = min(target_w / w, target_h / h)
            new_w, new_h = int(w * scale), int(h * scale)
            
            # Resize
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # Create canvas and center image
            canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            y_offset = (target_h - new_h) // 2
            x_offset = (target_w - new_w) // 2
            canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
            
            return canvas, scale, (x_offset, y_offset)
        else:
            return cv2.resize(image, self.target_size), 1.0, (0, 0)
    
    def enhance_contrast(self, image):
        """Apply CLAHE for contrast enhancement"""
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge and convert back
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
        
        return enhanced
    
    def remove_background(self, image, method='grabcut'):
        """
        Background removal (simple version)
        In production, use a trained segmentation model
        """
        if method == 'grabcut':
            mask = np.zeros(image.shape[:2], np.uint8)
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)
            
            # Define rectangle around animal (center 80% of image)
            h, w = image.shape[:2]
            rect = (int(w*0.1), int(h*0.1), int(w*0.8), int(h*0.8))
            
            # Apply GrabCut
            cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
            
            # Create mask
            mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
            result = image * mask2[:, :, np.newaxis]
            
            return result, mask2
        
        return image, np.ones(image.shape[:2], dtype=np.uint8)
    
    def denoise(self, image):
        """Remove noise from image"""
        denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        return denoised
    
    def normalize(self, image):
        """Normalize pixel values to [0, 1]"""
        return image.astype(np.float32) / 255.0
    
    def preprocess_pipeline(self, image_path, enhance=True, denoise_img=True):
        """
        Complete preprocessing pipeline
        
        Args:
            image_path: Path to input image
            enhance: Apply contrast enhancement
            denoise_img: Apply denoising
            
        Returns:
            Processed image, scale factor, offset
        """
        # Load image
        img = self.load_image(image_path)
        
        # Resize
        img, scale, offset = self.resize_image(img)
        
        # Optional enhancements
        if denoise_img:
            img = self.denoise(img)
        
        if enhance:
            img = self.enhance_contrast(img)
        
        return img, scale, offset
    
    def save_processed_image(self, image, output_path):
        """Save processed image"""
        # Convert RGB to BGR for OpenCV
        img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, img_bgr)


# Example usage
if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = ImagePreprocessor(target_size=(640, 480))
    
    # Process single image
    try:
        img, scale, offset = preprocessor.preprocess_pipeline(
            "data/raw/sample_cattle.jpg",
            enhance=True,
            denoise_img=True
        )
        
        # Save result
        preprocessor.save_processed_image(img, "data/processed/sample_processed.jpg")
        print("✅ Image preprocessing completed!")
        print(f"Scale factor: {scale}")
        print(f"Offset: {offset}")
        
    except FileNotFoundError:
        print("⚠️ Sample image not found. Please add images to data/raw/ folder")