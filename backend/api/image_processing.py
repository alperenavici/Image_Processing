import numpy as np
import base64
from io import BytesIO
import json
import PIL.Image

def read_image_from_base64(base64_string):
    """Decode a base64 image into a numpy array"""
    # Base64 gelen veriyi çöz
    img_data = base64.b64decode(base64_string)
    
    # Daha güvenli bir yaklaşım - standart görüntü formatlarını doğrudan okuyalım
    import PIL.Image
    
    # Gelen veriyi bir PIL Image nesnesine dönüştür
    img = PIL.Image.open(BytesIO(img_data))
    
    # NumPy dizisine dönüştür
    img_array = np.array(img)
    
    return img_array

def image_to_base64(img):
    """Convert numpy array image to base64 string"""
    # NumPy dizisini PIL Image'e dönüştür
    from io import BytesIO
    
    # Eğer grayscale ise ve 2 boyutlu dizi ise, RGB'ye dönüştür
    if len(img.shape) == 2:
        temp_img = PIL.Image.fromarray(img.astype(np.uint8))
    else:
        temp_img = PIL.Image.fromarray(img.astype(np.uint8))
    
    # BytesIO ile görüntüyü belleğe kaydet
    buffered = BytesIO()
    temp_img.save(buffered, format="JPEG")
    
    # Base64'e dönüştür
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    return img_str

def to_grayscale(img):
    """Convert image to grayscale"""
    if len(img.shape) == 2:
        return img  # Already grayscale
        
    # Use weighted average method for RGB to grayscale conversion
    gray = np.dot(img[...,:3], [0.299, 0.587, 0.114])
    return gray.astype(np.uint8)

def to_binary(img, threshold=127):
    """Convert image to binary using threshold"""
    gray = to_grayscale(img) if len(img.shape) > 2 else img
    binary = np.where(gray > threshold, 255, 0).astype(np.uint8)
    
    # If original was RGB, convert back to RGB format
    if len(img.shape) > 2:
        binary = np.stack([binary] * 3, axis=-1)
    
    return binary

def rotate_image(img, angle):
    """Rotate image by angle (in degrees)"""
    # Convert angle to radians
    angle_rad = np.radians(angle)
    
    # Get image dimensions
    height, width = img.shape[0], img.shape[1]
    
    # Calculate new dimensions
    cos_a = np.abs(np.cos(angle_rad))
    sin_a = np.abs(np.sin(angle_rad))
    new_width = int(height * sin_a + width * cos_a)
    new_height = int(height * cos_a + width * sin_a)
    
    # Create a new empty image
    channels = img.shape[2] if len(img.shape) > 2 else 1
    if channels == 1:
        rotated = np.zeros((new_height, new_width), dtype=np.uint8)
    else:
        rotated = np.zeros((new_height, new_width, channels), dtype=np.uint8)
    
    # Calculate center points
    center_old = (width // 2, height // 2)
    center_new = (new_width // 2, new_height // 2)
    
    # Rotate pixel by pixel (inefficient but works without libraries)
    for y in range(new_height):
        for x in range(new_width):
            # Translate to origin
            x_centered = x - center_new[0]
            y_centered = y - center_new[1]
            
            # Rotate point
            x_rotated = int(x_centered * np.cos(angle_rad) + y_centered * np.sin(angle_rad))
            y_rotated = int(-x_centered * np.sin(angle_rad) + y_centered * np.cos(angle_rad))
            
            # Translate back
            x_orig = x_rotated + center_old[0]
            y_orig = y_rotated + center_old[1]
            
            # Check if the original pixel is within bounds
            if 0 <= x_orig < width and 0 <= y_orig < height:
                rotated[y, x] = img[y_orig, x_orig]
    
    return rotated

def crop_image(img, x_start, y_start, width, height):
    """Crop image to specified dimensions"""
    # Ensure coordinates are within image bounds
    img_height, img_width = img.shape[0], img.shape[1]
    x_start = max(0, min(x_start, img_width-1))
    y_start = max(0, min(y_start, img_height-1))
    
    # Ensure width and height don't exceed image bounds
    width = min(width, img_width - x_start)
    height = min(height, img_height - y_start)
    
    return img[y_start:y_start+height, x_start:x_start+width].copy()

def zoom_image(img, scale_factor):
    """Zoom in/out image by scale factor"""
    if scale_factor <= 0:
        return img
        
    height, width = img.shape[0], img.shape[1]
    channels = img.shape[2] if len(img.shape) > 2 else 1
    
    # Calculate new dimensions
    new_height = int(height * scale_factor)
    new_width = int(width * scale_factor)
    
    if channels == 1:
        zoomed = np.zeros((new_height, new_width), dtype=np.uint8)
    else:
        zoomed = np.zeros((new_height, new_width, channels), dtype=np.uint8)
    
    # Simple nearest neighbor interpolation
    for y in range(new_height):
        for x in range(new_width):
            orig_y = min(height-1, int(y / scale_factor))
            orig_x = min(width-1, int(x / scale_factor))
            zoomed[y, x] = img[orig_y, orig_x]
    
    return zoomed

def convert_color_space(img, conversion_type):
    """Convert image between color spaces"""
    if conversion_type == "rgb_to_grayscale":
        return to_grayscale(img)
    elif conversion_type == "grayscale_to_rgb":
        if len(img.shape) == 2:
            return np.stack([img] * 3, axis=-1)
        return img
    elif conversion_type == "rgb_to_hsv":
        if len(img.shape) < 3:
            img = np.stack([img] * 3, axis=-1)
        
        # Normalize RGB to [0,1]
        img_norm = img.astype(float) / 255.0
        
        r, g, b = img_norm[:,:,0], img_norm[:,:,1], img_norm[:,:,2]
        
        max_val = np.maximum(np.maximum(r, g), b)
        min_val = np.minimum(np.minimum(r, g), b)
        diff = max_val - min_val
        
        # Initialize H, S, V channels
        h = np.zeros_like(r)
        s = np.zeros_like(r)
        v = max_val
        
        # Calculate H
        # If max and min are the same (grayscale), H = 0
        mask = diff != 0
        
        # If max is R
        mask_r = mask & (max_val == r)
        h[mask_r] = (60 * ((g[mask_r] - b[mask_r]) / diff[mask_r]) + 360) % 360
        
        # If max is G
        mask_g = mask & (max_val == g)
        h[mask_g] = (60 * ((b[mask_g] - r[mask_g]) / diff[mask_g]) + 120) % 360
        
        # If max is B
        mask_b = mask & (max_val == b)
        h[mask_b] = (60 * ((r[mask_b] - g[mask_b]) / diff[mask_b]) + 240) % 360
        
        # Calculate S
        mask_not_zero = max_val != 0
        s[mask_not_zero] = diff[mask_not_zero] / max_val[mask_not_zero]
        
        # Scale H to [0, 255]
        h = (h / 360.0 * 255.0).astype(np.uint8)
        # Scale S and V to [0, 255]
        s = (s * 255.0).astype(np.uint8)
        v = (v * 255.0).astype(np.uint8)
        
        return np.stack([h, s, v], axis=-1)
    
    # Default: return the original image
    return img

def calculate_histogram(img):
    """Calculate histogram of image"""
    if len(img.shape) > 2:
        # Convert to grayscale for histogram
        gray = to_grayscale(img)
    else:
        gray = img
    
    # Calculate histogram (frequency of each intensity value 0-255)
    histogram = np.zeros(256, dtype=np.int32)
    for i in range(256):
        histogram[i] = np.sum(gray == i)
        
    return histogram

def histogram_equalization(img):
    """Apply histogram equalization to enhance contrast"""
    if len(img.shape) > 2:
        # For color images, equalize the luminance channel
        # Convert to HSV
        hsv = convert_color_space(img, "rgb_to_hsv")
        # Equalize the V channel
        v_channel = hsv[:,:,2]
        v_eq = histogram_equalization(v_channel)
        hsv[:,:,2] = v_eq
        # Convert back to RGB (not implemented here)
        return img  # Placeholder
    
    # Create a normalized cumulative histogram
    hist = calculate_histogram(img)
    cum_hist = np.cumsum(hist)
    cum_hist_normalized = (cum_hist * 255 / cum_hist[-1]).astype(np.uint8)
    
    # Create the equalized image by mapping each intensity to its equalized value
    img_eq = np.zeros_like(img)
    for i in range(256):
        img_eq[img == i] = cum_hist_normalized[i]
    
    return img_eq

def histogram_stretching(img):
    """Apply histogram stretching to enhance contrast"""
    if len(img.shape) > 2:
        # For color images, process each channel
        stretched = np.zeros_like(img)
        for i in range(3):
            stretched[:,:,i] = histogram_stretching(img[:,:,i])
        return stretched
    
    # Find the minimum and maximum pixel intensities
    min_val = np.min(img)
    max_val = np.max(img)
    
    # If min and max are the same, return the image as is
    if min_val == max_val:
        return img
    
    # Apply linear stretching
    stretched = ((img - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    
    return stretched

def add_images(img1, img2):
    """Add two images together with pixel-wise addition"""
    # Ensure both images have the same shape
    if img1.shape != img2.shape:
        # Resize the smaller image to match the larger one
        if img1.size < img2.size:
            img1 = resize_image(img1, img2.shape[1], img2.shape[0])
        else:
            img2 = resize_image(img2, img1.shape[1], img1.shape[0])
    
    # Add images and clip to valid range
    result = np.clip(img1.astype(np.int16) + img2.astype(np.int16), 0, 255).astype(np.uint8)
    return result

def divide_images(img1, img2):
    """Divide img1 by img2 (pixel-wise division)"""
    # Ensure both images have the same shape
    if img1.shape != img2.shape:
        # Resize the smaller image to match the larger one
        if img1.size < img2.size:
            img1 = resize_image(img1, img2.shape[1], img2.shape[0])
        else:
            img2 = resize_image(img2, img1.shape[1], img1.shape[0])
    
    # Avoid division by zero by setting zero pixels to 1
    img2_safe = np.where(img2 == 0, 1, img2)
    
    # Divide and scale to [0, 255]
    result = np.clip((img1.astype(np.float32) / img2_safe.astype(np.float32) * 255), 0, 255).astype(np.uint8)
    return result

def resize_image(img, new_width, new_height):
    """Resize image to specified dimensions using nearest neighbor interpolation"""
    height, width = img.shape[0], img.shape[1]
    channels = img.shape[2] if len(img.shape) > 2 else 1
    
    if channels == 1:
        resized = np.zeros((new_height, new_width), dtype=np.uint8)
    else:
        resized = np.zeros((new_height, new_width, channels), dtype=np.uint8)
    
    # Scale factors
    x_ratio = width / new_width
    y_ratio = height / new_height
    
    # Simple nearest neighbor interpolation
    for y in range(new_height):
        for x in range(new_width):
            px = min(width-1, int(x * x_ratio))
            py = min(height-1, int(y * y_ratio))
            resized[y, x] = img[py, px]
    
    return resized

def enhance_contrast(img, factor=1.5):
    """Enhance image contrast by scaling around the mean"""
    if len(img.shape) > 2:
        # For color images, process each channel
        enhanced = np.zeros_like(img)
        for i in range(3):
            enhanced[:,:,i] = enhance_contrast(img[:,:,i], factor)
        return enhanced
    
    # Calculate the mean pixel intensity
    mean = np.mean(img)
    
    # Apply contrast enhancement
    enhanced = np.clip(mean + (img - mean) * factor, 0, 255).astype(np.uint8)
    
    return enhanced

def convolution(img, kernel):
    """Apply convolution with the given kernel"""
    if len(img.shape) > 2:
        # For color images, apply convolution to each channel
        result = np.zeros_like(img)
        for i in range(img.shape[2]):
            result[:,:,i] = convolution(img[:,:,i], kernel)
        return result
    
    # Get kernel dimensions
    k_height, k_width = kernel.shape
    
    # Calculate padding
    pad_h = k_height // 2
    pad_w = k_width // 2
    
    # Create padded image
    padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
    
    # Initialize output image
    output = np.zeros_like(img)
    
    # Apply convolution
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            # Extract region of interest
            roi = padded[y:y+k_height, x:x+k_width]
            # Apply kernel
            output[y, x] = np.sum(roi * kernel)
    
    # Ensure values are in valid range
    return np.clip(output, 0, 255).astype(np.uint8)

def mean_filter(img, size=3):
    """Apply a mean filter for smoothing"""
    # Create a mean filter kernel
    kernel = np.ones((size, size)) / (size * size)
    return convolution(img, kernel)

def thresholding(img, threshold=127):
    """Apply simple thresholding to create a binary image"""
    if len(img.shape) > 2:
        # Convert to grayscale first
        img = to_grayscale(img)
    
    # Apply threshold
    binary = np.where(img > threshold, 255, 0).astype(np.uint8)
    
    return binary

def prewitt_edge_detection(img):
    """Apply Prewitt edge detection"""
    if len(img.shape) > 2:
        # Convert to grayscale first
        img = to_grayscale(img)
    
    # Prewitt kernels
    kernel_x = np.array([[-1, 0, 1],
                         [-1, 0, 1],
                         [-1, 0, 1]])
    
    kernel_y = np.array([[-1, -1, -1],
                         [0, 0, 0],
                         [1, 1, 1]])
    
    # Apply convolution
    edges_x = convolution(img, kernel_x)
    edges_y = convolution(img, kernel_y)
    
    # Combine the results
    edges = np.sqrt(edges_x.astype(np.float32)**2 + edges_y.astype(np.float32)**2)
    edges = np.clip(edges, 0, 255).astype(np.uint8)
    
    return edges

def add_salt_pepper_noise(img, amount=0.05):
    """Add salt and pepper noise to an image"""
    output = np.copy(img)
    
    # Add salt (white) noise
    num_salt = int(amount * img.size)
    salt_coords = [np.random.randint(0, i - 1, num_salt) for i in img.shape]
    if len(img.shape) > 2:
        output[salt_coords[0], salt_coords[1], :] = 255
    else:
        output[salt_coords[0], salt_coords[1]] = 255
    
    # Add pepper (black) noise
    num_pepper = int(amount * img.size)
    pepper_coords = [np.random.randint(0, i - 1, num_pepper) for i in img.shape]
    if len(img.shape) > 2:
        output[pepper_coords[0], pepper_coords[1], :] = 0
    else:
        output[pepper_coords[0], pepper_coords[1]] = 0
    
    return output

def median_filter(img, size=3):
    """Apply a median filter to remove noise"""
    if len(img.shape) > 2:
        # For color images, apply median filter to each channel
        result = np.zeros_like(img)
        for i in range(img.shape[2]):
            result[:,:,i] = median_filter(img[:,:,i], size)
        return result
    
    # Calculate padding
    pad = size // 2
    
    # Create padded image
    padded = np.pad(img, ((pad, pad), (pad, pad)), mode='reflect')
    
    # Initialize output image
    output = np.zeros_like(img)
    
    # Apply median filter
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            # Extract window
            window = padded[y:y+size, x:x+size]
            # Find median
            output[y, x] = np.median(window)
    
    return output.astype(np.uint8)

def unsharp_mask(img, strength=1.0):
    """Apply unsharp mask filter to sharpen the image"""
    # Apply a blur
    blurred = mean_filter(img, size=5)
    
    # Calculate the mask
    mask = img.astype(np.float32) - blurred.astype(np.float32)
    
    # Apply the mask
    sharpened = np.clip(img.astype(np.float32) + strength * mask, 0, 255).astype(np.uint8)
    
    return sharpened

def morphology_erosion(img, kernel_size=3):
    """Apply morphological erosion"""
    # Ensure the image is binary
    if len(img.shape) > 2:
        # Convert to grayscale first
        img = to_grayscale(img)
    
    binary = thresholding(img)
    
    # Create a kernel
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    
    # Get kernel dimensions
    k_height, k_width = kernel.shape
    
    # Calculate padding
    pad_h = k_height // 2
    pad_w = k_width // 2
    
    # Create padded image
    padded = np.pad(binary, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    
    # Initialize output image
    output = np.zeros_like(binary)
    
    # Apply erosion
    for y in range(binary.shape[0]):
        for x in range(binary.shape[1]):
            # Extract window
            window = padded[y:y+k_height, x:x+k_width]
            # If all pixels under the kernel are 255, then set output pixel to 255
            if np.all(window[kernel == 1] == 255):
                output[y, x] = 255
    
    return output

def morphology_dilation(img, kernel_size=3):
    """Apply morphological dilation"""
    # Ensure the image is binary
    if len(img.shape) > 2:
        # Convert to grayscale first
        img = to_grayscale(img)
    
    binary = thresholding(img)
    
    # Create a kernel
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    
    # Get kernel dimensions
    k_height, k_width = kernel.shape
    
    # Calculate padding
    pad_h = k_height // 2
    pad_w = k_width // 2
    
    # Create padded image
    padded = np.pad(binary, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    
    # Initialize output image
    output = np.zeros_like(binary)
    
    # Apply dilation
    for y in range(binary.shape[0]):
        for x in range(binary.shape[1]):
            # Extract window
            window = padded[y:y+k_height, x:x+k_width]
            # If any pixel under the kernel is 255, then set output pixel to 255
            if np.any(window[kernel == 1] == 255):
                output[y, x] = 255
    
    return output

def morphology_opening(img, kernel_size=3):
    """Apply morphological opening (erosion followed by dilation)"""
    eroded = morphology_erosion(img, kernel_size)
    opened = morphology_dilation(eroded, kernel_size)
    return opened

def morphology_closing(img, kernel_size=3):
    """Apply morphological closing (dilation followed by erosion)"""
    dilated = morphology_dilation(img, kernel_size)
    closed = morphology_erosion(dilated, kernel_size)
    return closed

def process_image(img_data, operation, params=None):
    """Process the image based on the specified operation"""
    # Convert base64 to image array
    img = read_image_from_base64(img_data)
    
    # Operasyon tek bir string ya da bir string listesi/array olabilir
    operations = operation if isinstance(operation, list) else [operation]
    params_list = params if isinstance(params, list) else [params] * len(operations)
    
    # Her bir işlemi sırayla uygula
    processed = img
    histogram = None
    
    for idx, op in enumerate(operations):
        # Mevcut işlem için parametreleri al
        current_params = params_list[idx] if idx < len(params_list) else None
        
        # İşlemi uygula
        if op == "grayscale":
            processed = to_grayscale(processed)
            # Convert back to 3 channel if original was 3 channel
            if len(img.shape) > 2:
                processed = np.stack([processed] * 3, axis=-1)
        elif op == "binary":
            threshold = int(current_params.get("threshold", 127)) if current_params else 127
            processed = to_binary(processed, threshold)
        elif op == "rotate":
            angle = float(current_params.get("angle", 90)) if current_params else 90
            processed = rotate_image(processed, angle)
        elif op == "crop":
            if current_params:
                x = int(current_params.get("x", 0))
                y = int(current_params.get("y", 0))
                width = int(current_params.get("width", processed.shape[1]//2))
                height = int(current_params.get("height", processed.shape[0]//2))
            else:
                x, y = 0, 0
                width, height = processed.shape[1]//2, processed.shape[0]//2
            processed = crop_image(processed, x, y, width, height)
        elif op == "zoom":
            factor = float(current_params.get("factor", 1.5)) if current_params else 1.5
            processed = zoom_image(processed, factor)
        elif op == "color_space":
            conv_type = current_params.get("type", "rgb_to_grayscale") if current_params else "rgb_to_grayscale"
            processed = convert_color_space(processed, conv_type)
        elif op == "histogram":
            processed = histogram_stretching(processed)
            # Son işlemse histogram hesapla
            if idx == len(operations) - 1:
                histogram = calculate_histogram(processed).tolist()
        elif op == "add_images":
            # For add_images, we need a second image
            img2_data = current_params.get("img2_data") if current_params else None
            if img2_data:
                img2 = read_image_from_base64(img2_data)
                processed = add_images(processed, img2)
        elif op == "divide_images":
            # For divide_images, we need a second image
            img2_data = current_params.get("img2_data") if current_params else None
            if img2_data:
                img2 = read_image_from_base64(img2_data)
                processed = divide_images(processed, img2)
        elif op == "contrast":
            factor = float(current_params.get("factor", 1.5)) if current_params else 1.5
            processed = enhance_contrast(processed, factor)
        elif op == "convolution_mean":
            kernel_size = int(current_params.get("size", 3)) if current_params else 3
            processed = mean_filter(processed, kernel_size)
        elif op == "threshold":
            threshold = int(current_params.get("threshold", 127)) if current_params else 127
            processed = thresholding(processed, threshold)
        elif op == "edge_prewitt":
            processed = prewitt_edge_detection(processed)
            # Convert back to 3 channel if original was 3 channel
            if len(img.shape) > 2:
                processed = np.stack([processed] * 3, axis=-1)
        elif op == "salt_pepper":
            amount = float(current_params.get("amount", 0.05)) if current_params else 0.05
            processed = add_salt_pepper_noise(processed, amount)
        elif op == "filter_mean":
            size = int(current_params.get("size", 3)) if current_params else 3
            processed = mean_filter(processed, size)
        elif op == "filter_median":
            size = int(current_params.get("size", 3)) if current_params else 3
            processed = median_filter(processed, size)
        elif op == "unsharp":
            strength = float(current_params.get("strength", 1.0)) if current_params else 1.0
            processed = unsharp_mask(processed, strength)
        elif op == "morphology_erosion":
            kernel_size = int(current_params.get("kernel_size", 3)) if current_params else 3
            processed = morphology_erosion(processed, kernel_size)
            # Convert back to 3 channel if original was 3 channel
            if len(img.shape) > 2:
                processed = np.stack([processed] * 3, axis=-1)
        elif op == "morphology_dilation":
            kernel_size = int(current_params.get("kernel_size", 3)) if current_params else 3
            processed = morphology_dilation(processed, kernel_size)
            # Convert back to 3 channel if original was 3 channel
            if len(img.shape) > 2:
                processed = np.stack([processed] * 3, axis=-1)
        elif op == "morphology_opening":
            kernel_size = int(current_params.get("kernel_size", 3)) if current_params else 3
            processed = morphology_opening(processed, kernel_size)
            # Convert back to 3 channel if original was 3 channel
            if len(img.shape) > 2:
                processed = np.stack([processed] * 3, axis=-1)
        elif op == "morphology_closing":
            kernel_size = int(current_params.get("kernel_size", 3)) if current_params else 3
            processed = morphology_closing(processed, kernel_size)
            # Convert back to 3 channel if original was 3 channel
            if len(img.shape) > 2:
                processed = np.stack([processed] * 3, axis=-1)
    
    # Convert back to base64
    result_base64 = image_to_base64(processed)
    
    return {
        "processed_image": result_base64,
        "histogram": histogram
    } 