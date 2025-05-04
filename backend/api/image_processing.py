import numpy as np
import base64
from io import BytesIO
import json
import PIL.Image
from scipy import ndimage

def read_image_from_base64(base64_string):
    """Decode a base64 image into a numpy array"""
   
    img_data = base64.b64decode(base64_string)
    
    
    import PIL.Image
    
   
    img = PIL.Image.open(BytesIO(img_data))
    
    img_array = np.array(img)
    
    return img_array

def image_to_base64(img):
    """Convert numpy array image to base64 string"""
    from io import BytesIO
    
    if len(img.shape) == 2:
        temp_img = PIL.Image.fromarray(img.astype(np.uint8), 'L')
        temp_img = temp_img.convert('RGB')  
    else:
        temp_img = PIL.Image.fromarray(img.astype(np.uint8))
        if img.shape[2] == 4:  
            print("Converting RGBA image to RGB for JPEG compatibility")
            temp_img = temp_img.convert('RGB')
    
    buffered = BytesIO()
    temp_img.save(buffered, format="JPEG")
    
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    return img_str

def to_grayscale(img):
    """Convert image to grayscale"""
    if len(img.shape) == 2:
        return img  
        
   
    gray = np.dot(img[...,:3], [0.299, 0.587, 0.114])
    return gray.astype(np.uint8)

def to_binary(img, threshold=127):
    """Convert image to binary using threshold"""
    gray = to_grayscale(img) if len(img.shape) > 2 else img
    binary = np.where(gray > threshold, 255, 0).astype(np.uint8)
    
   
    if len(img.shape) > 2:
        binary = np.stack([binary] * 3, axis=-1)
    
    return binary

def rotate_image(img, angle):
    """Rotate image by angle (in degrees)"""
    
    angle_rad = np.radians(angle)
    
    
    height, width = img.shape[0], img.shape[1]
    

    cos_a = np.abs(np.cos(angle_rad))
    sin_a = np.abs(np.sin(angle_rad))
    new_width = int(height * sin_a + width * cos_a)
    new_height = int(height * cos_a + width * sin_a)
    
   
    channels = img.shape[2] if len(img.shape) > 2 else 1
    if channels == 1:
        rotated = np.zeros((new_height, new_width), dtype=np.uint8)
    else:
        rotated = np.zeros((new_height, new_width, channels), dtype=np.uint8)
    

    center_old = (width // 2, height // 2)
    center_new = (new_width // 2, new_height // 2)
    

    y_coords, x_coords = np.meshgrid(np.arange(new_height), np.arange(new_width), indexing='ij')
    

    x_centered = x_coords - center_new[0]
    y_centered = y_coords - center_new[1]
    

    x_rotated = np.round(x_centered * np.cos(angle_rad) + y_centered * np.sin(angle_rad)).astype(int)
    y_rotated = np.round(-x_centered * np.sin(angle_rad) + y_centered * np.cos(angle_rad)).astype(int)
    

    x_orig = x_rotated + center_old[0]
    y_orig = y_rotated + center_old[1]
    

    valid_mask = (0 <= x_orig) & (x_orig < width) & (0 <= y_orig) & (y_orig < height)
    
    if channels == 1:
        rotated[valid_mask] = img[y_orig[valid_mask], x_orig[valid_mask]]
    else:

        for c in range(channels):
            rotated[valid_mask, c] = img[y_orig[valid_mask], x_orig[valid_mask], c]
    
    return rotated

def crop_image(img, x_start, y_start, width, height):
    """Crop image to specified dimensions"""

    img_height, img_width = img.shape[0], img.shape[1]
    x_start = max(0, min(x_start, img_width-1))
    y_start = max(0, min(y_start, img_height-1))
    

    width = min(width, img_width - x_start)
    height = min(height, img_height - y_start)
    
    return img[y_start:y_start+height, x_start:x_start+width].copy()

def zoom_image(img, scale_factor):
    """Zoom in/out image by scale factor using vectorized operations"""
    print(f"Zoom operation called with scale_factor: {scale_factor}, type: {type(scale_factor)}")
    
    if scale_factor <= 0:
        print("Invalid scale factor (≤ 0), returning original image")
        return img
        
    height, width = img.shape[0], img.shape[1]
    channels = img.shape[2] if len(img.shape) > 2 else 1
    

    new_height = int(height * scale_factor)
    new_width = int(width * scale_factor)
    
    print(f"Original image shape: {img.shape}, New dimensions: {new_width}x{new_height}")
    
    if channels == 1:
        pil_img = PIL.Image.fromarray(img)
    else:
        pil_img = PIL.Image.fromarray(img)
    
   
    try:
      
        resampling_method = PIL.Image.Resampling.LANCZOS
    except AttributeError:
        
        resampling_method = PIL.Image.LANCZOS
    
    
    try:
        pil_resized = pil_img.resize((new_width, new_height), resampling_method)
    except Exception as e:
        print(f"Error during PIL resize: {e}, trying with BICUBIC")
        
        try:
            if hasattr(PIL.Image, 'Resampling'):
                pil_resized = pil_img.resize((new_width, new_height), PIL.Image.Resampling.BICUBIC)
            else:
                pil_resized = pil_img.resize((new_width, new_height), PIL.Image.BICUBIC)
        except Exception as e:
            print(f"Error during BICUBIC resize: {e}, trying with default")

            pil_resized = pil_img.resize((new_width, new_height))
    
   
    zoomed = np.array(pil_resized)
    
    print(f"Zoom operation completed. Result shape: {zoomed.shape}")
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
        
       
        img_norm = img.astype(float) / 255.0
        
        r, g, b = img_norm[:,:,0], img_norm[:,:,1], img_norm[:,:,2]
        
        max_val = np.maximum(np.maximum(r, g), b)
        min_val = np.minimum(np.minimum(r, g), b)
        diff = max_val - min_val
        
        
        h = np.zeros_like(r)
        s = np.zeros_like(r)
        v = max_val
        
        
        mask = diff != 0
        
       
        mask_r = mask & (max_val == r)
        h[mask_r] = (60 * ((g[mask_r] - b[mask_r]) / diff[mask_r]) + 360) % 360
        
        
        mask_g = mask & (max_val == g)
        h[mask_g] = (60 * ((b[mask_g] - r[mask_g]) / diff[mask_g]) + 120) % 360
        
        
        mask_b = mask & (max_val == b)
        h[mask_b] = (60 * ((r[mask_b] - g[mask_b]) / diff[mask_b]) + 240) % 360
        
        
        mask_not_zero = max_val != 0
        s[mask_not_zero] = diff[mask_not_zero] / max_val[mask_not_zero]
        
        
        h = (h / 360.0 * 255.0).astype(np.uint8)
       
        s = (s * 255.0).astype(np.uint8)
        v = (v * 255.0).astype(np.uint8)
        
        return np.stack([h, s, v], axis=-1)
    
    
    return img

def calculate_histogram(img):
    """Calculate histogram of image"""
    print(f"Calculating histogram for image of shape {img.shape}")
    
    if len(img.shape) > 2:
        gray = to_grayscale(img)
        print(f"Converted to grayscale for histogram, shape: {gray.shape}")
    else:
        gray = img
    
    histogram = np.zeros(256, dtype=np.int32)
    for i in range(256):
        histogram[i] = np.sum(gray == i)
    
    print(f"Histogram min count: {np.min(histogram)}, max count: {np.max(histogram)}")
    return histogram

def histogram_equalization(img):
    """Apply histogram equalization to enhance contrast"""
    if len(img.shape) > 2:
        hsv = convert_color_space(img, "rgb_to_hsv")
        v_channel = hsv[:,:,2]
        v_eq = histogram_equalization(v_channel)
        hsv[:,:,2] = v_eq
        return img  
    
    hist = calculate_histogram(img)
    cum_hist = np.cumsum(hist)
    cum_hist_normalized = (cum_hist * 255 / cum_hist[-1]).astype(np.uint8)
    
    img_eq = np.zeros_like(img)
    for i in range(256):
        img_eq[img == i] = cum_hist_normalized[i]
    
    return img_eq

def histogram_stretching(img):
    """Apply histogram stretching to enhance contrast"""
    print(f"Histogram stretching: Input shape: {img.shape}")
    print(f"Input min: {np.min(img)}, max: {np.max(img)}")
    print(f"Input data type: {img.dtype}")
    
    if len(img.shape) > 2 and img.shape[2] == 4:
        print("Converting RGBA to RGB before histogram stretching")
        img = img[:, :, :3]  
    
    if len(img.shape) > 2:
        stretched = np.zeros_like(img)
        for i in range(min(3, img.shape[2])):  
            channel = img[:,:,i]
            min_val = np.min(channel)
            max_val = np.max(channel)
            
            print(f"Channel {i} - min: {min_val}, max: {max_val}")
            
            if min_val == max_val:
                stretched[:,:,i] = channel
                continue
                
            stretched[:,:,i] = ((channel - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        
        print(f"Stretched output min: {np.min(stretched)}, max: {np.max(stretched)}")
        return stretched
    
    min_val = np.min(img)
    max_val = np.max(img)
    
    print(f"Grayscale channel - min: {min_val}, max: {max_val}")
    
    if min_val == max_val:
        print("Warning: min and max values are the same, no stretching applied")
        return img
    
    stretched = ((img.astype(np.float32) - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    
    print(f"Output min: {np.min(stretched)}, max: {np.max(stretched)}")
    
    return stretched

def add_images(img1, img2):
    """Add two images together with pixel-wise addition"""
    if img1.shape != img2.shape:
        if len(img1.shape) != len(img2.shape):
            if len(img1.shape) == 2:
                img1 = np.stack([img1] * 3, axis=-1)
            elif len(img2.shape) == 2:
                img2 = np.stack([img2] * 3, axis=-1)
                
        if img1.size < img2.size:
            img1 = resize_image(img1, img2.shape[1], img2.shape[0])
        else:
            img2 = resize_image(img2, img1.shape[1], img1.shape[0])
    
    result = np.clip(img1.astype(np.int16) + img2.astype(np.int16), 0, 255).astype(np.uint8)
    return result

def divide_images(img1, img2):
    """Divide img1 by img2 (pixel-wise division)"""
    if img1.shape != img2.shape:
        if len(img1.shape) != len(img2.shape):
            if len(img1.shape) == 2:
                img1 = np.stack([img1] * 3, axis=-1)
            elif len(img2.shape) == 2:
                img2 = np.stack([img2] * 3, axis=-1)
                
        if img1.size < img2.size:
            img1 = resize_image(img1, img2.shape[1], img2.shape[0])
        else:
            img2 = resize_image(img2, img1.shape[1], img1.shape[0])
    
    img2_safe = np.where(img2 == 0, 1, img2)
    
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
    
    x_ratio = width / new_width
    y_ratio = height / new_height
    
    for y in range(new_height):
        for x in range(new_width):
            px = min(width-1, int(x * x_ratio))
            py = min(height-1, int(y * y_ratio))
            resized[y, x] = img[py, px]
    
    return resized

def enhance_contrast(img, factor=1.5):
    """Enhance image contrast by scaling around the mean"""
    if len(img.shape) > 2:
        enhanced = np.zeros_like(img)
        for i in range(3):
            enhanced[:,:,i] = enhance_contrast(img[:,:,i], factor)
        return enhanced
    
    mean = np.mean(img)
    
    enhanced = np.clip(mean + (img - mean) * factor, 0, 255).astype(np.uint8)
    
    return enhanced

def convolution(img, kernel):
    """Apply convolution with a kernel"""
    from scipy import ndimage
    
    if len(img.shape) == 3:
        result = np.zeros_like(img)
        for c in range(img.shape[2]):
            result[:,:,c] = ndimage.convolve(img[:,:,c], kernel, mode='constant', cval=0.0)
        return np.clip(result, 0, 255).astype(np.uint8)
    else:
        return np.clip(ndimage.convolve(img, kernel, mode='constant', cval=0.0), 0, 255).astype(np.uint8)

def mean_filter(img, size=3):
    """Apply mean filter (box blur)"""
    kernel = np.ones((size, size), dtype=np.float32) / (size * size)
    return convolution(img, kernel)

def thresholding(img, threshold=127):
    """Apply simple thresholding to create a binary image"""
    if len(img.shape) > 2:
        img = to_grayscale(img)
    
    print(f"Thresholding: Image shape: {img.shape}, min: {img.min()}, max: {img.max()}, mean: {img.mean():.1f}")
    print(f"Using threshold: {threshold}")
    
    binary = np.where(img > threshold, 255, 0).astype(np.uint8)
    
    white_pixels = np.sum(binary == 255)
    black_pixels = np.sum(binary == 0)
    total_pixels = white_pixels + black_pixels
    
    print(f"Binary result: White pixels: {white_pixels} ({white_pixels/total_pixels*100:.1f}%), Black pixels: {black_pixels} ({black_pixels/total_pixels*100:.1f}%)")
    
    return binary

def prewitt_edge_detection(img):
    """Apply Prewitt edge detection"""
    if len(img.shape) > 2:
        img = to_grayscale(img)
    
    kernel_x = np.array([[-1, 0, 1],
                         [-1, 0, 1],
                         [-1, 0, 1]])
    
    kernel_y = np.array([[-1, -1, -1],
                         [0, 0, 0],
                         [1, 1, 1]])
    
    edges_x = ndimage.convolve(img.astype(float), kernel_x)
    edges_y = ndimage.convolve(img.astype(float), kernel_y)
    
    edges = np.sqrt(edges_x**2 + edges_y**2)
    
    edges = edges * (255.0 / edges.max()) if edges.max() > 0 else edges
    
    return edges.astype(np.uint8)

def add_salt_pepper_noise(img, amount=0.05):
    """Add salt and pepper noise to an image"""
    output = np.copy(img)
    
    num_salt = int(amount * img.size)
    salt_coords = [np.random.randint(0, i - 1, num_salt) for i in img.shape]
    if len(img.shape) > 2:
        output[salt_coords[0], salt_coords[1], :] = 255
    else:
        output[salt_coords[0], salt_coords[1]] = 255
    
    num_pepper = int(amount * img.size)
    pepper_coords = [np.random.randint(0, i - 1, num_pepper) for i in img.shape]
    if len(img.shape) > 2:
        output[pepper_coords[0], pepper_coords[1], :] = 0
    else:
        output[pepper_coords[0], pepper_coords[1]] = 0
    
    return output

def median_filter(img, size=3):
    """Apply median filter to remove noise"""
    from scipy import ndimage
    
    if len(img.shape) == 3:
        result = np.zeros_like(img)
        for c in range(img.shape[2]):
            result[:,:,c] = ndimage.median_filter(img[:,:,c], size=size)
        return result
    else:
        return ndimage.median_filter(img, size=size)

def unsharp_mask(img, strength=1.0):
    """Apply unsharp mask filter to sharpen the image"""
    blurred = mean_filter(img, size=5)
    
    mask = img.astype(np.float32) - blurred.astype(np.float32)
    
    sharpened = np.clip(img.astype(np.float32) + strength * mask, 0, 255).astype(np.uint8)
    
    return sharpened

def morphology_erosion(img, kernel_size=3):
    """Apply morphological erosion"""
    if len(img.shape) > 2:
        img = to_grayscale(img)
    
    binary = thresholding(img)
    
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    
    print(f"Erosion: Input shape: {binary.shape}, kernel size: {kernel_size}")
    print(f"Input min: {binary.min()}, max: {binary.max()}")
    
    iterations = 1
    if kernel_size >= 5:
        iterations = 2
    if kernel_size >= 7:
        iterations = 3
        
    eroded = ndimage.binary_erosion(binary > 128, structure=kernel, iterations=iterations)
    
    output = np.where(eroded, 255, 0).astype(np.uint8)
    
    print(f"Output min: {output.min()}, max: {output.max()}, unique values: {np.unique(output)}")
    print(f"Effect applied: Shrinking/eroding white regions")
    
    return output

def morphology_dilation(img, kernel_size=3):
    """Apply morphological dilation"""
    if len(img.shape) > 2:
        img = to_grayscale(img)
    
    binary = thresholding(img)
    
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    
    print(f"Dilation: Input shape: {binary.shape}, kernel size: {kernel_size}")
    print(f"Input min: {binary.min()}, max: {binary.max()}")
    
    iterations = 1
    if kernel_size >= 5:
        iterations = 2
    if kernel_size >= 7:
        iterations = 3
        
    dilated = ndimage.binary_dilation(binary > 128, structure=kernel, iterations=iterations)
    
    output = np.where(dilated, 255, 0).astype(np.uint8)
    
    print(f"Output min: {output.min()}, max: {output.max()}, unique values: {np.unique(output)}")
    print(f"Effect applied: Expanding/growing white regions")
    
    return output

def morphology_opening(img, kernel_size=3):
    """Apply morphological opening (erosion followed by dilation)"""
    if len(img.shape) > 2:
        img = to_grayscale(img)
    
    binary = thresholding(img)
    
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    
    print(f"Opening: Input shape: {binary.shape}, kernel size: {kernel_size}")
    
    iterations = 1
    if kernel_size >= 5:
        iterations = 2
    if kernel_size >= 7:
        iterations = 3
        
    opened = ndimage.binary_opening(binary > 128, structure=kernel, iterations=iterations)
    
    output = np.where(opened, 255, 0).astype(np.uint8)
    
    print(f"Opening applied: Erode followed by dilate - removes small white spots and thin connections")
    
    return output

def morphology_closing(img, kernel_size=3):
    """Apply morphological closing (dilation followed by erosion)"""
    if len(img.shape) > 2:
        img = to_grayscale(img)
    
    binary = thresholding(img)
    
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    
    print(f"Closing: Input shape: {binary.shape}, kernel size: {kernel_size}")
    
    iterations = 1
    if kernel_size >= 5:
        iterations = 2
    if kernel_size >= 7:
        iterations = 3
        
    closed = ndimage.binary_closing(binary > 128, structure=kernel, iterations=iterations)
    
    output = np.where(closed, 255, 0).astype(np.uint8)
    
    print(f"Closing applied: Dilate followed by erode - fills small black holes and connects nearby white regions")
    
    return output

def process_image(img_data, operation, params=None):
    """Process the image based on the specified operation"""
    img = read_image_from_base64(img_data)
    
    operations = operation if isinstance(operation, list) else [operation]
    params_list = params if isinstance(params, list) else [params] * len(operations)
    
    processed = img
    histogram = None
    
    for idx, op in enumerate(operations):
        current_params = params_list[idx] if idx < len(params_list) else None
        
        if op == "grayscale":
            processed = to_grayscale(processed)
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
            try:
                print(f"Zoom processing with params: {current_params}")
                
                if current_params and "factor" in current_params:
                    raw_factor = current_params["factor"]
                    try:
                        factor = float(raw_factor)
                        factor = max(0.1, min(5.0, factor))
                    except (ValueError, TypeError):
                        print(f"Invalid factor value: {raw_factor}, using default")
                        factor = 1.5
                else:
                    factor = 1.5
                
                print(f"Applying zoom with factor: {factor}")
                processed = zoom_image(processed, factor)
                print(f"Zoom operation result shape: {processed.shape}")
            except Exception as e:
                print(f"Error during zoom operation: {e}")
                import traceback
                traceback.print_exc()
                print("Returning original image due to zoom error")
        elif op == "color_space":
            conv_type = current_params.get("type", "rgb_to_grayscale") if current_params else "rgb_to_grayscale"
            processed = convert_color_space(processed, conv_type)
        elif op == "histogram":
            print(f"Applying histogram stretching to image of shape {processed.shape}")
            try:
                processed = histogram_stretching(processed)
                
                if len(processed.shape) > 2:
                    if processed.shape[2] > 3:
                        processed = processed[:, :, :3]
                    elif processed.shape[2] == 1:
                        processed = processed[:, :, 0]
                
                print(f"Histogram stretching applied successfully. Result shape: {processed.shape}")
            except Exception as e:
                print(f"Error in histogram stretching: {e}")
                import traceback
                traceback.print_exc()
            if idx == len(operations) - 1:
                histogram = calculate_histogram(processed).tolist()
        elif op == "add_images":
            img2_data = current_params.get("img2_data") if current_params else None
            if img2_data:
                img2 = read_image_from_base64(img2_data)
                processed = add_images(processed, img2)
        elif op == "divide_images":
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
            
            if len(img.shape) > 2:
                processed = np.stack([processed] * 3, axis=-1)
        elif op == "morphology_dilation":
            kernel_size = int(current_params.get("kernel_size", 3)) if current_params else 3
            processed = morphology_dilation(processed, kernel_size)
           
            if len(img.shape) > 2:
                processed = np.stack([processed] * 3, axis=-1)
        elif op == "morphology_opening":
            kernel_size = int(current_params.get("kernel_size", 3)) if current_params else 3
            processed = morphology_opening(processed, kernel_size)
           
            if len(img.shape) > 2:
                processed = np.stack([processed] * 3, axis=-1)
        elif op == "morphology_closing":
            kernel_size = int(current_params.get("kernel_size", 3)) if current_params else 3
            processed = morphology_closing(processed, kernel_size)
           
            if len(img.shape) > 2:
                processed = np.stack([processed] * 3, axis=-1)
    
   
    result_base64 = image_to_base64(processed)
    
    return {
        "processed_image": result_base64,
        "histogram": histogram
    } 