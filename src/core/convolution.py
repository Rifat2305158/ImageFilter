"""Signal processing core logic."""
import numpy as np

def convolve2d(image: np.ndarray, kernel: np.ndarray, padding_mode: str = 'zero') -> np.ndarray:
    """
    Perform a 2D discrete convolution on a grayscale image using a given kernel.
    
    The convolution is implemented manually using sliding window loops.
    The kernel is mathematically flipped prior to applying the window.
    
    Args:
        image (np.ndarray): 2D numpy array representing the input image.
        kernel (np.ndarray): 2D numpy array representing the filter kernel. Must have odd dimensions.
        padding_mode (str): Padding strategy. One of 'zero', 'reflect', 'edge'.
        
    Returns:
        np.ndarray: The convolved image as a 2D float64 numpy array.
    """
    # 1. Input Validation
    if not isinstance(image, np.ndarray) or not isinstance(kernel, np.ndarray):
        raise TypeError("Image and kernel must be numpy arrays.")
        
    if image.ndim != 2:
        raise ValueError(f"Image must be a 2D array, got {image.ndim}D.")
        
    if kernel.ndim != 2:
        raise ValueError(f"Kernel must be a 2D array, got {kernel.ndim}D.")
        
    if kernel.size == 0:
        raise ValueError("Kernel cannot be empty.")
        
    k_height, k_width = kernel.shape
    if k_height % 2 == 0 or k_width % 2 == 0:
        raise ValueError(f"Kernel dimensions must be odd, got {kernel.shape}.")
        
    if padding_mode not in ('zero', 'reflect', 'edge'):
        raise ValueError(f"Unsupported padding mode: {padding_mode}")

    # Convert to float64 to maintain precision and avoid overflow
    image_f = image.astype(np.float64)
    kernel_f = kernel.astype(np.float64)
    
    # 2. Kernel Flipping
    # Mathematical convolution requires flipping the kernel horizontally and vertically.
    flipped_kernel = np.flip(kernel_f)
    
    # 3. Padding
    pad_h = k_height // 2
    pad_w = k_width // 2
    
    if padding_mode == 'zero':
        padded_image = np.pad(image_f, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    elif padding_mode == 'reflect':
        padded_image = np.pad(image_f, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
    elif padding_mode == 'edge':
        padded_image = np.pad(image_f, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
        
    # 4. Manual Convolution sliding window loop
    img_height, img_width = image.shape
    output = np.zeros((img_height, img_width), dtype=np.float64)
    
    # Iterate over every pixel in the output image
    for y in range(img_height):
        for x in range(img_width):
            # Extract the local neighborhood from the padded image
            region = padded_image[y : y + k_height, x : x + k_width]
            
            # Element-wise multiplication and summation using NumPy (equivalent to inner nested loops)
            output[y, x] = np.sum(region * flipped_kernel)
            
    return output
