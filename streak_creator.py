import os
from PIL import Image
import numpy as np
import random
import scipy.stats as stats

def add_gaussian_noise(image, mean=0, std=1):
    """
    Add Gaussian noise to an image.
    """
    gaussian = np.random.normal(mean, std, image.shape)
    noisy_image = np.clip(image + gaussian, 0, 255)
    return noisy_image

def irregular_gaussian_streak(width, noise_std=5):
    """
    Create an irregular Gaussian distributed streak with additional noise.
    """
    x = np.linspace(-3, 3, width)
    mean = random.uniform(-1, 1)  # Random mean
    std = random.uniform(0.5, 2.5)  # Random standard deviation
    gaussian_distribution = stats.norm.pdf(x, mean, std)
    gaussian_distribution /= gaussian_distribution.max()  # Normalize to 1
    gaussian_distribution *= 255  # Scale to [0, 255]

    # Adding additional Gaussian noise to the streak
    noise = np.random.normal(0, noise_std, width)
    noisy_streak = gaussian_distribution + noise
    return np.clip(noisy_streak, 0, 255)  # Ensuring values stay within [0, 255]

def random_intensity_variation(length, intensity_variation=0.5):
    """
    Create a random intensity variation along the streak.
    """
    variation = np.random.normal(1, intensity_variation, length)
    variation = np.clip(variation, 0, 2)  # Ensuring values stay within a reasonable range
    return variation

def create_randomized_streaks_and_mask(image, max_streak_width=34, max_streak_length_factor=0.5, streak_spacing_range=(70, 150), blend_factor=0.6):
    """
    Create randomized and irregular streaks on an image and a corresponding mask.
    """
    img_array = np.array(image)
    height, width = img_array.shape[:2]
    mask = np.zeros_like(img_array)

    x = 0
    while x < width:
        streak_width = random.randint(12, max_streak_width)
        streak_length = random.randint(int(height * 0.1), int(height * max_streak_length_factor))
        streak_start = random.randint(0, height - streak_length)
        streak_intensity = irregular_gaussian_streak(streak_width)
        intensity_variation = random_intensity_variation(streak_length)

        if x + streak_width < width:
            for i in range(streak_width):
                streak_slice = img_array[streak_start:streak_start + streak_length, x + i].astype(float)
                blended_streak = streak_slice * (1 - blend_factor) + streak_intensity[i] * blend_factor * intensity_variation[:, np.newaxis]
                img_array[streak_start:streak_start + streak_length, x + i] = np.clip(blended_streak, 0, 255)
                mask[streak_start:streak_start + streak_length, x + i] = 255

        x += streak_width + random.randint(*streak_spacing_range)

    img_array = add_gaussian_noise(img_array)

    return Image.fromarray(img_array.astype('uint8')), mask[:, :, 0] > 0

def process_images(input_dir, output_dir_images, output_dir_masks):
    """
    Process all images in the given directory and save the modified images and masks.
    """
    # Create output directories if they don't exist
    os.makedirs(output_dir_images, exist_ok=True)
    os.makedirs(output_dir_masks, exist_ok=True)

    # Process each image in the directory
    for i, filename in enumerate(os.listdir(input_dir)):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_dir, filename)
            original_image = Image.open(image_path)

            # Create randomized streaks and mask
            streaked_image, streak_mask = create_randomized_streaks_and_mask(original_image)

            # Save the modified image and mask
            modified_image_path = os.path.join(output_dir_images, f"{str(i).zfill(3)}.png")
            mask_image_path = os.path.join(output_dir_masks, f"{str(i).zfill(3)}.png")
            streaked_image.save(modified_image_path)
            mask_image = Image.fromarray(np.uint8(streak_mask) * 255)
            mask_image.save(mask_image_path)

# Specify the input and output directories
input_directory = './cropped_images_e926c20632_without_streaks'  # Directory containing the original images
output_directory_images = './modified_images'  # Directory to save modified images
output_directory_masks = './label_images'  # Directory to save masks

# Process the images
process_images(input_directory, output_directory_images, output_directory_masks)
