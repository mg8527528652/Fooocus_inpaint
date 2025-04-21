from PIL import Image
import uuid
import threading
import traceback
import random

def preprocess_watermark(watermark_img, main_width, main_height):
    watermark = watermark_img
    watermark_height = int(main_height * 0.07)  # 7% of the main image's height
    watermark = watermark.resize((int(watermark.width * (watermark_height / watermark.height)), watermark_height))
    watermark_width = watermark.width
    x_position = (main_width - watermark_width) // 2  # Centered on x-axis
    y_position = main_height - watermark_height - int(main_height * 0.02)
    return watermark, x_position, y_position


def add_watermark(image: Image, watermark, x_position, y_position):
    try:
        image = image.convert("RGBA")
        image.paste(watermark, (x_position, y_position), watermark)
        return image.convert("RGB")
    
    except:
        print(f"Error in adding watermark")
        print(traceback.format_exc())
        return image


def process_and_save_image(output_dir, img, date, order_id, image_extension, watermark, x_position, y_position, static_urls = [], output_doc = {}):
    try:
        idx = 0
        # Save the original image
        image_path = f"{output_dir}/images/{date}/{order_id}_{random.randint(0,1000000)}_{idx}.{image_extension}"
        img.save(image_path)
        url_path = image_path.replace(f"{output_dir}/images/", "")
        # Generate URL for the original image
        without_watermark_url = f"https://static-cb2.phot.ai/bg_replacer/{url_path}"
        # Process and save watermark image if watermark is provided
        if watermark:
            watermark_image = add_watermark(img, watermark, x_position, y_position)
            watermark_file_name = str(uuid.uuid4())
            watermark_image_path = f"{output_dir}/images/{date}/{watermark_file_name}.{image_extension}"
            watermark_image.save(watermark_image_path)

            # Generate URL for the watermarked image
            with_watermark_url = f"https://static-cb2.phot.ai/bg_replacer/{date}/{watermark_file_name}.{image_extension}"
        else:
            with_watermark_url = None


        output_doc[f"{idx}"] = {
            "without_watermark": without_watermark_url,
            "with_watermark": with_watermark_url
        }
        static_urls.append(without_watermark_url)
        return output_doc, static_urls
    except Exception as e:
        print(f"Error processing image {idx}: {e}")

def parallel_postprocess_images(output_dir, date, images, order_id, image_extension, watermark, x_position, y_position):

    threads = []
    lock = threading.Lock()
    output_doc = {}
    static_urls = []

    # Create and start threads
    for idx, img in enumerate(images):
        thread = threading.Thread(
            target=process_and_save_image,
            args=(output_dir,idx, img, date, order_id, image_extension, watermark, x_position, y_position, output_doc, static_urls, lock)
        )
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    return output_doc, static_urls