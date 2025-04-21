import io
import json
import time
import random
import os
import sys
import numpy as np
from PIL import Image, ImageOps
from io import BytesIO
import cv2
import requests
import threading
import traceback
import logging
from logging.handlers import RotatingFileHandler
import ray
from ray import serve
from starlette.requests import Request
from postprocessing_utils import process_and_save_image, preprocess_watermark
import torch
# Set up environment variables
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["RAY_SERVE_QUEUE_LENGTH_RESPONSE_DEADLINE_S"] = "60"

# Initialize logging
logger = logging.getLogger("ray.serve")
logger.setLevel(logging.INFO)

os.makedirs('logs', exist_ok=True)

handler = RotatingFileHandler('logs/inpaint_history.log',
                              maxBytes=2 * 1024 * 1024,
                              backupCount=5)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def get_ist(t, logger):
    # Define the offset for IST (UTC + 5 hours 30 minutes = 19800 seconds)
    IST_OFFSET = 5 * 3600 + 30 * 60

    # Convert UTC time to IST by adding the IST offset
    current_time_ist = t + IST_OFFSET

    # Convert to an integer timestamp in IST
    current_time_ist_int = int(current_time_ist)

    # Optional: Convert the IST timestamp to a human-readable format
    time_struct_ist = time.localtime(current_time_ist)
    current_time_ist_str = time.strftime('%Y-%m-%d %H:%M:%S', time_struct_ist)

    logger.info(f"current IST Time: {current_time_ist_str}")
    return current_time_ist_str

@serve.deployment(num_replicas=1, ray_actor_options={"num_gpus": 1})
class InpaintService:
    def __init__(self):
        # Initialize Fooocus
        import args_manager
        self.args = args_manager.args
        self.args.disable_image_log = True
        self.args.disable_metadata = True

        import modules.config as config
        import modules.flags as flags
        import modules.async_worker as worker
        import modules.core as core
        import ldm_patched.modules.model_management as model_management
        from modules.sdxl_styles import legal_style_names
        from modules.util import HWC3, resize_image, get_file_from_folder_list
        from modules.private_logger import log
        from modules.async_worker import AsyncTask, handler

        # Store necessary imports for later use
        self.worker = worker
        self.model_management = model_management
        self.HWC3 = HWC3
        self.resize_image = resize_image
        self.AsyncTask = AsyncTask
        self.handler = handler

        # Create temp directory for processing
        self.temp_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'temp')
        os.makedirs(self.temp_path, exist_ok=True)
        self.watermark = Image.open('watermark.png').convert('RGBA')
        # Output directory for logging images
        self.output_dir = 'images-log'
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
        
        # Default args dictionary path
        self.args_json_path = '/root/Fooocus_inpaint/tmp_args_dict.json'
        
        logger.info("InpaintService initialized")

    def get_inpaint_task(self, image_path, mask_path, prompt, strength=0.7):
        """
        Create an inpainting task similar to how webui.py does it
        
        Args:
            image_path: Path to the input image
            mask_path: Path to the mask image
            prompt: The prompt for inpainting
            strength: Inpainting strength (0.0-1.0)
            
        Returns:
            worker.AsyncTask: The task ready to be processed
        """
        # Load the base parameters from the JSON file
        with open(self.args_json_path, 'r') as f:
            args_dict = json.load(f)
        
        # Load and prepare images
        input_image = self.HWC3(np.array(Image.open(image_path)))
        mask = np.array(Image.open(mask_path))
        if len(mask.shape) == 3:
            mask = mask[:,:,0]  # Take first channel if RGB
        
        # Ensure proper dimensions
        H, W, C = input_image.shape
        mask = self.resize_image(mask, W, H)
        
        # Set critical inpainting parameters
        black = np.zeros_like(input_image)
        image_input = {
            'image': input_image,
            'mask': black
        }
        mask_input = {
            'image': mask,
            'mask': black
        }
        
        args_dict['inpaint_input_image'] = image_input
        args_dict['inpaint_mask_image_upload'] = mask_input
        args_dict['inpaint_additional_prompt'] = prompt
        args_dict['prompt'] = prompt
        args_dict['seed'] = random.randint(0, 10000000)
        
        # Create the AsyncTask
        task = self.AsyncTask(args=args_dict)
        return task, (W, H)

    def run_inpaint(self, image_path, mask_path, prompt, strength=0.7):
        """
        Run inpainting using Fooocus async worker pattern

        Args:
            image_path: Path to the input image
            mask_path: Path to the mask image 
            prompt: The prompt describing what to generate
            output_path: Where to save the result
            strength: Inpainting strength (0.0-1.0)
            
        Returns:
            str: Path to the generated image
        """
        try:
            # Make sure we won't be interrupted
            with self.model_management.interrupt_processing_mutex:
                self.model_management.interrupt_processing = False
            
            # Create the task
            logger.info("Creating inpainting task...")
            task, (w, h) = self.get_inpaint_task(image_path, mask_path, prompt, strength)
            
            # Process the task
            results = self.handler(task)
            
            # Save and return result
            if not results:
                raise RuntimeError("No image was generated")
                
            # output_path = output_path or os.path.join(os.getcwd(), "inpainted_result.png")
            # if isinstance(results[0], str) and os.path.exists(results[0]):
            #     # If the result is a file path, rename it to our target
            #     os.rename(results[0], output_path)
            # else:
            #     # If the result is an image array
            #     Image.fromarray(results[0]).save(output_path)
            results = Image.fromarray(results[0])
            return results, (w, h)
            
        except Exception as e:
            logger.error(f"Error in run_inpaint: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def save_url_image(self, url, save_path):
        '''
        Save image from url to save_path
        '''
        response = requests.get(url)
        with open(save_path, 'wb') as f:
            f.write(response.content)

    async def inpaint_image(self, prompt, image_url, mask_url, order_id, 
                           negative_prompt="", strength=0.7,
                           image_extension="webp", user_type="FREE"):
        """
        Process inpainting request with images from URLs
        """
        start_time = time.time()    

        # Create client id
        client_id = order_id + '_' + str(int(random.uniform(0, 10000)))
        logger.info(f'Running inpainting for {client_id}............')
        # get output dir as: output_path/date/order_id/inpainted.webp
        curr_time = get_ist(time.time(), logger)
        date = curr_time.split(" ")[0]
        # Temp path for storing images and masks
        client_path = os.path.join(self.temp_path, date, client_id)

        output_save_path = os.path.join(self.output_dir, date)
        os.makedirs(output_save_path, exist_ok=True)
        os.makedirs(client_path, exist_ok=True)
        
        # Download and process image and mask
        def process_image():
            nonlocal image_url, image_path

            image_path = os.path.join(client_path, 'input_image.png')

            if image_url.startswith('http'):
                # Stream image directly from URL
                response = requests.get(image_url, timeout=10)
                response.raise_for_status()  # Ensure we catch bad responses

                with Image.open(BytesIO(response.content)) as img:
                    img = ImageOps.exif_transpose(img)
                    img.save(image_path)
            else:
                # Handle local image path
                with Image.open(image_url) as img:
                    img = ImageOps.exif_transpose(img)
                    img.save(image_path)
        def process_mask():
            nonlocal mask_url, mask_path

            mask_path = os.path.join(client_path, 'input_mask.png')

            if mask_url.startswith('http'):
                # Stream mask directly from URL
                response = requests.get(mask_url, timeout=10)
                response.raise_for_status()

                with Image.open(BytesIO(response.content)) as mask:
                    mask = ImageOps.exif_transpose(mask)
                    mask.save(mask_path)
            else:
                # Local file handling
                with Image.open(mask_url) as mask:
                    mask = ImageOps.exif_transpose(mask)
                    mask.save(mask_path)
        # Initialize paths
        image_path = None
        mask_path = None
        # Create and start threads
        image_thread = threading.Thread(target=process_image)
        mask_thread = threading.Thread(target=process_mask)
        image_thread.start()
        mask_thread.start()
        # Wait for both threads to complete
        image_thread.join()
        mask_thread.join()
        # Generate output filename
        print(f'Preprocessing time {time.time() - start_time:.2f} seconds')

       
        # Run the inpainting process
        try:
            inpaint_start_time = time.time()
            result, (w, h) = self.run_inpaint(
                image_path,
                mask_path, 
                prompt,
                strength=strength
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                if hasattr(torch.cuda, 'memory_stats'):
                    torch.cuda.reset_peak_memory_stats()
            # print(result_path)
            logger.info(f"Inpainting completed in {time.time() - inpaint_start_time:.2f} seconds")
            
            post_process_upload_time = time.time()
            curr_time = get_ist(time.time(), logger)
            # URL
             # if user is free, add watermark
            preprocess_watermark_time = time.time()
            if user_type == "FREE":
                watermark, x_position, y_position = preprocess_watermark(self.watermark, w, h)
            else:
                watermark, x_position, y_position = None, None, None
            print(f'Preprocessing time {time.time() - preprocess_watermark_time:.2f} seconds')
        
            static_urls =[]
            output_doc, static_urls = process_and_save_image(self.output_dir, result, date, order_id, image_extension, watermark, x_position, y_position)
            logger.info(f"Wasabi upload and adding watermark time: {time.time() - post_process_upload_time:.2f} seconds")
            
            end_time = time.time()
           
            print(f"Total time taken: {end_time - start_time} seconds")
            # Saving and uploading to s3 and adding watermark
            return {
                "output_doc": output_doc,
                "static_urls": static_urls,
            }
        except Exception as e:
            logger.error(f"Error during inpainting: {e}")
            logger.error(traceback.format_exc())
            return {"error": str(e)}

    async def __call__(self, request: Request):
        path = request.url.path
        method = request.method
        
        if path == "/" and method == 'GET':
            return {"status": "Fooocus Inpainting Service is running"}
        
        elif path == "/inpaint" and method == "POST":
            try:
                json_body = await request.json()
                
                # Validate required parameters
                if "prompt" not in json_body:
                    return json.dumps({"error": "prompt is required"})
                if "image_url" not in json_body:
                    return json.dumps({"error": "image_url is required"})
                if "mask_url" not in json_body:
                    return json.dumps({"error": "mask_url is required"})
                
                # Extract parameters from the request
                prompt = json_body.get("prompt", "")
                image_url = json_body.get("image_url", "")
                mask_url = json_body.get("mask_url", "")
                order_id = json_body.get("order_id", "default_order")
                negative_prompt = json_body.get("negative_prompt", "")
                strength = float(json_body.get("strength", 1.0))
                image_extension = json_body.get("image_extension", "webp")
                user_type = json_body.get("user_type", "FREE")
                # Process the inpainting request
                result = await self.inpaint_image(
                    prompt,
                    image_url,
                    mask_url,
                    order_id,
                    negative_prompt=negative_prompt,
                    strength=strength,
                    image_extension=image_extension,
                    user_type=user_type
                )
                return {'status_code': 200, 'body': json.dumps(result)}
                
            except Exception as e:
                logger.error(f"Error processing request: {e}")
                logger.error(traceback.format_exc())
                return {'status_code': 500, 'body': json.dumps({"error": str(e)})}
        
        else:
            return json.dumps({"error": "Invalid endpoint or method"})

# Initialize Ray and start the deployment
# if __name__ == "__main__":
if not ray.is_initialized():
    ray.init(ignore_reinit_error=True, dashboard_host='0.0.0.0', include_dashboard=True)
    logger.info("Ray initialized.")
    serve.start(http_options={"host": "0.0.0.0", "port": 8000})

deployment = InpaintService.bind()
logger.info("InpaintService deployment started") 