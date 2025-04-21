import os
import sys
import json
import time
import numpy as np
from PIL import Image
import random
# Set up environment variables
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

# Initialize Fooocus
import args_manager
args = args_manager.parse_args()
args.disable_image_log = True
args.disable_metadata = True


import ldm_patched.modules.model_management as model_management
from modules.util import HWC3, resize_image
from modules.async_worker import AsyncTask, handler



def get_inpaint_task(image_path, mask_path, prompt, strength=0.7, args_json_path='/root/Fooocus_inpaint/tmp_args_dict.json'):
    """
    Create an inpainting task similar to how webui.py does it
    
    Args:
        image_path: Path to the input image
        mask_path: Path to the mask image
        prompt: The prompt for inpainting
        strength: Inpainting strength (0.0-1.0)
        args_json_path: Path to the JSON config
        
    Returns:
        worker.AsyncTask: The task ready to be processed
    """
    # Load the base parameters from the JSON file
    with open(args_json_path, 'r') as f:
        args_dict = json.load(f)
    
    # Load and prepare images
    input_image = HWC3(np.array(Image.open(image_path)))
    mask = np.array(Image.open(mask_path))
    if len(mask.shape) == 3:
        mask = mask[:,:,0]  # Take first channel if RGB
    
    # Ensure proper dimensions
    H, W, C = input_image.shape
    mask = resize_image(mask, W, H)
    
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
    # args_dict['inpaint_strength'] = strength
    # args_dict['inpaint_disable_initial_latent'] = False  # Critical for proper inpainting
    # args_dict['inpaint_respective_field'] = 0.618  # Balanced respective field
    # args_dict['current_tab'] = 'inpaint'
    
    # Make sure we're using the inpaint pipeline
    # args_dict['mixing_image_prompt_and_inpaint'] = False
    # args_dict['input_image_checkbox'] = True

    # Convert dict to list of args (the format async_worker expects)
    args_list = args_dict
    
    # Create the AsyncTask - the first element in args should be the task itself
    # which gets removed in get_task function in webui.py
    task = AsyncTask(args=args_list)
    return task

def run_inpaint(image_path, mask_path, prompt, output_path=None, strength=0.7):
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
        with model_management.interrupt_processing_mutex:
            model_management.interrupt_processing = False
        
        # Create the task
        print("Creating inpainting task...")
        task = get_inpaint_task(image_path, mask_path, prompt, strength)
        
        results = handler(task)
        # # Clean up CUDA memory
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, 'memory_stats'):
                torch.cuda.reset_peak_memory_stats()
        
        # Save and return result
        if not results:
            raise RuntimeError("No image was generated")
            
        output_path = output_path or os.path.join(os.getcwd(), "inpainted_result.png")
        if isinstance(results[0], str) and os.path.exists(results[0]):
            # If the result is a file path, rename it to our target
            os.rename(results[0], output_path)
        else:
            import cv2
            # If the result is an image array
            # cv2.imwrite(output_path, results[0])
            Image.fromarray(results[0]).save(output_path)
        
        return output_path
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error in run_inpaint: {str(e)}")
        raise

if __name__ == "__main__":
    # Example usage
    images_path = r'/root/Fooocus_inpaint/data/input'
    masks_path = r'/root/Fooocus_inpaint/data/mask'
    output_path = r'/root/Fooocus_inpaint/data/output'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # for i in range(10):
    for image_name in os.listdir(images_path):
        # if image_name != '12_replace basket flower with fruits and vegetables.jpg':
        #     continue
        image_path = os.path.join(images_path, image_name)
        mask_name, prompt = image_name.split('.')[0].split('_')
        # prompt = 'mountain'
        mask_path = os.path.join(masks_path, mask_name + '.jpg')
        time_start = time.time()
        result_path = run_inpaint(
            image_path,
            mask_path, 
            prompt,
            os.path.join(output_path, image_name),
            strength=1.0  # Using a more moderate strength to preserve more of the original image
        )          
        time_end = time.time()
        print(f"Total Time taken: {time_end - time_start:.2f} seconds for {image_name}")
        print(f"Generated image saved to: {result_path}")
        print('*'*100   , 'Inference completed', '*'*100)


    # result_path = run_inpaint(
    #         "/root/comfyui/data/input/13_a chair.jpg",
    #         "/root/comfyui/data/mask/13.jpg", 
    #         "a chair",
    #         "output.png",
    #         strength=1.0  # Using a more moderate strength to preserve more of the original image
    #     )