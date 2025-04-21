import json
import requests
import time
import argparse

def test_inpaint_service(image_path, mask_path, prompt, host="localhost", port=8000):
    """
    Test the inpaint service by sending a request to the deployed Ray Serve endpoint
    """
    url = f"http://{host}:{port}/inpaint"
    
    # Prepare the request payload
    payload = {
        "prompt": prompt,
        "image_url": image_path,
        "mask_url": mask_path,
        "order_id": f"test_{int(time.time())}",
        "negative_prompt": "low quality, blurry, distortion",
        "strength": 1.0,
        "image_extension": "png"
    }
    
    print(f"Sending request to {url} with payload:")
    print(json.dumps(payload, indent=2))
    
    # Send the request
    start_time = time.time()
    response = requests.post(url, json=payload)
    end_time = time.time()
    
    print(f"Response received in {end_time - start_time:.2f} seconds")
    
    # Process the response
    try:
        result = response.json()
        print("Response:")
        print(json.dumps(result, indent=2))
        
        # If successful, show the URL to access the image
        if "output_url" in result:
            print(f"Image available at: http://{host}:{result['output_url']}")
            
        return result
    except Exception as e:
        print(f"Error parsing response: {e}")
        print(f"Response content: {response.content}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the Fooocus Inpaint service")
    parser.add_argument("--image", required=False, default = 'https://phot-user-uploads.s3.us-east-2.amazonaws.com/frontend_upload/file_drops/c102d7b2-f4c9-4801-9e56-53add7ea7743.jpg', help="Path or URL to the input image")
    parser.add_argument("--mask", required=False, default = 'https://phot-user-uploads.s3.us-east-2.amazonaws.com/base64URLs/2025-04-21/bcd26904-4175-48c7-a151-bba2dc284ac6.webp', help="Path or URL to the mask image")
    parser.add_argument("--prompt", required=False, default = 'fruits', help="Prompt for inpainting")
    parser.add_argument("--host", default="localhost", help="Host where the service is running")
    parser.add_argument("--port", default=8000, type=int, help="Port where the service is running")
    
    args = parser.parse_args()
    # for i in range(10):
    test_inpaint_service(
        args.image,
        args.mask,
        args.prompt,
        host=args.host,
        port=args.port
        ) 