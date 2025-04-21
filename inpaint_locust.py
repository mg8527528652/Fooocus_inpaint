from locust import HttpUser, task
import time
import random

class InpaintAPIUser(HttpUser):

    def on_start(self):
        self.headers = {
            'Content-Type': 'application/json'
        }  

        # Base test data
        self.test_images = [
            "https://phot-user-uploads.s3.us-east-2.amazonaws.com/frontend_upload/file_drops/bdba4be0-6361-45ac-808c-e5346c81e9d5.png",
        ]
        
        self.test_masks = [
            "https://phot-user-uploads.s3.us-east-2.amazonaws.com/base64URLs/2025-04-21/f7b33033-e335-4778-8279-4acb68585ab8.webp",
        ]
        
        self.test_prompts = [
            "a beautiful tree on a beach",
            "a cute cat",
            "a scenic landscape",
            "a modern building"
        ]

    def get_test_payload(self):
        """Generate random test data for each request"""
        return {
            "prompt": random.choice(self.test_prompts),
            "image_url": random.choice(self.test_images),
            "mask_url": random.choice(self.test_masks),
            "order_id": f"locust_test_{int(time.time())}_{random.randint(1000, 9999)}",
            "negative_prompt": "low quality, distorted, blurry",
            "strength": 1.0,
            "image_extension": "webp"
        }

    @task
    def inpaint_image(self):
        """Send request to the inpaint endpoint"""
        payload = self.get_test_payload()
        
        with self.client.post('/inpaint',
                             headers=self.headers,
                             json=payload,
                             catch_response=True) as response:
            
            if response.status_code == 200:
                try:
                    result = response.json()
                    if "output_url" in result:
                        response.success()
                    else:
                        response.failure(f"Missing output_url in response: {result}")
                except Exception as e:
                    response.failure(f"Invalid JSON response: {e}")
            else:
                response.failure(f"Failed with status code: {response.status_code}")

    # @task(weight=3)
    # def health_check(self):
    #     """Check service health - more frequent, lighter request"""
    #     with self.client.get('/', catch_response=True) as response:
    #         if response.status_code == 200:
    #             response.success()
    #         else:
    #             response.failure(f"Health check failed: {response.status_code}")

def run_load_test(users, duration):
    from subprocess import call

    # Construct the command
    cmd = f"locust -f {__file__} --headless -u {users} -r {users} --run-time {duration}s --host=http://localhost:8000"
    
    # Run the command
    call(cmd, shell=True)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run load test for Inpainting API")
    parser.add_argument("--users", type=int, default=10, help="Number of users to simulate")
    parser.add_argument("--duration", type=int, default=60, help="Duration of the test in seconds")
    parser.add_argument("--host", type=str, default="http://localhost:8000", help="Host to test against")
    
    args = parser.parse_args()
    
    print(f"Starting inpainting load test with {args.users} users for {args.duration} seconds...")
    print(f"Testing against host: {args.host}")
    run_load_test(args.users, args.duration)