"""
Fal AI MCP Server
"""

from mcp.server.fastmcp import FastMCP
from pydantic import Field

import mcp.types as types
import asyncio
import logging
import os
from typing import Dict, Optional
import fal_client
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mcp = FastMCP("Fal AI Image Generator", port=3000, stateless_http=True, debug=True)

# Configuration
FAL_API_KEY = os.getenv("FAL_KEY")
if not FAL_API_KEY:
    logger.warning("FAL_KEY environment variable not set. Make sure to configure it.")

# Model endpoint mappings
MODEL_ENDPOINTS = {
    "seedream-v4": "fal-ai/bytedance/seedream/v4/text-to-image",
    "imagen4": "fal-ai/imagen4/preview",
    "seedream": "fal-ai/bytedance/seedream/v4/text-to-image",  # alias
    "imagen": "fal-ai/imagen4/preview"  # alias
}

# Default parameters for each model
MODEL_DEFAULTS = {
    "seedream-v4": {
        "num_inference_steps": 50,
        "guidance_scale": 7.5,
        "image_size": {"width": 1024, "height": 1024}
    },
    "imagen4": {
        "num_inference_steps": 40,
        "guidance_scale": 4.0,
        "image_size": "1024x1024"
    }
}

@mcp.tool(
    title="Generate Image",
    description="Generate images using Fal AI models. Supports SeeDream v4 (artistic/creative) and Imagen4 (photorealistic). Returns the image URL."
)
async def generate_image(
    prompt: str = Field(description="The text prompt describing the image you want to generate"),
    model: str = Field(description="The model to use: 'seedream-v4' (artistic), 'imagen4' (photorealistic), 'seedream' (alias), or 'imagen' (alias)")
) -> Dict[str, str]:
    """
    Generate an image using the specified Fal AI model.
    
    Input format:
    {
        "prompt": "your image description here",
        "model": "seedream-v4" or "imagen4"
    }
    
    Output format:
    {
        "img_output": "https://generated-image-url.com/image.jpg"
    }
    """
    try:
        # Validate model
        model_key = model.lower()
        if model_key not in MODEL_ENDPOINTS:
            available_models = list(MODEL_ENDPOINTS.keys())
            return {
                "img_output": f"Error: Invalid model '{model}'. Available models: {', '.join(available_models)}"
            }
        
        # Get model endpoint and defaults
        endpoint = MODEL_ENDPOINTS[model_key]
        defaults = MODEL_DEFAULTS.get(model_key, MODEL_DEFAULTS["seedream-v4"])
        
        # Prepare arguments based on model type
        arguments = {
            "prompt": prompt,
            **defaults
        }
        
        logger.info(f"Starting {model} generation with prompt: {prompt[:50]}...")
        
        # Submit request to Fal AI
        handler = await fal_client.submit_async(endpoint, arguments=arguments)
        
        # Process events (optional logging)
        async for event in handler.iter_events(with_logs=True):
            logger.debug(f"Event: {event}")
        
        # Get final result
        result = await handler.get()
        
        # Extract image URL from result
        if result and "images" in result and len(result["images"]) > 0:
            image_url = result["images"][0]["url"]
            logger.info(f"Successfully generated image: {image_url}")
            return {"img_output": image_url}
        else:
            logger.error("No image data in response")
            return {"img_output": "Error: No image generated - empty response from API"}
            
    except Exception as e:
        error_message = f"Error generating image: {str(e)}"
        logger.error(error_message)
        return {"img_output": f"Error: {str(e)}"}

@mcp.resource(
    uri="fal://info/models",
    description="Information about available Fal AI models",
    name="Available Models",
)
def get_models_info() -> str:
    """Get information about all available models"""
    return """
# Available Fal AI Models

## SeeDream v4 (Bytedance)
- **Model ID**: `seedream-v4` or `seedream`
- **Endpoint**: fal-ai/bytedance/seedream/v4/text-to-image
- **Best for**: Artistic, creative, stylized images
- **Strengths**: Creative interpretation, artistic flair, complex scenes
- **Default settings**: 50 steps, 7.5 guidance scale

## Imagen4 (Google) 
- **Model ID**: `imagen4` or `imagen`
- **Endpoint**: fal-ai/imagen4/preview
- **Best for**: Photorealistic, detailed images
- **Strengths**: Photorealism, fine details, accurate representations
- **Default settings**: 40 steps, 4.0 guidance scale

## Usage Examples

### Artistic Generation (SeeDream v4)
"""


if __name__ == "__main__":
    mcp.run(transport="streamable-http")