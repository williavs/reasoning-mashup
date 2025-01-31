from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
import os
from dotenv import load_dotenv
import time
import re
import httpx
import logging
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Model Constants
REASONING_MODEL = "deepseek-r1:8b"  # First model for reasoning
RESPONSE_MODEL = "phi4"          # Second model for final response

# Proxy Configuration
PROXY_URL = "http://127.0.0.1:8080"  # mitmproxy default address
USE_PROXY = os.getenv("USE_PROXY", "false").lower() == "true"

def create_client_with_proxy(base_url: str, api_key: str) -> OpenAI:
    """Create an OpenAI client with proxy configuration if enabled"""
    client_params = {
        "base_url": base_url,
        "api_key": api_key,
    }
    
    if USE_PROXY:
        client_params["http_client"] = httpx.Client(
            proxies={
                "http://": PROXY_URL,
                "https://": PROXY_URL
            },
            verify=False  # This allows mitmproxy to intercept HTTPS traffic
        )
        logger.info(f"Created client for {base_url} with proxy configuration")
    else:
        logger.info(f"Created client for {base_url} without proxy")
    
    return OpenAI(**client_params)

app = FastAPI(
    title="Reasoning Mashup API (Ollama Only)",
    description="Local API for the Reasoning Mashup LLM chain using Ollama models",
    version="1.0.0"
)

class ChatRequest(BaseModel):
    message: str
    show_reasoning: bool = True
    model: str = RESPONSE_MODEL

class ChatResponse(BaseModel):
    reasoning: str
    response: str
    model: str
    elapsed_time: float
    request_ids: dict = {}  # Store request IDs for tracing

class ModelChain:
    def __init__(self):
        # Initialize client with proxy support
        self.ollama_client = create_client_with_proxy(
            base_url='http://localhost:11434/v1',
            api_key='ollama'
        )

    def get_deepseek_reasoning(self, user_input: str) -> tuple[str, str]:
        try:
            logger.info(f"Making request to Ollama for reasoning using {REASONING_MODEL}")
            response = self.ollama_client.chat.completions.create(
                model=REASONING_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": user_input}
                ]
            )
            
            request_id = getattr(response, 'id', 'unknown')
            logger.info(f"Received response from Ollama (ID: {request_id})")
            
            full_content = response.choices[0].message.content
            think_match = re.search(r'<think>(.*?)</think>', full_content, re.DOTALL)
            return (think_match.group(1).strip() if think_match else full_content), request_id
        except Exception as e:
            logger.error(f"Error in reasoning request: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error getting reasoning: {str(e)}")

    def get_final_response(self, user_input: str, reasoning: str, model: str) -> tuple[str, str]:
        try:
            logger.info(f"Making request to Ollama for final response using {model}")
            response = self.ollama_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that provides clear, accurate, and well-structured responses."},
                    {"role": "user", "content": user_input},
                    {"role": "assistant", "content": f"<thinking>{reasoning}</thinking>"},
                    {"role": "user", "content": "Based on this reasoning, please provide a clear and comprehensive response."}
                ]
            )

            request_id = getattr(response, 'id', 'unknown')
            logger.info(f"Received response from Ollama (ID: {request_id})")
            
            return response.choices[0].message.content, request_id

        except Exception as e:
            logger.error(f"Error in final response request: {str(e)}")
            logger.error(f"Full error details: {repr(e)}")
            raise HTTPException(status_code=500, detail=f"Error getting response: {str(e)}")

# Create a global instance of ModelChain
model_chain = ModelChain()

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Chat endpoint that processes messages through the dual model chain.
    
    Parameters:
    - message: The user's input message
    - show_reasoning: Whether to include reasoning in the response (default: True)
    - model: The model to use for the final response (default: RESPONSE_MODEL)
    
    Returns:
    - reasoning: The reasoning process from the first model
    - response: The final response from the second model
    - model: The model used for the final response
    - elapsed_time: Time taken to process the request
    - request_ids: Dictionary of request IDs for tracing
    """
    start_time = time.time()
    logger.info(f"Processing chat request: {request.message[:100]}...")
    
    try:
        # Get reasoning from first model (DeepSeek)
        reasoning, reasoning_id = model_chain.get_deepseek_reasoning(request.message)
        
        # Get response from second model (user-specified model)
        response, response_id = model_chain.get_final_response(
            request.message,
            reasoning,
            request.model
        )
        
        elapsed_time = time.time() - start_time
        logger.info(f"Request completed in {elapsed_time:.2f} seconds")
        
        return ChatResponse(
            reasoning=reasoning if request.show_reasoning else "",
            response=response,
            model=request.model,
            elapsed_time=elapsed_time,
            request_ids={
                "reasoning": reasoning_id,
                "response": response_id
            }
        )
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting API server...")
    uvicorn.run(app, host="0.0.0.0", port=8001) 