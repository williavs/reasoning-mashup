from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
import os
from dotenv import load_dotenv
import time
import re
import httpx
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Model Constants
OLLAMA_MODEL = "deepseek-r1:8b"
CLAUDE_MODEL = "anthropic/claude-3.5-sonnet"

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
    title="Reasoning Mashup API",
    description="Local API for the Reasoning Mashup LLM chain",
    version="1.0.0"
)

class ChatRequest(BaseModel):
    message: str
    show_reasoning: bool = True
    model: str = CLAUDE_MODEL

class ChatResponse(BaseModel):
    reasoning: str
    response: str
    model: str
    elapsed_time: float
    request_ids: dict = {}  # Store request IDs for tracing

class ModelChain:
    def __init__(self):
        # Initialize clients with proxy support
        self.ollama_client = create_client_with_proxy(
            base_url='http://localhost:11434/v1',
            api_key='ollama'
        )

        self.openrouter_client = create_client_with_proxy(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY")
        )

    def get_deepseek_reasoning(self, user_input: str) -> tuple[str, str]:
        try:
            logger.info("Making request to Ollama for reasoning")
            response = self.ollama_client.chat.completions.create(
                model=OLLAMA_MODEL,
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
            logger.error(f"Error in Ollama request: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error getting reasoning: {str(e)}")

    def get_claude_response(self, user_input: str, reasoning: str, model: str) -> tuple[str, str]:
        try:
            logger.info("Making request to OpenRouter for response")
            messages = [
                {
                    "role": "user",
                    "content": user_input
                },
                {
                    "role": "assistant",
                    "content": f"<thinking>{reasoning}</thinking>"
                }
            ]

            logger.info(f"Using model: {model}")
            response = self.openrouter_client.chat.completions.create(
                model=model,
                messages=messages,
                stream=False,
                max_tokens=8000
            )

            # Add detailed response logging
            logger.info(f"OpenRouter raw response: {response}")
            
            if not hasattr(response, 'choices') or not response.choices:
                logger.error(f"Invalid response format from OpenRouter: {response}")
                raise ValueError("Invalid response format from OpenRouter")

            request_id = getattr(response, 'id', 'unknown')
            logger.info(f"Received response from OpenRouter (ID: {request_id})")
            
            return response.choices[0].message.content, request_id
        except Exception as e:
            logger.error(f"Error in OpenRouter request: {str(e)}")
            logger.error(f"Full error details: {repr(e)}")
            raise HTTPException(status_code=500, detail=f"Error getting response: {str(e)}")

# Create a global instance of ModelChain
model_chain = ModelChain()

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Chat endpoint that processes messages through the reasoning chain.
    
    Parameters:
    - message: The user's input message
    - show_reasoning: Whether to include reasoning in the response (default: True)
    - model: The model to use for the final response (default: CLAUDE_MODEL)
    
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
        # Get reasoning from first model
        reasoning, reasoning_id = model_chain.get_deepseek_reasoning(request.message)
        
        # Get response from second model
        response, response_id = model_chain.get_claude_response(
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
    uvicorn.run(app, host="0.0.0.0", port=8000) 