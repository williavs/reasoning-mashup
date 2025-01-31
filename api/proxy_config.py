from typing import Optional, Callable
import requests
import logging
import httpx
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class ProxyMonitoring:
    """Configuration for mitmproxy monitoring"""
    MITMPROXY_URL = "http://127.0.0.1:8080"
    USE_PROXY = os.getenv("USE_PROXY", "false").lower() == "true"
    
    @staticmethod
    def get_client_config():
        """Get httpx client configuration with proxy if enabled"""
        if ProxyMonitoring.USE_PROXY:
            return {
                "proxies": {
                    "http://": ProxyMonitoring.MITMPROXY_URL,
                    "https://": ProxyMonitoring.MITMPROXY_URL
                },
                "verify": False  # Allows mitmproxy to intercept HTTPS traffic
            }
        return {}

class ProxyAPIConfig:
    """Configuration for different proxy APIs"""
    OPENROUTER = "openrouter"
    OLLAMA = "ollama"
    
    @staticmethod
    def get_api_url(api_type: str) -> str:
        """Get the API URL based on type"""
        if api_type == ProxyAPIConfig.OPENROUTER:
            return "http://localhost:8000"  # OpenRouter proxy
        elif api_type == ProxyAPIConfig.OLLAMA:
            return "http://localhost:8001"  # Ollama proxy
        else:
            raise ValueError(f"Unknown API type: {api_type}")

class ProxyAPITool:
    """Tool to interact with configurable proxy APIs"""
    def __init__(self, api_type: str = "openrouter", logger: Optional[Callable] = None):
        self.api_type = api_type
        self.api_url = ProxyAPIConfig.get_api_url(api_type)
        self.logger = logger or print
    
    async def execute(self, input_data: str) -> dict:
        try:
            self.logger(f"🔄 Sending request to {self.api_type} proxy API...")
            
            # Both proxies expect the same format
            payload = {
                "message": input_data,
                "show_reasoning": True,
                "model": "anthropic/claude-3-sonnet:beta" if self.api_type == ProxyAPIConfig.OPENROUTER else "deepseek-r1:8b"
            }
            
            # Get client config with proxy if enabled
            client_config = ProxyMonitoring.get_client_config()
            
            async with httpx.AsyncClient(**client_config) as client:
                response = await client.post(
                    f"{self.api_url}/chat",
                    json=payload,
                    timeout=None
                )
                
                if response.status_code == 200:
                    self.logger(f"✅ Received response from {self.api_type} proxy API")
                    return response.json()
                else:
                    self.logger(f"⚠️ API returned status code: {response.status_code}")
                    self.logger(f"Response content: {response.text}")
                    return None
                
        except Exception as e:
            self.logger(f"⚠️ Error calling {self.api_type} proxy API: {str(e)}")
            return None 