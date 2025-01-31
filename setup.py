from setuptools import setup, find_packages

setup(
    name="research-mashup-api",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "requests",
        "python-dotenv",
        "duckduckgo_search",
        "honcho",
        "httpx",
        "fastapi",
        "uvicorn",
        "streamlit"
    ],
) 