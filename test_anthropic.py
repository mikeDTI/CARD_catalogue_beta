#!/usr/bin/env python3
import anthropic
import os
import sys

def test_anthropic():
    # Get API key from environment
    api_key = os.getenv('ANTHROPIC_API_KEY')

    if not api_key:
        print("ANTHROPIC_API_KEY environment variable not set")
        return

    print(f"Anthropic package version: {anthropic.__version__}")
    print(f"API key found: {'Yes' if api_key else 'No'}")
    
    # Check for any proxy-related environment variables
    proxy_vars = [k for k in os.environ.keys() if 'proxy' in k.lower()]
    if proxy_vars:
        print(f"Proxy environment variables found: {proxy_vars}")
        for var in proxy_vars:
            print(f"  {var}: {os.environ[var]}")

    try:
        # Try to create client with minimal arguments
        print("Attempting to create Anthropic client...")
        
        # Let's try to see what arguments the client constructor accepts
        import inspect
        sig = inspect.signature(anthropic.Anthropic.__init__)
        print(f"Anthropic client constructor signature: {sig}")
        
        client = anthropic.Anthropic(api_key=api_key, proxies=None)
        print("Client initialized successfully")
        
        # Test a simple message
        print("Testing API call...")
        response = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=100,
            messages=[
                {"role": "user", "content": "Say hello"}
            ]
        )
        
        # Extract response content (different structure in 0.18.1)
        if hasattr(response, 'content') and response.content:
            content = response.content[0].text
        elif hasattr(response, 'text'):
            content = response.text
        else:
            content = "No content found"
        
        print(f"Response: {content}")
        print("Test successful!")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        print(f"Error type: {type(e)}")
        
        # Try to get more details about the error
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_anthropic() 