#!/usr/bin/env python3
"""Quick audio test"""
import os
import asyncio
from dotenv import load_dotenv

load_dotenv()

async def quick_test():
    api_key = os.getenv("ELEVENLABS_API_KEY")
    
    if not api_key or api_key == "your-api-key-here":
        print("❌ Please add your ElevenLabs API key to .env first!")
        return
    
    print("✅ API key found!")
    print("\nTesting basic imports...")
    
    try:
        from cocoa_audio import AudioCognition
        print("✅ Audio system imports successfully!")
        
        # Initialize
        audio = AudioCognition(api_key)
        print("✅ Audio cognition initialized!")
        
        print("\n✨ All tests passed! Audio system is ready.")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please ensure cocoa_audio.py is properly installed")
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    asyncio.run(quick_test())
