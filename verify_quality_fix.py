
import asyncio
import httpx
import json

async def test_quality_fix():
    url = "http://127.0.0.1:8000/api/v2/agent/run"
    
    payload = {
        "task": "help me buy a used mac mini",
        "interactive_mode": True,
        "answers": {
            "What specific model of the Mac do you have in mind?": "Mac mini only—it's compact and fits my desk setup perfectly.",
            "What will be the primary use?": "Personal use: schoolwork, note-taking, web browsing, streaming Netflix/YouTube, and light programming for classes.",
            "What is your budget range?": "$500–$900 max.",
            "Do you have a preference for new, refurbished, or used?": "Refurbished or used—I'm a student on a tight budget.",
            "Are there specific specifications?": "At least M4 chip, 16GB RAM, and 256GB–512GB storage. Needs to handle multiple browser tabs and Zoom calls smoothly.",
            "Will you need any accessories?": "Maybe a cheap monitor, keyboard, and mouse bundle if available.",
            "Where would you like to purchase?": "Apple's refurbished site, Amazon, or eBay.",
            "How soon do you need it?": "Before the new semester starts in about a month.",
            "Any aesthetic preferences?": "Standard silver/space gray is fine."
        }
    }
    
    print("🚀 Sending detailed payload to Local Server...")
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(url, json=payload)
            result = resp.json()
            
            print("\n" + "="*50)
            print("STATUS:", result.get("status"))
            print("--- FULL PROMPT ---")
            prompt = result.get("final_prompt", "NO PROMPT")
            print(prompt)
            print("="*50)
            
            # Refined check
            has_specs = "M4" in prompt or "16GB" in prompt or "RAM" in prompt
            has_budget = "$500" in prompt or "900" in prompt
            has_persona = "student" in prompt.lower()
            
            if has_specs and has_budget and has_persona:
                print("\n✅ SUCCESS: The prompt correctly incorporates user details!")
            else:
                print("\n❌ FAILURE: Some details are missing.")
                
    except Exception as e:
        print(f"❌ Error connecting to server: {e}")

if __name__ == "__main__":
    asyncio.run(test_quality_fix())
