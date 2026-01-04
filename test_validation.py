
import json
from promptforge_core import DraftPrompt

def test_validation_robustness():
    print("🧪 Testing DraftPrompt Validation Robustness...")
    
    # Simulate the "dirty" payload from the crash log
    dirty_data = {
        "persona": {"name": "User", "interest": "Mac Mini", "context": "home user"},
        "context": {"product": "Mac Mini", "details": "student budget", "timeline": "one month"},
        "task": "Help me buy a mac mini",
        "output_requirements": {"recommendations": ["Apple Refurbished", "Amazon", "eBay"]},
        "permission_to_fail": True
    }
    
    try:
        draft = DraftPrompt(**dirty_data)
        print("\n✅ SUCCESS: DraftPrompt initialized without error despite structured input!")
        
        print("\nField Previews (Flattened):")
        print(f"PERSONA: {draft.persona[:100]}...")
        print(f"CONTEXT: {draft.context[:100]}...")
        print(f"OUTPUT REQS: {draft.output_requirements[:100]}...")
        
        # Verify they are strings
        assert isinstance(draft.persona, str)
        assert isinstance(draft.context, str)
        assert isinstance(draft.output_requirements, str)
        
    except Exception as e:
        print(f"\n❌ FAILURE: DraftPrompt still crashed: {e}")

if __name__ == "__main__":
    test_validation_robustness()
