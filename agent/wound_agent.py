# wound_care_program.py
import dspy

from modules import WoundClassifierModule, FirstAidResponderModule


class WoundCarePipeline(dspy.Module):
    """
    Complete end-to-end wound care pipeline
    
    Pipeline:
    1. Image → Vision Model (C-RADIOv2) → Wound Classification
    2. Classification → LLM (Qwen GRPO) → First Aid Instructions
    """
    
    def __init__(self):
        super().__init__()
        
        # ONLY CHANGE HERE WITH ACTUAL LM'S WE ARE USING!!
        # https://dspy.ai/learn/programming/language_models/#__tabbed_1_6
        self.vlm = dspy.LM("openai/gpt-5-mini", temperature=1.0, max_tokens=16000)
        self.lm = dspy.LM("openai/gpt-5-mini", temperature=1.0, max_tokens=16000)

        self.classifier = WoundClassifierModule()
        self.responder = FirstAidResponderModule()
        
    
    def forward(self, image_path: str) -> dict:
        """
        Complete pipeline execution
        
        Args:
            image_path: Path to wound image
        
        Returns:
            Dictionary with classification and first aid instructions
        """
        image = dspy.Image(url=image_path)
          
        # 1. WOUND CLASSIFICATION
        with dspy.context(lm=self.vlm):
            wound_classification = self.classifier(
                wound_image=image
            )
        
        # 2. FIRST AID HELP
        with dspy.context(lm=self.lm):
            first_aid = self.responder(
                wound_classification=wound_classification.classification
            )
        
        return {
            "classification": wound_classification.classification,
            "first_aid": first_aid.first_aid_instructions,
            "raw_result": first_aid
        }


def test_pipeline():
    """Test the complete pipeline"""
    
    # Initialize pipeline
    pipeline = WoundCarePipeline()
    
    # Test image
    image_path = "/Users/johnkim/Downloads/wound_image_test_1.png"
    
    # Run pipeline
    result = pipeline(image_path=image_path)
    
    # Display results
    print("="*50)
    print("WOUND CLASSIFICATION:")
    print("="*50)
    print(result["classification"])
    print("\n")
    
    if "reasoning" in result:
        print("="*50)
        print("MEDICAL REASONING:")
        print("="*50)
        print(result["reasoning"])
        print("\n")
    
    print("="*50)
    print("FIRST AID INSTRUCTIONS:")
    print("="*50)
    print(result["first_aid"])
    print("\n")


if __name__ == "__main__":
    test_pipeline()