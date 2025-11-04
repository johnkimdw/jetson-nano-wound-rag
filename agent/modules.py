# modules.py
import dspy

from signatures import WoundImageClassification, FirstAidInstructions, WoundAnalysisWithReasoning

class WoundClassifierModule(dspy.Module):
    """
    Module 1: Takes image features and classifies the wound
    """
    def __init__(self):
        super().__init__()
        self.classify = dspy.Predict(WoundImageClassification)
    
    def forward(self, wound_image) -> dspy.Prediction:
        """
        Args:
            image_description: Text description or features from vision model
        
        Returns:
            Prediction with wound classification
        """
        return self.classify(wound_image=wound_image)


class FirstAidResponderModule(dspy.Module):
    """
    Module 2: Generates first aid instructions from wound classification
    """
    def __init__(self):
        super().__init__()
        self.respond = dspy.Predict(FirstAidInstructions)
    
    def forward(self, wound_classification: str) -> dspy.Prediction:
        """
        Args:
            wound_classification: Detailed wound classification
        
        Returns:
            Prediction with first aid instructions
        """
        return self.respond(wound_classification=wound_classification)


# class MedicalReasoningModule(dspy.Module):
#     """
#     Alternative: Combined module using GRPO model's reasoning capability
#     """
#     def __init__(self):
#         super().__init__()
#         self.analyze = dspy.ChainOfThought(WoundAnalysisWithReasoning)
    
#     def forward(self, wound_classification: str) -> dspy.Prediction:
#         """
#         Uses model's built-in reasoning (from GRPO training)
#         """
#         return self.analyze(wound_classification=wound_classification)