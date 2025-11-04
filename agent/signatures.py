# signatures.py
import dspy

class WoundImageClassification(dspy.Signature):
    """
    Analyze a wound image and provide detailed classification.
    
    The classification should include:
    - Wound type (laceration, abrasion, burn, puncture, etc.)
    - Severity (minor, moderate, severe)
    - Size and depth characteristics
    - Location on body
    - Notable features (bleeding, infection signs, etc.)
    """
    
    wound_image: dspy.Image = dspy.InputField(desc="Wound image to classify.")

    classification = dspy.OutputField(
        desc="Detailed wound classification including type, severity, size, location, and characteristics"
    )


class FirstAidInstructions(dspy.Signature):
    """
    Generate safe, clear, and actionable first aid instructions based on wound classification.
    
    Instructions should include:
    - Immediate actions to take
    - Step-by-step treatment procedure
    - Materials/supplies needed
    - Warning signs requiring professional medical attention
    - What NOT to do
    """
    
    wound_classification: dspy.Image = dspy.InputField(
        desc="Detailed wound classification with type, severity, and characteristics"
    )
    
    first_aid_instructions: str = dspy.OutputField(
        desc="Clear, step-by-step first aid instructions with safety warnings"
    )


class WoundAnalysisWithReasoning(dspy.Signature):
    """
    Analyze wound and provide first aid with medical reasoning.
    
    This combines classification and treatment in one step with
    explicit reasoning about the medical decision-making process.
    """
    
    wound_classification: str = dspy.InputField(
        desc="Wound classification details"
    )
    
    reasoning: str = dspy.OutputField(
        desc="Medical reasoning about wound severity and treatment approach"
    )
    
    first_aid_instructions: str = dspy.OutputField(
        desc="Step-by-step first aid instructions"
    )