"""Golden dataset for evaluation."""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class QueryTestCase(BaseModel):
    """Test case for evaluating query analysis.
    
    Attributes:
        query: The customer query text.
        expected_category: Expected category classification.
        expected_sentiment: Expected sentiment analysis.
        expected_priority: Expected priority level.
        expected_escalate: Whether query should be escalated.
        notes: Optional notes about the test case.
    """
    
    query: str = Field(description="The customer query text")
    expected_category: Literal[
        "Billing",
        "Technical Support",
        "Returns & Refunds",
        "Shipping & Delivery",
        "Account Management",
        "General Inquiry"
    ] = Field(description="Expected category classification")
    expected_sentiment: Literal[
        "Positive",
        "Neutral",
        "Negative",
        "Urgent",
        "Frustrated"
    ] = Field(description="Expected sentiment analysis")
    expected_priority: Literal["Critical", "High", "Medium", "Low"] = Field(
        description="Expected priority level"
    )
    expected_escalate: bool = Field(description="Whether query should be escalated")
    notes: Optional[str] = Field(default=None, description="Optional notes about the test case")


def get_golden_dataset() -> List[QueryTestCase]:
    """Get the golden dataset with 30 test cases.
    
    Returns:
        List of QueryTestCase objects covering all categories,
        sentiments, priorities, and edge cases.
    """
    return [
        # Billing - various sentiments
        QueryTestCase(
            query="My bill this month is $50 higher than usual, please explain",
            expected_category="Billing",
            expected_sentiment="Negative",
            expected_priority="Medium",
            expected_escalate=False,
            notes="Standard billing inquiry"
        ),
        QueryTestCase(
            query="I've been charged twice for the same order! This is unacceptable!",
            expected_category="Billing",
            expected_sentiment="Frustrated",
            expected_priority="High",
            expected_escalate=True,
            notes="Duplicate charge - should escalate"
        ),
        QueryTestCase(
            query="Thank you for the quick refund processing!",
            expected_category="Billing",
            expected_sentiment="Positive",
            expected_priority="Low",
            expected_escalate=False,
            notes="Positive feedback"
        ),
        
        # Technical Support - various sentiments
        QueryTestCase(
            query="The app keeps crashing every time I try to upload a photo",
            expected_category="Technical Support",
            expected_sentiment="Negative",
            expected_priority="Medium",
            expected_escalate=False,
            notes="Bug report"
        ),
        QueryTestCase(
            query="I cannot login at all, password reset email never arrives",
            expected_category="Technical Support",
            expected_sentiment="Frustrated",
            expected_priority="High",
            expected_escalate=True,
            notes="Login issue affecting user"
        ),
        QueryTestCase(
            query="How do I change my profile picture in the settings?",
            expected_category="Technical Support",
            expected_sentiment="Neutral",
            expected_priority="Low",
            expected_escalate=False,
            notes="Simple how-to question"
        ),
        
        # Returns & Refunds
        QueryTestCase(
            query="I received a damaged product and want a full refund",
            expected_category="Returns & Refunds",
            expected_sentiment="Negative",
            expected_priority="Medium",
            expected_escalate=False,
            notes="Damaged item return"
        ),
        QueryTestCase(
            query="This is the third time I've returned the same product, nothing works!",
            expected_category="Returns & Refunds",
            expected_sentiment="Frustrated",
            expected_priority="High",
            expected_escalate=True,
            notes="Repeated issue - should escalate"
        ),
        QueryTestCase(
            query="What's your return policy for electronics?",
            expected_category="Returns & Refunds",
            expected_sentiment="Neutral",
            expected_priority="Low",
            expected_escalate=False,
            notes="Policy question"
        ),
        
        # Shipping & Delivery
        QueryTestCase(
            query="My package shows delivered but I never received it",
            expected_category="Shipping & Delivery",
            expected_sentiment="Negative",
            expected_priority="High",
            expected_escalate=True,
            notes="Lost package - should escalate"
        ),
        QueryTestCase(
            query="When will my order arrive? It's been 2 weeks",
            expected_category="Shipping & Delivery",
            expected_sentiment="Urgent",
            expected_priority="Medium",
            expected_escalate=False,
            notes="Delayed delivery inquiry"
        ),
        QueryTestCase(
            query="Can I change my shipping address before the order ships?",
            expected_category="Shipping & Delivery",
            expected_sentiment="Neutral",
            expected_priority="Low",
            expected_escalate=False,
            notes="Shipping modification question"
        ),
        
        # Account Management
        QueryTestCase(
            query="I need to update my billing address for my account",
            expected_category="Account Management",
            expected_sentiment="Neutral",
            expected_priority="Low",
            expected_escalate=False,
            notes="Simple account update"
        ),
        QueryTestCase(
            query="Someone hacked my account, please help immediately!",
            expected_category="Account Management",
            expected_sentiment="Urgent",
            expected_priority="Critical",
            expected_escalate=True,
            notes="Security issue - should escalate"
        ),
        QueryTestCase(
            query="How do I cancel my subscription? I can't find the option",
            expected_category="Account Management",
            expected_sentiment="Negative",
            expected_priority="Medium",
            expected_escalate=False,
            notes="Subscription question"
        ),
        
        # General Inquiry
        QueryTestCase(
            query="What are your business hours?",
            expected_category="General Inquiry",
            expected_sentiment="Neutral",
            expected_priority="Low",
            expected_escalate=False,
            notes="Simple information question"
        ),
        QueryTestCase(
            query="Do you offer international shipping?",
            expected_category="General Inquiry",
            expected_sentiment="Neutral",
            expected_priority="Low",
            expected_escalate=False,
            notes="General question"
        ),
        QueryTestCase(
            query="Your service is absolutely amazing, thank you so much!",
            expected_category="General Inquiry",
            expected_sentiment="Positive",
            expected_priority="Low",
            expected_escalate=False,
            notes="Compliment"
        ),
        
        # Edge cases - multilingual
        QueryTestCase(
            query="Hola, necesito ayuda con mi cuenta. No puedo iniciar sesion.",
            expected_category="Account Management",
            expected_sentiment="Negative",
            expected_priority="Medium",
            expected_escalate=False,
            notes="Spanish query - login issue"
        ),
        QueryTestCase(
            query="Bonjour, je voudrais savoir quand ma commande arrivera?",
            expected_category="Shipping & Delivery",
            expected_sentiment="Neutral",
            expected_priority="Low",
            expected_escalate=False,
            notes="French query - shipping question"
        ),
        QueryTestCase(
            query="नमस्ते, मैं अपना बिल कैसे देख सकता हूं?",
            expected_category="Billing",
            expected_sentiment="Neutral",
            expected_priority="Low",
            expected_escalate=False,
            notes="Hindi query - billing question"
        ),
        
        # Edge cases - ambiguous/sarcastic
        QueryTestCase(
            query="Oh great, another error message. Just what I needed today.",
            expected_category="Technical Support",
            expected_sentiment="Frustrated",
            expected_priority="Medium",
            expected_escalate=False,
            notes="Sarcastic but not escalated"
        ),
        QueryTestCase(
            query="Is it supposed to take 3 weeks to process a refund? Just asking for a friend.",
            expected_category="Billing",
            expected_sentiment="Frustrated",
            expected_priority="Medium",
            expected_escalate=False,
            notes="Passive aggressive but manageable"
        ),
        
        # Edge cases - short and long queries
        QueryTestCase(
            query="Help",
            expected_category="General Inquiry",
            expected_sentiment="Neutral",
            expected_priority="Low",
            expected_escalate=False,
            notes="Very short query"
        ),
        QueryTestCase(
            query="I am writing to inquire about your services. Specifically, I would like to know more about the premium tier. Additionally, I have questions about billing cycles. Furthermore, I want to understand the cancellation policy. Finally, I would appreciate information about any current promotions.",
            expected_category="General Inquiry",
            expected_sentiment="Neutral",
            expected_priority="Low",
            expected_escalate=False,
            notes="Very long query"
        ),
        
        # Edge cases - borderline categories
        QueryTestCase(
            query="My account was charged but I can't access the billing section to see what for",
            expected_category="Billing",
            expected_sentiment="Negative",
            expected_priority="Medium",
            expected_escalate=False,
            notes="Billing + Technical - tiebreak to Billing"
        ),
        QueryTestCase(
            query="I want to return a product I bought but the website won't let me",
            expected_category="Returns & Refunds",
            expected_sentiment="Negative",
            expected_priority="Medium",
            expected_escalate=False,
            notes="Returns + Technical - tiebreak to Returns"
        ),
        
        # Edge cases - urgent
        QueryTestCase(
            query="CRITICAL: System down during our business hours, need immediate help!",
            expected_category="Technical Support",
            expected_sentiment="Urgent",
            expected_priority="Critical",
            expected_escalate=True,
            notes="Critical business impact"
        ),
        QueryTestCase(
            query="Need to change shipping address before tomorrow morning. Is that possible?",
            expected_category="Shipping & Delivery",
            expected_sentiment="Urgent",
            expected_priority="High",
            expected_escalate=False,
            notes="Time-sensitive request"
        ),
        
        # More edge cases
        QueryTestCase(
            query="I'll wait as long as it takes but I need this resolved today",
            expected_category="General Inquiry",
            expected_sentiment="Urgent",
            expected_priority="High",
            expected_escalate=False,
            notes="Urgent but polite"
        ),
    ]