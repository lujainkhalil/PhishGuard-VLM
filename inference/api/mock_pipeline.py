"""Mock pipeline for UI demo when model is not available."""
import random
import time
from dataclasses import dataclass

@dataclass  
class MockResult:
    label: int
    confidence: float
    explanation: str
    final_url: str
    crawl_status: str
    model_probability: float
    phishing_probability: float
    knowledge_used: bool
    aggregated: None = None
    cross_modal_consistency: float = None
    cross_modal: dict = None

class MockInferencePipeline:
    def analyze(self, url: str, **kwargs) -> MockResult:
        time.sleep(2)  # simulate crawl time
        
        phishing_keywords = ['paypal-secure', 'verify', 'login-confirm', 
                             'account-update', 'signin', 'secure-bank']
        is_phish = any(kw in url.lower() for kw in phishing_keywords)
        
        if is_phish:
            confidence = round(random.uniform(0.82, 0.98), 3)
            label = 1
            explanation = (
                f"PHISHING DETECTED: The URL '{url}' exhibits multiple phishing indicators. "
                "Visual analysis shows brand impersonation in the page layout. "
                "The domain does not match the identified brand's legitimate domains. "
                "Cross-modal inconsistency detected between visual branding and URL structure."
            )
        else:
            confidence = round(random.uniform(0.75, 0.96), 3)
            label = 0
            explanation = (
                f"BENIGN: The URL '{url}' appears legitimate. "
                "Visual and textual analysis shows consistent branding. "
                "Domain matches expected patterns for this type of website. "
                "No cross-modal inconsistencies detected."
            )
        
        return MockResult(
           label=label,
           confidence=confidence,
           explanation=explanation,
           final_url=url,
           crawl_status="ok",
           model_probability=confidence if label == 1 else 1 - confidence,
           phishing_probability=confidence if label == 1 else 1 - confidence,
           knowledge_used=True,
           cross_modal_consistency=0.23 if is_phish else 0.91,
           cross_modal={
              "text_brands": ["PayPal"] if is_phish else [],
              "domain_registrable": url.split("/")[2] if "://" in url else url,
              "notes": "Brand-domain mismatch detected" if is_phish else "Consistent branding"
              }
        )