from pydantic import BaseModel, Field, field_validator
class SupportRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    @field_validator("message")
    @classmethod
    def validate_message(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("Message cannot be empty.")

        return value
class ClassificationResponse(BaseModel):
    message: str
    intent: str
    confidence: float
class RetrievalCase(BaseModel):
    customer_text: str
    support_text: str
    similarity: float
class RetrievalResponse(BaseModel):
    message: str
    cases: list[RetrievalCase]
class SupportResponse(BaseModel):
    customer_message: str
    intent: str
    classifier_confidence: float
    top_retrieval_similarity: float | None
    decision: str
    decision_reason: str
    response: str | None