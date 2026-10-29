from pydantic import BaseModel, Field


class Prediction(BaseModel):
    label: str
    score: float = Field(ge=0, le=1)


class Timing(BaseModel):
    queue_ms: float
    preprocess_ms: float
    inference_ms: float
    total_ms: float


class PredictResponse(BaseModel):
    request_id: str
    model: str
    predictions: list[Prediction]
    batch_size: int
    timing: Timing
