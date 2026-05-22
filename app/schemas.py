#schemas.py
from typing import Literal
from pydantic import BaseModel, Field


class RoworientedInput(BaseModel):

    num_passengers: int = Field(..., ge=1, le=20)
    purchase_lead: int = Field(..., ge=0, le=1000)
    length_of_stay: int = Field(..., ge=0, le=1000)
    flight_hour: int = Field(..., ge=0, le=23)
    flight_duration: float = Field(..., ge=0.0, le=24.0)

    wants_extra_baggage: Literal[0, 1]
    wants_preferred_seat: Literal[0, 1]
    wants_in_flight_meals: Literal[0, 1]

    sales_channel: Literal["Internet", "Mobile"]
    trip_type: Literal["RoundTrip", "CircleTrip", "OneWay"]
    flight_day: Literal["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    route: str = Field(..., min_length=1, max_length=20)
    booking_origin: str = Field(..., min_length=1, max_length=100)

class ColumnorientedInput(BaseModel):
    
    num_passengers: list[int]
    purchase_lead: list[int]
    length_of_stay: list[int]
    flight_hour: list[int]
    flight_duration: list[float]

    wants_extra_baggage: list[Literal[0, 1]]
    wants_preferred_seat: list[Literal[0, 1]]
    wants_in_flight_meals: list[Literal[0, 1]]

    sales_channel: list[Literal["Internet", "Mobile"]]
    trip_type: list[Literal["RoundTrip", "CircleTrip", "OneWay"]]
    flight_day: list[Literal["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]]
    route: list[str]
    booking_origin: list[str]


class BusinessLogicRoworiented(BaseModel):
    priority_score: int
    value_tier: Literal["High", "Medium", "Low"]
    expected_value_usd: float
    potential_revenue_usd: float
    marginal_profit_usd: float

class BusinessLogicColumnoriented(BaseModel):
    priority_score: list[int]
    value_tier: list[Literal["High", "Medium", "Low"]]
    expected_value_usd: list[float]
    potential_revenue_usd: list[float]
    marginal_profit_usd: list[float]


class Meta(BaseModel):
    model_version: str
    threshold_used: float

class _Roworiented(BaseModel):
    probability: float
    booking_prediction: int
    business_logic: BusinessLogicRoworiented

class _Columnoriented(BaseModel):
    probability: list[float]
    booking_prediction: list[int]
    business_logic: BusinessLogicColumnoriented

class PredictionRoworiented(BaseModel):
    predictions: list[_Roworiented]
    meta: Meta

class PredictionColumnoriented(BaseModel):
    predictions: _Columnoriented
    meta: Meta


class RAGGenerateRequest(BaseModel):
    customer_id: str
    customer_name: str
    email: str
    route: str
    booking_origin: str
    haul_type: str
    num_passengers: int
    wants_extra_baggage: bool
    wants_preferred_seat: bool
    wants_in_flight_meals: bool


class RAGGenerateResponse(BaseModel):
    subject: str
    body: str
    retrieved_sources: list[str]
    system_prompt_id: str
    tokens_input: int
    tokens_output: int
    latency_ms: int


class RAGFeedbackRequest(BaseModel):
    customer_id: str
    customer_name: str | None = None
    email: str | None = None
    route: str | None = None
    booking_origin: str | None = None
    haul_type: str | None = None
    num_passengers: int | None = None
    wants_extra_baggage: bool | None = None
    wants_preferred_seat: bool | None = None
    wants_in_flight_meals: bool | None = None
    retrieved_sources: list[str]
    system_prompt_id: str
    generated_subject: str
    generated_body: str
    edited_subject: str
    edited_body: str
    rating: int = Field(..., ge=1, le=5)
    accepted: bool | None = None
    tokens_input: int = 0
    tokens_output: int = 0
    latency_ms: int = 0


class RAGFeedbackResponse(BaseModel):
    status: str
    message: str