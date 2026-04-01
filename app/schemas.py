from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


class BookingInput(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "num_passengers": 2,
                "purchase_lead": 13,
                "length_of_stay": 7,
                "flight_hour": 10,
                "flight_duration": 8.5,
                "wants_extra_baggage": 1,
                "wants_preferred_seat": 0,
                "wants_in_flight_meals": 1,
                "sales_channel": "Internet",
                "trip_type": "RoundTrip",
                "flight_day": "Sat",
                "route": "AKLHND",
                "booking_origin": "Australia",
            }
        }
    )

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


class BusinessLogic(BaseModel):
    segment: str
    category: Literal["Category 0", "Category 1", "Category 2", "Category 3"]
    recommended_action: str
    value_tier: Literal["Low", "Medium", "High"]
    expected_value_usd: float
    potential_revenue_usd: float
    marginal_profit_usd: float
    priority_score: int


class PredictionMeta(BaseModel):
    model_version: str
    threshold_used: float


class PredictionResponse(BaseModel):
    probability: float
    booking_prediction: int
    business_logic: BusinessLogic
    meta: PredictionMeta