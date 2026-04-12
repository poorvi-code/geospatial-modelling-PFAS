from typing import List, Optional, Union, Dict, Any
from pydantic import BaseModel, Field, validator
from datetime import datetime

class PFASValue(BaseModel):
    cas_id: Optional[str] = None
    substance: Optional[str] = None
    isomer: Optional[str] = None
    value: Optional[float] = None
    less_than: Optional[float] = None
    unit: Optional[str] = None

class PDHPoint(BaseModel):
    id: Union[int, str] = Field(..., alias="id")
    dataset_id: Optional[int] = None
    dataset_name: Optional[str] = None
    category: Optional[str] = None # e.g., 'Biota', 'Water', 'Soil'
    type: Optional[str] = None
    sector: Optional[str] = None
    name: Optional[str] = None
    lat: float
    lon: float
    city: Optional[str] = None
    country: Optional[str] = None
    date: Optional[str] = None
    year: Optional[int] = None
    matrix: Optional[str] = None
    unit: Optional[str] = None
    pfas_sum: Optional[float] = None
    pfas_values: List[PFASValue] = []
    
    # Metadata / Attribution
    source_text: Optional[str] = None
    source_url: Optional[str] = None
    source_type: Optional[str] = None
    data_collection_method: Optional[str] = None
    
    # Internal metadata
    ingestion_timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        populate_by_name = True

class PDHExportResponse(BaseModel):
    count: int
    results: List[PDHPoint]
