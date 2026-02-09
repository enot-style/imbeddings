from typing import List, Optional, Union, Literal

from pydantic import BaseModel, Field, StrictStr, field_validator


EmbeddingInput = Union[
    StrictStr,
    List[StrictStr],
]


class EmbeddingRequest(BaseModel):
    input: EmbeddingInput = Field(
        ...,
        description="Input image(s) as URL or base64 string(s)",
    )
    model: str = Field(..., description="Model ID", min_length=1)
    encoding_format: Optional[Literal["float", "base64"]] = Field(
        "float",
        description="Return format for embeddings",
    )
    dimensions: Optional[int] = Field(
        None,
        description="Optional embedding size override (truncation)",
        gt=0,
    )
    user: Optional[str] = Field(
        None,
        description="End-user identifier (passed through)",
    )

    @field_validator("encoding_format", mode="before")
    @classmethod
    def _coerce_encoding_format(cls, value):
        if value in (None, ""):
            return "float"
        return value


class EmbeddingItem(BaseModel):
    object: str = "embedding"
    index: int
    embedding: Union[List[float], str]


class Usage(BaseModel):
    prompt_tokens: int
    total_tokens: int


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[EmbeddingItem]
    model: str
    usage: Usage
