from datetime import datetime
from pydantic import BaseModel, Field


class KBMetadata(BaseModel):
    """Represents a Knowledge Base entry in the registry.

    Attributes:
        kb_id (str): Unique identifier for the Knowledge Base.
        name (str): Human-readable name.
        description (str): Summary of topics/domain.
        created_at (datetime): Timestamp of creation.
    """

    kb_id: str = Field(..., description="Unique identifier for the KB")
    name: str = Field(..., description="Human-readable name")
    description: str = Field(..., description="Summary of topics/domain")
    created_at: datetime = Field(
        default_factory=datetime.now, description="Timestamp of creation"
    )
