"""Request schemas shared by the API routers."""

from pydantic import BaseModel, Field


class NewConversation(BaseModel):
    corpus_id: str | None = None


class NewTurn(BaseModel):
    question: str = Field(min_length=1, max_length=4000)


class NewBranchTurn(NewTurn):
    conversation_id: str


class IngestRequest(BaseModel):
    folder: str


class TableQuery(BaseModel):
    document_id: str
    sheet: str | None = None
    filters: dict[str, str] = Field(default_factory=dict)
    operation: str = "rows"
    column: str | None = None
    header_row: int = Field(default=1, ge=1, le=100)
