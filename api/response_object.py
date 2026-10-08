from pydantic import Field, BaseModel


class Response_Object(BaseModel):
    query: str
