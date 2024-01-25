from pydantic import BaseModel


class ImageModel(BaseModel):
    image_bytes: str
