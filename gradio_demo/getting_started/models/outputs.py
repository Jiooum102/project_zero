from pydantic import BaseModel


class FluxOutput(BaseModel):
    output_url: str = None
    output_path: str = None
    output_record_id: str = None
