from pydantic import BaseModel


class FluxOutput(BaseModel):
    output_url: str = ""
    output_path: str = ""
    output_record_id: str = ""
