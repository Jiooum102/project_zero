import uuid
from datetime import datetime

from pydantic import BaseModel
from pydantic.networks import EmailStr


class User(BaseModel):
    email: EmailStr = None
    user_id: str = None
    created_at: datetime = datetime.now()

    def model_post_init(self, *args, **kwargs):
        if self.user_id is None:
            self.user_id = str(uuid.uuid4())
