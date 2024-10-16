from pydantic import BaseModel, field_validator


class FluxInput(BaseModel):
    __MIN_SIZE__ = 512
    __MAX_SIZE__ = 1920
    __DEFAULT_SIZE__ = 720

    __MAX_NUM_INFERENCE_STEPS__ = 10
    __MIN_NUM_INFERENCE_STEPS__ = 1
    __DEFAULT_NUM_INFERENCE_STEPS__ = 4

    __MAX_GUIDANCE_SCALE__ = 5.0
    __MIN_GUIDANCE_SCALE__ = 0.5
    __DEFAULT_GUIDANCE_SCALE__ = 3.5

    __DEFAULT_GENERATOR_SEED__ = 12345

    prompt: str = ""
    width: int = __DEFAULT_SIZE__
    height: int = __DEFAULT_SIZE__
    num_inference_steps: int = __DEFAULT_NUM_INFERENCE_STEPS__
    generator_seed: int = __DEFAULT_GENERATOR_SEED__
    guidance_scale: float = __DEFAULT_GUIDANCE_SCALE__

    @field_validator('width', 'height')
    def validate_sizes(cls, v):
        assert (
            cls.__MIN_SIZE__ <= v <= cls.__MAX_SIZE__
        ), f"value must be in range {cls.__MIN_SIZE__} to {cls.__MAX_SIZE__}"

    @field_validator('num_inference_steps')
    def validate_num_inference_steps(cls, v):
        assert v <= 10, 'num_inference_steps must not be greater than 10'
        return v

    @field_validator('guidance_scale')
    def validate_guidance_scale(cls, v):
        assert (
            cls.__MIN_GUIDANCE_SCALE__ <= v <= cls.__MAX_GUIDANCE_SCALE__
        ), f'guidance_scale must be in range {cls.__MIN_GUIDANCE_SCALE__} to {cls.__MAX_GUIDANCE_SCALE__}'

    @field_validator('num_inference_steps')
    def validate_num_inference_steps(cls, v):
        assert cls.__MIN_NUM_INFERENCE_STEPS__ <= v <= cls.__MAX_NUM_INFERENCE_STEPS__, (
            f'num_inference_steps must be in range '
            f'{cls.__MIN_NUM_INFERENCE_STEPS__} to {cls.__MAX_NUM_INFERENCE_STEPS__}'
        )
