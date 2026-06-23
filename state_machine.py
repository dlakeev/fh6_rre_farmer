from enum import Enum, auto


class State(Enum):
    WAITING_FOR_TEMPLATE = auto()
    TEMPLATE_SEEN = auto()


class StateMachine:
    def __init__(
        self,
        confidence_threshold: float = 0.9,
        template_name: str = "autopilot_off",
    ):
        self.state = State.WAITING_FOR_TEMPLATE
        self.confidence_threshold = confidence_threshold
        self.template_name = template_name
        self.last_detection = None

    def update(self, detection: dict | None) -> State:
        self.last_detection = detection

        template_seen = (
            detection is not None
            and detection["name"] == self.template_name
            and detection["confidence"] >= self.confidence_threshold
        )

        if template_seen:
            self.state = State.TEMPLATE_SEEN
        else:
            self.state = State.WAITING_FOR_TEMPLATE

        return self.state
