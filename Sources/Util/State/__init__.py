from enum import Enum

class TaskState(Enum):
    WAITING_FOR_PHOTOS = 1
    PROCESSING = 2
    WAITING_FOR_CHOICES = 3