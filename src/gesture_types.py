from enum import Enum


class Gesture(str, Enum):
    START = "START"  # Thumbs up
    STOP = "STOP"    # Open palm facing camera
    ONE = "1"
    TWO = "2"
    THREE = "3"
    FOUR = "4"
    FIVE = "5"

    @staticmethod
    def from_count(count: int):
        mapping = {
            1: Gesture.ONE,
            2: Gesture.TWO,
            3: Gesture.THREE,
            4: Gesture.FOUR,
            5: Gesture.FIVE,
        }
        return mapping.get(count)

