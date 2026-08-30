import re

MAX_NAME_LEN = 200
MAX_SUMMARY_LEN = 5000
MAX_TAG_LEN = 50
MAX_PLANNING_INFO_LEN = 8000
MAX_REVIEW_NOTE_LEN = 500
MAX_REVIEW_NOTES_COUNT = 100
SHORT_HASH_RE = re.compile(r'^[0-9a-f]{8}$', re.IGNORECASE)


class AmbiguousHashError(Exception):
    def __init__(self, count: int):
        self.count = count
        super().__init__("Ambiguous hash: use ^^ticket to find the task")
