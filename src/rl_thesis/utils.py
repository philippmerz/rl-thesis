"""Shared utility functions."""


def format_time(seconds: float) -> str:
    """Format seconds as H:MM:SS."""
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}"
