"""Exception types shared across the app."""


class JobCancelled(Exception):
    """Raised when the user stops a job that is still running."""
