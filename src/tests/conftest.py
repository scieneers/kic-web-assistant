import os


def pytest_sessionstart(session):
    # src/env.py sets LANGFUSE_PUBLIC_KEY to the string "UNSET", which Langfuse
    # treats as a valid key and starts a background upload thread. Clearing it
    # here (after env.py has been imported during collection) disables Langfuse
    # before any @observe()-decorated functions are called.
    os.environ["LANGFUSE_PUBLIC_KEY"] = ""
    os.environ["LANGFUSE_SECRET_KEY"] = ""
