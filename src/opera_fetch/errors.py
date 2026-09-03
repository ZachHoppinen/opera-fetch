"""Exceptions the package raises on purpose."""


class NoAcquisitions(ValueError):
    """These files hold no acquisitions to read.

    Its own type because the loop that skips an empty burst has to skip only that. Every
    other error a reader raises means the data is wrong rather than absent, and a
    misaligned stack that gets swallowed as "nothing here" disappears with one warning.
    """
