"""How the notebooks write their figures in prose. Only the notebooks import this module."""


def note(text: str) -> None:
    """Show a text as Markdown in the output of the cell."""
    from IPython.display import Markdown, display

    display(Markdown(text))  # type: ignore[no-untyped-call]


def percent(share: float) -> str:
    """Write a share as a percentage with one decimal."""
    return f"{100 * share:.1f} %"


def count(number: float) -> str:
    """Write a whole number with spaces between its thousands."""
    return f"{int(number):,}".replace(",", " ")
