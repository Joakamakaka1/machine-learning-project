"""Reproduce the audit of the inflated figures: `uv run python -m experiments.audit_inflation`.

It is the same as `hotel-ml audit`, and it refuses to run before the single execution of the test.
"""

from hotel_cancellation import config, pipeline
from hotel_cancellation.evaluation import audit


def main() -> None:
    """Run the audit and print its table."""
    table = audit.audit_inflation(pipeline.load_cleaned(), settings=config.get_settings())
    print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
