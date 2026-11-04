"""HTTP client of the API. The interface never loads the model by itself."""

from collections.abc import Mapping
from typing import Any

import httpx
import streamlit as st

from hotel_cancellation import config

_TIMEOUT_SECONDS = 8.0
_API_PREFIX = "/api/v1"
_UNPROCESSABLE = 422


class ApiUnavailableError(RuntimeError):
    """The API did not answer, or it failed. The reason stays out of what the user reads."""


class BookingRejectedError(ValueError):
    """The API refused the booking. `errors` holds what it said about each field."""

    def __init__(self, errors: list[dict[str, Any]]) -> None:
        """Keep the validation errors."""
        self.errors = errors
        super().__init__(f"The API rejected the booking with {len(errors)} errors.")


class ApiClient:
    """Talks to the API of this project."""

    def __init__(self, base_url: str, *, transport: httpx.BaseTransport | None = None) -> None:
        """Prepare the connection.

        Args:
            base_url: Address of the API, without the version prefix.
            transport: Replaces the network, for tests.
        """
        # The API runs on this machine, so a system proxy must not get in the way.
        self._http = httpx.Client(
            base_url=base_url.rstrip("/") + _API_PREFIX,
            timeout=_TIMEOUT_SECONDS,
            transport=transport,
            trust_env=False,
        )

    def _request(self, method: str, path: str, **options: Any) -> httpx.Response:
        try:
            return self._http.request(method, path, **options)
        except httpx.HTTPError as error:
            raise ApiUnavailableError(str(error)) from error

    def health(self) -> dict[str, Any]:
        """Ask for the state of the service.

        Raises:
            ApiUnavailableError: If it does not answer or does not answer well.
        """
        response = self._request("GET", "/health")
        if not response.is_success:
            raise ApiUnavailableError(f"Health check answered {response.status_code}.")
        content: dict[str, Any] = response.json()
        return content

    def predict(self, booking: Mapping[str, Any]) -> dict[str, Any]:
        """Ask for the prediction of one booking.

        Raises:
            ApiUnavailableError: If it does not answer or fails.
            BookingRejectedError: If it refuses the booking.
        """
        response = self._request("POST", "/predict", json=dict(booking))
        if response.status_code == _UNPROCESSABLE:
            raise BookingRejectedError(response.json()["detail"])
        if not response.is_success:
            raise ApiUnavailableError(f"Prediction answered {response.status_code}.")
        content: dict[str, Any] = response.json()
        return content


@st.cache_resource(show_spinner=False)
def default_client() -> ApiClient:
    """Build the client for the address in the settings, shared across sessions."""
    return ApiClient(config.get_settings().api_url)
