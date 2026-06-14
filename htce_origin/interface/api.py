"""Short operator API interface definitions."""
ALLOWED_ENDPOINTS = (
    "/wake", "/observe", "/think", "/plan", "/act_simulated",
    "/sleep", "/trace", "/health", "/snapshot", "/restore",
)

def is_allowed_endpoint(path: str) -> bool:
    return path in ALLOWED_ENDPOINTS
