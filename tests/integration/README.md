# Integration tests

Tests that talk to something real — Firestore, Vertex AI, the Him Cab Spring backend,
or the live geocoding/routing providers. They are **not** run by default.

`tests/conftest.py` installs an autouse `no_network` fixture, so anything placed here
must opt out of it explicitly. Mark every test in this directory:

```python
pytestmark = pytest.mark.integration
```

and run them with `pytest -m integration` (the marker is declared in `pyproject.toml`).
CI runs `-m "not integration"`, so these never gate a pull request — they need real
credentials and somebody else's uptime, neither of which belongs in a merge check.

For one-off diagnosis against the live backend, prefer the scripts in `scripts/`
(`check_fare.py`, `check_rental.py`); they name the step that broke rather than
asserting.
