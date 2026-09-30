"""
app/repositories/
──────────────────
Persistence. The only place that knows Firestore exists.

Everything here reads and writes app/domain/ entities and hides the storage shape
behind that — collection names, document ids, atomic array appends and the
serialisation round-trip are implementation details of this package, not facts the
agent or the API layer should have to know.
"""
