"""
app/integrations/
──────────────────
Outbound HTTP to systems this service does not own.

  himcab/    OUR Spring backend — fares, rental rate card, coupons. Authenticated
             with the rider's own JWT, forwarded from the request and never logged.
  external/  Third parties — geocoding, routing, weather.

The split matters because the failure policies differ. A Him Cab call failing means
we cannot quote a real price and must say so; a third-party call failing means we lose
detail in the reply. Neither may fail the turn, so every client here degrades to None
rather than raising — see the per-tool timeout in app/agent/nodes/tool_executor.py.
"""
