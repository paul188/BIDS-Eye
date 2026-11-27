"""What stands between a request and a route.

- middleware.py: the middleware stack main.py installs
  (CORS -> CSRF -> Session -> RequireLogin), deny-by-default login.
- sessions.py: the signed session cookie and the user read from it.
- rate_limits.py: one request budget per kind of route, as a dependency.
"""
