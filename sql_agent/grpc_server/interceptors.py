"""gRPC server interceptors (phase 8.5).

Currently only one interceptor — a simple bearer-token auth check that
gates every RPC on an ``authorization: Bearer <token>`` metadata header.
Disabled by default; enabled when ``GRPC_AUTH_TOKEN`` is set.

Intentionally NOT included here:
    * mTLS — let the service mesh / ingress handle it.
    * OAuth2 / JWT verification — application code should delegate to a
      dedicated identity service (Auth0, Cognito, Keycloak) rather than
      re-implementing JWT validation inline.
    * Rate limiting — belongs to the ingress / envoy layer.
"""

from __future__ import annotations

from typing import Any, Callable

import grpc

from sql_agent.config import get_logger


_log = get_logger("grpc_server.interceptors")


# Methods that are always allowed without a token. The gRPC health service
# is probe-level and must remain reachable for kube-probes regardless of
# auth config.
_UNAUTHENTICATED_ALLOWLIST = frozenset(
    {
        "/grpc.health.v1.Health/Check",
        "/grpc.health.v1.Health/Watch",
        "/sql_agent.v1.SqlAgent/HealthCheck",
    }
)


class BearerAuthInterceptor(grpc.ServerInterceptor):
    """Reject RPCs whose ``authorization`` metadata header does not match
    the configured bearer token. Health-check methods are exempt.
    """

    def __init__(self, expected_token: str) -> None:
        self._expected = f"Bearer {expected_token}".strip()
        _log.info("BearerAuthInterceptor installed (token length=%d)", len(expected_token))

    def intercept_service(
        self,
        continuation: Callable[[grpc.HandlerCallDetails], grpc.RpcMethodHandler],
        handler_call_details: grpc.HandlerCallDetails,
    ) -> grpc.RpcMethodHandler:
        method = handler_call_details.method or ""
        if method in _UNAUTHENTICATED_ALLOWLIST:
            return continuation(handler_call_details)

        metadata = dict(handler_call_details.invocation_metadata or ())
        provided = metadata.get("authorization", "")
        if provided != self._expected:
            # Return a unary handler that immediately aborts with
            # UNAUTHENTICATED so clients get a clean error. This works for
            # both unary and stream methods (the client just sees the
            # rejection on first send).
            def _deny(_request: Any, context: grpc.ServicerContext) -> Any:  # noqa: ARG001
                context.abort(
                    grpc.StatusCode.UNAUTHENTICATED,
                    "invalid or missing bearer token",
                )

            return grpc.unary_unary_rpc_method_handler(_deny)

        return continuation(handler_call_details)
