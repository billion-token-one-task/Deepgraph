"""Role-separated LLM routing with explicit fallback and accounting hooks."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Protocol

from contracts.meta_harness import ResourceGrant
from meta_harness.grants import ResourceRequest, authorize


VALID_ROLES = {"proposer", "evaluator", "reviewer"}


class LLMRouteError(RuntimeError):
    pass


class LLMRouteUnavailableError(LLMRouteError):
    pass


class LLMExecutionFailure(LLMRouteError):
    """Executor failure with any metered usage returned by the provider."""

    def __init__(
        self,
        message: str,
        *,
        category: str = "transient",
        usage: "RouteUsage | None" = None,
    ):
        super().__init__(message)
        self.failure_category = category
        self.usage = usage or RouteUsage(0, 0, None)


@dataclass(frozen=True)
class ProviderRoute:
    route_id: str
    provider: str
    model: str
    model_family: str
    prompt_version: str
    timeout_seconds: int
    transient_retries: int = 0
    auth_cooldown_seconds: int = 900
    transient_cooldown_seconds: int = 180

    def validate(self) -> None:
        if not all(
            (
                self.route_id,
                self.provider,
                self.model,
                self.model_family,
                self.prompt_version,
            )
        ):
            raise LLMRouteError("provider route metadata is incomplete")
        if self.timeout_seconds <= 0 or self.transient_retries < 0:
            raise LLMRouteError("provider route retry/timeout values are invalid")


@dataclass(frozen=True)
class RouteRequest:
    agenda_id: int
    idea_id: int
    role: str
    stage: str
    resource_grant_id: int
    token_cap: int
    operation: str
    idempotency_key: str
    proposer_route: ProviderRoute | None = None
    max_attempts: int | None = None

    def validate(self) -> None:
        if min(self.agenda_id, self.idea_id, self.resource_grant_id) <= 0:
            raise LLMRouteError("LLM route scope ids must be positive")
        if self.role not in VALID_ROLES:
            raise LLMRouteError("invalid LLM role")
        if not self.stage or not self.operation or not self.idempotency_key:
            raise LLMRouteError("LLM route metadata is incomplete")
        if self.token_cap <= 0:
            raise LLMRouteError("token_cap must be positive")
        if self.max_attempts is not None and self.max_attempts <= 0:
            raise LLMRouteError("max_attempts must be positive")


@dataclass(frozen=True)
class RouteUsage:
    input_tokens: int
    output_tokens: int
    cost_usd: float | None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def validate(self) -> None:
        if self.input_tokens < 0 or self.output_tokens < 0:
            raise LLMRouteError("provider token usage cannot be negative")
        if self.cost_usd is not None and self.cost_usd < 0:
            raise LLMRouteError("provider cost cannot be negative")


@dataclass(frozen=True)
class RouteResult:
    output: Any
    route: ProviderRoute
    usage: RouteUsage
    attempts: int


@dataclass(frozen=True)
class RouteObservation:
    agenda_id: int
    idea_id: int
    role: str
    provider: str
    model: str
    model_family: str
    prompt_version: str
    input_tokens: int
    output_tokens: int
    cost_usd: float | None
    status: str
    failure_reason: str | None
    reservation_id: int | None


class ReservationLedger(Protocol):
    def reserve(
        self,
        *,
        agenda_id: int,
        operation: str,
        idempotency_key: str,
        token_cap: int,
        gpu_hours_cap: float = 0.0,
    ) -> Any: ...

    def settle(
        self,
        reservation_id: int,
        *,
        tokens_used: int,
        gpu_hours_used: float = 0.0,
        cost_usd: float | None = None,
    ) -> None: ...

    def release(self, reservation_id: int, *, reason: str) -> None: ...


class CooldownStore(Protocol):
    def load_active_cooldowns(
        self,
        route_ids: list[str],
        *,
        now: datetime,
    ) -> dict[str, datetime]: ...

    def save_cooldown(
        self,
        route: ProviderRoute,
        *,
        until: datetime,
        failure_category: str,
    ) -> None: ...


class LLMRouter:
    def __init__(
        self,
        routes_by_role: dict[str, list[ProviderRoute]],
        *,
        ledger: ReservationLedger,
        observation_sink: Callable[[RouteObservation], None],
        cooldown_store: CooldownStore | None = None,
    ):
        if set(routes_by_role) - VALID_ROLES:
            raise LLMRouteError("unknown LLM role")
        for role in VALID_ROLES:
            if not routes_by_role.get(role):
                raise LLMRouteError(f"no routes configured for {role}")
            for route in routes_by_role[role]:
                route.validate()
        self._routes = {role: tuple(routes) for role, routes in routes_by_role.items()}
        self._ledger = ledger
        self._observation_sink = observation_sink
        self._cooldown_store = cooldown_store
        self._cooldown_until: dict[str, datetime] = {}

    @staticmethod
    def _is_independent(route: ProviderRoute, proposer: ProviderRoute | None) -> bool:
        if proposer is None:
            return True
        return (
            route.provider != proposer.provider
            or route.model_family != proposer.model_family
        )

    def eligible_routes(
        self, request: RouteRequest, *, now: datetime | None = None
    ) -> list[ProviderRoute]:
        if request.role not in VALID_ROLES:
            raise LLMRouteError("invalid LLM role")
        current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        role_routes = self._routes[request.role]
        if self._cooldown_store is not None:
            durable = self._cooldown_store.load_active_cooldowns(
                [route.route_id for route in role_routes],
                now=current,
            )
            self._cooldown_until.update(durable)
        routes = [
            route
            for route in role_routes
            if self._cooldown_until.get(route.route_id, current) <= current
        ]
        if request.role in {"evaluator", "reviewer"}:
            independent = [
                route
                for route in routes
                if self._is_independent(route, request.proposer_route)
            ]
            if not independent:
                raise LLMRouteUnavailableError(
                    "no_independent_evaluator_route_available;manual_review_required"
                )
            routes = independent
        return routes

    def invoke(
        self,
        request: RouteRequest,
        *,
        grant: ResourceGrant | None,
        executor: Callable[[ProviderRoute, RouteRequest], tuple[Any, RouteUsage]],
    ) -> RouteResult:
        request.validate()
        authorize(
            grant,
            ResourceRequest(
                agenda_id=request.agenda_id,
                idea_id=request.idea_id,
                stage=request.stage,
                backend="llm",
                resource_grant_id=request.resource_grant_id,
                token_cap=request.token_cap,
            ),
        )
        reservation = self._ledger.reserve(
            agenda_id=request.agenda_id,
            operation=request.operation,
            idempotency_key=request.idempotency_key,
            token_cap=request.token_cap,
        )
        attempts = 0
        failures: list[str] = []
        consumed_input = 0
        consumed_output = 0
        consumed_cost = 0.0
        has_cost = False
        settled = False
        try:
            for route in self.eligible_routes(request):
                for _attempt in range(route.transient_retries + 1):
                    if (
                        request.max_attempts is not None
                        and attempts >= request.max_attempts
                    ):
                        raise LLMRouteUnavailableError(
                            "route_attempt_cap_exhausted;manual_review_required"
                        )
                    if consumed_input + consumed_output >= request.token_cap:
                        raise LLMRouteError("reserved_token_cap_exhausted")
                    attempts += 1
                    try:
                        output, usage = executor(route, request)
                        usage.validate()
                        total_tokens = (
                            consumed_input + consumed_output + usage.total_tokens
                        )
                        if total_tokens > request.token_cap:
                            consumed_input += usage.input_tokens
                            consumed_output += usage.output_tokens
                            if usage.cost_usd is not None:
                                consumed_cost += usage.cost_usd
                                has_cost = True
                            self._ledger.settle(
                                reservation.reservation_id,
                                tokens_used=request.token_cap,
                                cost_usd=consumed_cost if has_cost else None,
                            )
                            settled = True
                            self._observation_sink(
                                RouteObservation(
                                    agenda_id=request.agenda_id,
                                    idea_id=request.idea_id,
                                    role=request.role,
                                    provider=route.provider,
                                    model=route.model,
                                    model_family=route.model_family,
                                    prompt_version=route.prompt_version,
                                    input_tokens=usage.input_tokens,
                                    output_tokens=usage.output_tokens,
                                    cost_usd=usage.cost_usd,
                                    status="failed",
                                    failure_reason=(
                                        "provider_usage_exceeded_reserved_cap"
                                    ),
                                    reservation_id=reservation.reservation_id,
                                )
                            )
                            raise LLMRouteError("provider_usage_exceeded_reserved_cap")
                        consumed_input += usage.input_tokens
                        consumed_output += usage.output_tokens
                        if usage.cost_usd is not None:
                            consumed_cost += usage.cost_usd
                            has_cost = True
                        self._ledger.settle(
                            reservation.reservation_id,
                            tokens_used=consumed_input + consumed_output,
                            cost_usd=consumed_cost if has_cost else None,
                        )
                        settled = True
                        self._observation_sink(
                            RouteObservation(
                                agenda_id=request.agenda_id,
                                idea_id=request.idea_id,
                                role=request.role,
                                provider=route.provider,
                                model=route.model,
                                model_family=route.model_family,
                                prompt_version=route.prompt_version,
                                input_tokens=usage.input_tokens,
                                output_tokens=usage.output_tokens,
                                cost_usd=usage.cost_usd,
                                status="succeeded",
                                failure_reason=None,
                                reservation_id=reservation.reservation_id,
                            )
                        )
                        return RouteResult(output, route, usage, attempts)
                    except Exception as exc:
                        if isinstance(exc, LLMRouteError) and not isinstance(
                            exc, LLMExecutionFailure
                        ):
                            raise
                        category = getattr(exc, "failure_category", "transient")
                        failed_usage = getattr(exc, "usage", RouteUsage(0, 0, None))
                        failed_usage.validate()
                        if (
                            consumed_input
                            + consumed_output
                            + failed_usage.total_tokens
                            > request.token_cap
                        ):
                            consumed_input += failed_usage.input_tokens
                            consumed_output += failed_usage.output_tokens
                            if failed_usage.cost_usd is not None:
                                consumed_cost += failed_usage.cost_usd
                                has_cost = True
                            self._ledger.settle(
                                reservation.reservation_id,
                                tokens_used=request.token_cap,
                                cost_usd=consumed_cost if has_cost else None,
                            )
                            settled = True
                            failures.append(f"{route.route_id}:hard_token_cap")
                            self._observation_sink(
                                RouteObservation(
                                    agenda_id=request.agenda_id,
                                    idea_id=request.idea_id,
                                    role=request.role,
                                    provider=route.provider,
                                    model=route.model,
                                    model_family=route.model_family,
                                    prompt_version=route.prompt_version,
                                    input_tokens=failed_usage.input_tokens,
                                    output_tokens=failed_usage.output_tokens,
                                    cost_usd=failed_usage.cost_usd,
                                    status="failed",
                                    failure_reason=(
                                        "failed_attempt_usage_exceeded_reserved_cap"
                                    ),
                                    reservation_id=reservation.reservation_id,
                                )
                            )
                            raise LLMRouteError(
                                "failed_attempt_usage_exceeded_reserved_cap"
                            ) from exc
                        consumed_input += failed_usage.input_tokens
                        consumed_output += failed_usage.output_tokens
                        if failed_usage.cost_usd is not None:
                            consumed_cost += failed_usage.cost_usd
                            has_cost = True
                        failures.append(f"{route.route_id}:{category}:{type(exc).__name__}")
                        seconds = (
                            route.auth_cooldown_seconds
                            if category == "auth"
                            else route.transient_cooldown_seconds
                        )
                        cooldown_until = datetime.now(timezone.utc) + timedelta(
                            seconds=seconds
                        )
                        self._cooldown_until[route.route_id] = cooldown_until
                        if self._cooldown_store is not None:
                            self._cooldown_store.save_cooldown(
                                route,
                                until=cooldown_until,
                                failure_category=category,
                            )
                        self._observation_sink(
                            RouteObservation(
                                agenda_id=request.agenda_id,
                                idea_id=request.idea_id,
                                role=request.role,
                                provider=route.provider,
                                model=route.model,
                                model_family=route.model_family,
                                prompt_version=route.prompt_version,
                                input_tokens=failed_usage.input_tokens,
                                output_tokens=failed_usage.output_tokens,
                                cost_usd=failed_usage.cost_usd,
                                status="failed",
                                failure_reason=f"{category}:{type(exc).__name__}",
                                reservation_id=reservation.reservation_id,
                            )
                        )
                        if category == "auth":
                            break
            raise LLMRouteUnavailableError(
                "all_explicit_routes_failed;manual_review_required:" + ",".join(failures)
            )
        except Exception:
            if not settled and consumed_input + consumed_output > 0:
                self._ledger.settle(
                    reservation.reservation_id,
                    tokens_used=consumed_input + consumed_output,
                    cost_usd=consumed_cost if has_cost else None,
                )
                settled = True
            if not settled:
                self._ledger.release(
                    reservation.reservation_id,
                    reason="llm_route_failed_before_settlement",
                )
            raise
