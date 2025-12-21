"""
Agent OS Response Aggregator

Aggregates responses from multiple agents into a unified response.
Handles merging, conflict resolution, and response formatting.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any, Union
from enum import Enum, auto
import logging

from src.messaging.models import (
    FlowRequest,
    FlowResponse,
    ResponseContent,
    ResponseMetadata,
    MessageStatus,
)
from .flow import FlowResult, AgentResult
from .router import RoutingStrategy


logger = logging.getLogger(__name__)


class AggregationStrategy(Enum):
    """Strategies for aggregating multiple responses."""
    FIRST_SUCCESS = auto()   # Use first successful response
    LAST_SUCCESS = auto()    # Use last successful response (for sequential)
    MERGE = auto()           # Merge all responses
    BEST_CONFIDENCE = auto() # Use highest confidence response
    CONSENSUS = auto()       # Require agreement between agents


@dataclass
class AggregatedResponse:
    """Result of response aggregation."""
    primary_output: Any
    all_outputs: List[Any]
    strategy_used: AggregationStrategy
    sources: List[str]           # Agent names that contributed
    total_tokens: int = 0
    total_inference_time_ms: int = 0
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class ResponseAggregator:
    """
    Aggregates responses from multiple agent invocations.

    Supports:
    - First/last success selection
    - Response merging
    - Confidence-based selection
    - Consensus checking
    """

    def __init__(
        self,
        default_strategy: AggregationStrategy = AggregationStrategy.FIRST_SUCCESS,
        merge_separator: str = "\n\n---\n\n",
    ):
        """
        Initialize aggregator.

        Args:
            default_strategy: Default aggregation strategy
            merge_separator: Separator for merged responses
        """
        self.default_strategy = default_strategy
        self.merge_separator = merge_separator

    def aggregate(
        self,
        flow_result: FlowResult,
        request: FlowRequest,
        strategy: Optional[AggregationStrategy] = None,
    ) -> FlowResponse:
        """
        Aggregate flow results into a single response.

        Args:
            flow_result: Results from flow execution
            request: Original request
            strategy: Aggregation strategy (uses default if not specified)

        Returns:
            Unified FlowResponse
        """
        strategy = strategy or self._determine_strategy(flow_result)

        # Get successful results
        successful = flow_result.successful_results

        if not successful:
            # No successful results
            return self._create_error_response(request, flow_result)

        # Aggregate based on strategy
        if strategy == AggregationStrategy.FIRST_SUCCESS:
            aggregated = self._first_success(successful)
        elif strategy == AggregationStrategy.LAST_SUCCESS:
            aggregated = self._last_success(successful)
        elif strategy == AggregationStrategy.MERGE:
            aggregated = self._merge_responses(successful)
        elif strategy == AggregationStrategy.BEST_CONFIDENCE:
            aggregated = self._best_confidence(successful)
        elif strategy == AggregationStrategy.CONSENSUS:
            aggregated = self._consensus(successful)
        else:
            aggregated = self._first_success(successful)

        # Build response
        return self._build_response(request, aggregated, flow_result)

    def _determine_strategy(self, flow_result: FlowResult) -> AggregationStrategy:
        """Determine best strategy based on flow."""
        if flow_result.strategy_used == RoutingStrategy.SEQUENTIAL:
            return AggregationStrategy.LAST_SUCCESS
        elif flow_result.strategy_used == RoutingStrategy.PARALLEL:
            return AggregationStrategy.MERGE
        elif flow_result.strategy_used == RoutingStrategy.FALLBACK:
            return AggregationStrategy.FIRST_SUCCESS
        return self.default_strategy

    def _first_success(self, results: List[AgentResult]) -> AggregatedResponse:
        """Use first successful response."""
        first = results[0]
        return AggregatedResponse(
            primary_output=first.output,
            all_outputs=[r.output for r in results],
            strategy_used=AggregationStrategy.FIRST_SUCCESS,
            sources=[first.agent_name],
            total_tokens=first.tokens_consumed,
            total_inference_time_ms=first.duration_ms,
        )

    def _last_success(self, results: List[AgentResult]) -> AggregatedResponse:
        """Use last successful response (for sequential flows)."""
        last = results[-1]
        total_tokens = sum(r.tokens_consumed for r in results)
        total_time = sum(r.duration_ms for r in results)

        return AggregatedResponse(
            primary_output=last.output,
            all_outputs=[r.output for r in results],
            strategy_used=AggregationStrategy.LAST_SUCCESS,
            sources=[r.agent_name for r in results],
            total_tokens=total_tokens,
            total_inference_time_ms=total_time,
        )

    def _merge_responses(self, results: List[AgentResult]) -> AggregatedResponse:
        """Merge all responses into one."""
        outputs = []
        for result in results:
            output = result.output
            if isinstance(output, str):
                outputs.append(f"**{result.agent_name}:**\n{output}")
            elif isinstance(output, dict):
                outputs.append(output)
            else:
                outputs.append(str(output))

        # Merge strings with separator
        if all(isinstance(o, str) for o in outputs):
            merged = self.merge_separator.join(outputs)
        else:
            merged = outputs

        total_tokens = sum(r.tokens_consumed for r in results)
        total_time = sum(r.duration_ms for r in results)

        return AggregatedResponse(
            primary_output=merged,
            all_outputs=[r.output for r in results],
            strategy_used=AggregationStrategy.MERGE,
            sources=[r.agent_name for r in results],
            total_tokens=total_tokens,
            total_inference_time_ms=total_time,
        )

    def _best_confidence(self, results: List[AgentResult]) -> AggregatedResponse:
        """Select response with highest confidence."""
        # For now, use first result (confidence not in AgentResult)
        # In a full implementation, responses would include confidence
        return self._first_success(results)

    def _consensus(self, results: List[AgentResult]) -> AggregatedResponse:
        """Check for consensus between agents."""
        # Simple consensus: all outputs are similar
        # In practice, this would use semantic similarity

        outputs = [r.output for r in results]

        # For string outputs, check if they're similar
        if all(isinstance(o, str) for o in outputs):
            # Very simple similarity check
            first_words = set(outputs[0].lower().split()[:10])
            consensus = True

            for output in outputs[1:]:
                other_words = set(output.lower().split()[:10])
                overlap = len(first_words & other_words) / max(len(first_words), 1)
                if overlap < 0.3:
                    consensus = False
                    break

            if consensus:
                # Use first output
                return self._first_success(results)
            else:
                # Merge with note about differing opinions
                merged = self._merge_responses(results)
                merged.metadata["consensus"] = False
                return merged

        return self._first_success(results)

    def _build_response(
        self,
        request: FlowRequest,
        aggregated: AggregatedResponse,
        flow_result: FlowResult,
    ) -> FlowResponse:
        """Build final FlowResponse from aggregated results."""
        # Determine status
        if flow_result.status.name == "COMPLETED":
            status = MessageStatus.SUCCESS
        elif flow_result.status.name == "PARTIAL":
            status = MessageStatus.PARTIAL
        else:
            status = MessageStatus.ERROR

        # Build metadata
        metadata = ResponseMetadata(
            tokens_consumed=aggregated.total_tokens,
            inference_time_ms=aggregated.total_inference_time_ms,
        )

        # Build content
        content = ResponseContent(
            output=aggregated.primary_output,
            confidence=aggregated.confidence,
            metadata=metadata,
        )

        # Build response
        response = FlowResponse(
            request_id=request.request_id,
            source="whisper",
            destination=request.source,
            status=status,
            content=content,
        )

        # Add routing metadata
        response.next_actions.append({
            "action": "routing_complete",
            "agents_used": aggregated.sources,
            "strategy": aggregated.strategy_used.name,
            "flow_duration_ms": flow_result.total_duration_ms,
        })

        return response

    def _create_error_response(
        self,
        request: FlowRequest,
        flow_result: FlowResult,
    ) -> FlowResponse:
        """Create error response when all agents failed."""
        errors = [
            f"{r.agent_name}: {r.error}"
            for r in flow_result.failed_results
            if r.error
        ]

        return FlowResponse(
            request_id=request.request_id,
            source="whisper",
            destination=request.source,
            status=MessageStatus.ERROR,
            content=ResponseContent(
                output="Unable to process request. All agents failed.",
                errors=errors,
            ),
        )


class ResponseFormatter:
    """
    Formats aggregated responses for different output formats.
    """

    @staticmethod
    def to_text(response: FlowResponse) -> str:
        """Format response as plain text."""
        output = response.content.output

        if isinstance(output, str):
            return output
        elif isinstance(output, dict):
            import json
            return json.dumps(output, indent=2)
        else:
            return str(output)

    @staticmethod
    def to_markdown(response: FlowResponse) -> str:
        """Format response as markdown."""
        output = response.content.output

        if isinstance(output, str):
            return output

        # Format dict as markdown
        if isinstance(output, dict):
            lines = []
            for key, value in output.items():
                lines.append(f"**{key}:** {value}")
            return "\n".join(lines)

        return str(output)

    @staticmethod
    def to_json(response: FlowResponse) -> Dict[str, Any]:
        """Format response as JSON-serializable dict."""
        return {
            "request_id": str(response.request_id),
            "status": response.status.value,
            "output": response.content.output,
            "confidence": response.content.confidence,
            "errors": response.content.errors,
            "metadata": {
                "tokens": response.content.metadata.tokens_consumed,
                "time_ms": response.content.metadata.inference_time_ms,
            },
        }
