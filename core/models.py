from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List, Any
import time

@dataclass
class SubTaskSpec:
    role: str
    filename: str
    instructions: str

@dataclass
class HelmPlan:
    plan_summary: str
    subtasks: Dict[str, SubTaskSpec]
    project: Dict[str, Any] = field(default_factory=dict)
    usage: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AgentResult:
    agent_id: str
    model: str
    role: str
    filename: str
    instructions: str
    output: str
    started_at: float
    finished_at: float
    duration_s: float = field(init=False)
    tokens_in: Optional[int] = 0
    tokens_out: Optional[int] = 0
    cost_est: Optional[float] = 0.0
    error: Optional[str] = None

    def __post_init__(self):
        self.duration_s = round(max(self.finished_at - self.started_at, 0.001), 3)

@dataclass
class OrchestrationReport:
    run_id: str
    task: str
    helm_model: str
    num_agents: int
    plan_summary: str
    wall_clock_s: float
    sum_of_agent_times_s: float
    concurrency_speedup: float
    total_tokens_in: int
    total_tokens_out: int
    total_cost_est: float
    output_dir: str
    files_written: List[str]
    agents: List[Dict[str, Any]]
    created_at: float = field(default_factory=time.time)
    project: Dict[str, Any] = field(default_factory=dict)
    packaging: Dict[str, Any] = field(default_factory=dict)
    phase_usage: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
