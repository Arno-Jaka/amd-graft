"""Agent layer tests.

The agent is the one component that talks to a non-deterministic
external service, so most of these tests are about the absence path:
no endpoint, no network, no model. The layering claim in agent.py's
docstring says the product works without this directory, and that is
asserted for real at the end of this file by deleting it.

No test here contacts a model. Tests that need model behaviour inject a
fake agent object, so the suite stays deterministic and offline.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from graft import config
from graft.agent import agent, prompts, tools
from graft.core import converter
from graft.verify import cases

EXAMPLE = config.EXAMPLES_DIR / "01_torch_cuda_softmax.py"
CUDA_EXAMPLE = config.EXAMPLES_DIR / "02_cuda_add_kernel.cu"


class FakeResponse:
    def __init__(self, content: str):
        self.content = content
        self.text = content


class FakeAgent:
    """Stands in for an Agno agent. Records prompts instead of sending them."""

    def __init__(self, reply: str = "fake reply", raises: Exception | None = None):
        self.reply = reply
        self.raises = raises
        self.prompts: list[str] = []

    def run(self, prompt: str) -> FakeResponse:
        self.prompts.append(prompt)
        if self.raises:
            raise self.raises
        return FakeResponse(self.reply)


@pytest.fixture
def no_endpoint(monkeypatch):
    monkeypatch.setattr(config, "LLM_MODEL", "")
    monkeypatch.setattr(config, "LLM_BASE_URL", "")
    return monkeypatch


# --- the layering claim ---------------------------------------------------


def test_agent_is_unavailable_without_an_endpoint(no_endpoint):
    assert agent.is_available() is False


def test_unavailable_reason_names_the_missing_setting(no_endpoint):
    assert "LLM_MODEL" in agent.unavailable_reason()


def test_missing_base_url_is_named_separately(monkeypatch):
    monkeypatch.setattr(config, "LLM_MODEL", "qwen")
    monkeypatch.setattr(config, "LLM_BASE_URL", "")
    assert "LLM_BASE_URL" in agent.unavailable_reason()


def test_narrate_returns_text_not_an_exception(no_endpoint):
    """Absence must render as a sentence, never raise into the report."""
    result = agent.narrate({"backend": "CPU"})
    assert result.startswith(agent.NOT_AVAILABLE)
    assert isinstance(result, str)


def test_no_model_is_constructed_when_unconfigured(no_endpoint):
    assert agent.build_model() is None
    assert agent.build_agent() is None


def test_agent_can_be_disabled_entirely(monkeypatch):
    monkeypatch.setattr(config, "LLM_MODEL", "qwen")
    monkeypatch.setattr(config, "LLM_BASE_URL", "http://localhost:8000/v1")
    monkeypatch.setattr(config, "AGENT_ENABLED", False)
    assert agent.build_agent() is None


def test_deterministic_layers_do_not_import_the_agent():
    """core/, verify/, bench/ must not depend on this directory.

    Checked by reading the source rather than by importing, since an import
    inside a function body would not show up in sys.modules.
    """
    offenders = []
    for path in Path(config.PACKAGE_ROOT).rglob("*.py"):
        if "agent" in path.parts:
            continue
        text = path.read_text()
        if "from graft.agent" in text or "import agent" in text:
            offenders.append(str(path.relative_to(config.PACKAGE_ROOT)))
    assert offenders == [], f"deterministic layers import the agent: {offenders}"


def test_product_works_with_agent_directory_deleted():
    """The layering test the docstring promises, executed for real.

    Copies the package, removes agent/, and runs the full suite plus an
    end-to-end convert-verify-report in the copy. If this fails, the claim
    that the agent is removable is false.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        clone = Path(tmp) / "graft"
        shutil.copytree(
            config.PROJECT_ROOT,
            clone,
            ignore=shutil.ignore_patterns(
                ".git", "__pycache__", ".venv", "*.pyc", ".pytest_cache"
            ),
        )
        shutil.rmtree(clone / "src" / "graft" / "agent")
        assert not (clone / "src" / "graft" / "agent").exists()

        env = {
            "PATH": "/usr/bin:/bin",
            "HOME": str(Path.home()),
            "VIRTUAL_ENV": str(config.PROJECT_ROOT / ".venv"),
        }
        # Ignore this file: it copies the tree and runs pytest inside the
        # copy, which without --ignore would recurse without end.
        suite = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "tests",
                "-q",
                "--ignore=tests/test_agent.py",
            ],
            cwd=clone,
            capture_output=True,
            text=True,
            env=env,
            timeout=600,
        )
        assert suite.returncode == 0, suite.stdout[-2000:]

        script = (
            "from graft.core import converter, runner\n"
            "from graft.report import report\n"
            "from graft.verify import cases, parity\n"
            "from graft import config\n"
            "p = config.EXAMPLES_DIR / '01_torch_cuda_softmax.py'\n"
            "mig = converter.convert(p.read_text(), p)\n"
            "res = parity.summarize(parity_results)\n"
            "print('OK', res['passed'], res['total'])\n"
        )
        # Replace the placeholder line with a real invocation.
        script = script.replace(
            "res = parity.summarize(parity_results)",
            "import importlib.util\n"
            "s = importlib.util.spec_from_file_location('m', p)\n"
            "m = importlib.util.module_from_spec(s)\n"
            "s.loader.exec_module(m)\n"
            "out = parity.check(m, cases.generate())\n"
            "res = parity.summarize(out)\n"
            "txt = report.render(report.ReportInputs(migration=mig, parity=res))\n"
            "assert '## Readiness' in txt",
        )
        run = subprocess.run(
            [sys.executable, "-c", script],
            cwd=clone,
            capture_output=True,
            text=True,
            env=env,
            timeout=300,
        )
        assert run.returncode == 0, run.stderr[-2000:]
        assert "OK 40 40" in run.stdout


# --- failure containment --------------------------------------------------


def test_model_error_becomes_text_not_an_exception():
    boom = RuntimeError("connection refused")
    result = agent.narrate({"backend": "CPU"}, agent=FakeAgent(raises=boom))
    assert result.startswith(agent.NOT_AVAILABLE)
    assert "connection refused" in result


def test_timeout_becomes_text():
    result = agent.explain_finding(
        {"symbol": "x", "severity": "blocker"}, agent=FakeAgent(raises=TimeoutError())
    )
    assert result.startswith(agent.NOT_AVAILABLE)
    assert "TimeoutError" in result


def test_empty_model_response_is_reported():
    result = agent.narrate({}, agent=FakeAgent(reply="   "))
    assert result.startswith(agent.NOT_AVAILABLE)


def test_response_without_content_attribute_is_handled():
    class Odd:
        text = "fallback text"

    class Holder:
        def run(self, prompt):
            return Odd()

    assert agent.narrate({}, agent=Holder()) == "fallback text"


def test_null_agent_uses_the_configured_one(no_endpoint):
    """Passing agent=None must not be confused with a broken agent."""
    assert agent.narrate({}, agent=None).startswith(agent.NOT_AVAILABLE)


def test_one_failing_section_does_not_stop_the_others(monkeypatch):
    monkeypatch.setattr(config, "LLM_MODEL", "qwen")
    monkeypatch.setattr(config, "LLM_BASE_URL", "http://localhost:8000/v1")
    monkeypatch.setattr(agent, "is_available", lambda: True)

    def explode(*args, **kwargs):
        raise RuntimeError("model down")

    monkeypatch.setattr(agent, "summarize_parity", explode)
    monkeypatch.setattr(agent, "summarize_benchmark", lambda *a, **k: "bench ok")

    from graft.report.report import ReportInputs

    sections = agent.narrative_sections(ReportInputs())
    assert sections["parity_explainer"].startswith(agent.NOT_AVAILABLE)
    assert sections["benchmark_explainer"] == "bench ok"


def test_narrative_sections_all_populated_when_unavailable(no_endpoint):
    from graft.report.report import ReportInputs

    sections = agent.narrative_sections(ReportInputs())
    assert set(sections) == {
        "narration",
        "parity_explainer",
        "benchmark_explainer",
        "remediation",
    }
    assert all(v.startswith(agent.NOT_AVAILABLE) for v in sections.values())


# --- prompt content -------------------------------------------------------


def test_system_prompt_forbids_correctness_claims():
    text = prompts.SYSTEM.lower()
    assert "never decide whether a port is correct" in text
    assert "never invent a cuda-to-hip mapping" in text
    assert "never claim a performance result" in text


def test_every_prompt_template_has_placeholders_resolved():
    """An unfilled placeholder would ship a literal {braces} to a model."""
    samples = {
        "EXPLAIN_FINDING": {
            "symbol": "s",
            "line": 1,
            "severity": "review",
            "severity_meaning": "m",
            "reason": "r",
            "remediation": "x",
            "gpu_name": "MI300X",
        },
        "EXPLAIN_CHANGE": {
            "index": 1,
            "total": 2,
            "symbol": "s",
            "line": 1,
            "reason": "r",
            "before": "a",
            "after": "b",
        },
        "SUMMARIZE_PARITY": {
            "device": "cpu",
            "passed": 1,
            "total": 1,
            "all_passed": True,
            "worst_case": "w",
            "max_abs_error": 0.0,
            "tolerance_note": "t",
            "failures": "none",
        },
        "SUMMARIZE_BENCHMARK": {"rows": "- none", "noise_pct": 10},
        "PROPOSE_REMEDIATION": {"unmapped": "- none"},
        "PROPOSE_TEST_CASE": {
            "case_count": 40,
            "operation": "softmax",
            "distributions": "normal",
            "shapes": "(4, 8)",
        },
        "NARRATE": {
            "backend": "CPU",
            "gpu_present": False,
            "conversion_verdict": "production_ready",
            "parity_summary": "40/40",
            "execution_summary": "ok",
            "benchmark_summary": "n/a",
        },
    }
    for name, values in samples.items():
        rendered = getattr(prompts, name).format(**values)
        assert "{" not in rendered, f"{name} left a placeholder: {rendered[:120]}"


def test_parity_prompt_carries_the_real_numbers():
    fake = FakeAgent()
    agent.summarize_parity(
        {
            "passed": 39,
            "total": 40,
            "all_passed": False,
            "worst_case": "nan/normal",
            "max_abs_error": 0.01,
        },
        agent=fake,
    )
    prompt = fake.prompts[0]
    assert "39" in prompt and "40" in prompt
    assert "nan/normal" in prompt
    assert "0.01" in prompt


def test_parity_failures_are_listed_in_the_prompt():
    fake = FakeAgent()
    agent.summarize_parity(
        {
            "passed": 1,
            "total": 2,
            "all_passed": False,
            "failures": [("case_a", "tolerance exceeded")],
        },
        agent=fake,
    )
    assert "case_a" in fake.prompts[0]
    assert "tolerance exceeded" in fake.prompts[0]


def test_benchmark_prompt_states_the_noise_band():
    fake = FakeAgent()
    agent.summarize_benchmark(
        [{"label": "port vs ref", "ratio": 4.5, "verdict": "significant regression"}],
        noise_band=1.10,
        agent=fake,
    )
    assert "10" in fake.prompts[0]
    assert "ratio 4.5" in fake.prompts[0]
    assert "implementation" in fake.prompts[0]
    assert "quality" in fake.prompts[0]


def test_finding_prompt_includes_severity_meaning():
    fake = FakeAgent()
    agent.explain_finding(
        {
            "symbol": "__shfl",
            "severity": "review",
            "line": 4,
            "reason": "widths differ",
            "remediation": "verify lanes",
        },
        agent=fake,
    )
    assert "verify the semantics" in fake.prompts[0]


# --- tools ----------------------------------------------------------------


def test_tools_register_with_required_arguments():
    registered = {tool.name: tool for tool in tools.build_tools()}
    assert len(registered) == 6
    for name in ("inspect_changes", "describe_coverage", "python_entry_points"):
        assert registered[name].parameters["required"] == ["source"], name


def test_tools_take_json_scalars_only():
    """A model can only pass JSON, so no tool may take a domain object."""
    registered = {tool.name: tool for tool in tools.build_tools()}
    assert registered["existing_case_coverage"].parameters.get("required") == []
    for name, tool in registered.items():
        for prop in tool.parameters.get("properties", {}).values():
            assert "object" not in str(prop.get("type", "")), f"{name} takes an object"


def test_inspect_changes_runs_the_real_converter():
    result = tools.inspect_changes(EXAMPLE.read_text())
    assert result["total"] == len(converter.convert(EXAMPLE.read_text()).changes)
    assert result["verdict"]


def test_describe_coverage_returns_scanner_severities():
    result = tools.describe_coverage("__global__ void k(){__shfl_xor_sync(1,2,0);}")
    assert result["verdict"] == "needs_review"
    assert result["reviews"]
    assert result["blockers"] == []


def test_known_equivalents_come_from_the_conversion_table():
    """Must not drift from what conversion actually renames."""
    result = tools.known_hip_equivalents(CUDA_EXAMPLE.read_text())
    mapped = {entry["cuda"]: entry["hip"] for entry in result if entry["hip"]}
    assert mapped["cudaMalloc"] == "hipMalloc"
    assert mapped["cudaFree"] == "hipFree"


def test_unmapped_construct_reports_no_hip_equivalent():
    """Honest None, so the model cannot invent an equivalent.

    __shfl_xor_sync is a real coverage finding and is deliberately absent
    from CUDA_TO_HIP, because renaming it is not sufficient: the lane
    widths differ. An invented __shfl mapping would be worse than none.
    """
    source = "void k(){ float v = __shfl_xor_sync(0xffffffff, 1.0f, 0); }"
    result = tools.known_hip_equivalents(source)
    unmapped = [entry for entry in result if entry["hip"] is None]
    assert unmapped
    assert "shfl" in unmapped[0]["cuda"]
    assert unmapped[0]["note"]


def test_entry_points_are_read_from_the_ast():
    result = tools.python_entry_points(EXAMPLE.read_text())
    names = {entry["name"] for entry in result["functions"]}
    assert {"solve", "kernel", "softmax"} <= names
    assert result["operation"] == "softmax"


def test_entry_points_report_syntax_errors_instead_of_raising():
    result = tools.python_entry_points("def solve(x)\n  pass\n")
    assert "SyntaxError" in result["error"]
    assert result["functions"] == []


def test_case_coverage_reports_the_real_generated_set():
    result = tools.existing_case_coverage()
    assert result["case_count"] == len(cases.generate())
    assert result["dtype"] == "torch.float32"


def test_suggested_shapes_avoid_duplicating_existing_cases():
    suggested = {
        tuple(entry["shape"]) for entry in tools.suggest_test_shapes("softmax")
    }
    covered = {tuple(shape) for shape in cases.SHAPES}
    assert suggested and not (suggested & covered)


def test_every_suggested_shape_carries_a_reason():
    for entry in tools.suggest_test_shapes("softmax"):
        assert entry["why"]
        assert entry["operation"] == "softmax"


def test_test_case_proposal_works_without_an_endpoint(no_endpoint):
    """Deterministic suggestions must exist even with no LLM."""
    result = agent.propose_test_cases("softmax")
    assert result["deterministic_suggestions"]
    assert "deterministic" in result["note"]


# --- report integration ---------------------------------------------------


def test_report_renders_narrative_after_the_numbers():
    from graft.report.report import ReportInputs

    inputs = ReportInputs(narratives={"narration": "model prose here"})
    text = (
        Path(__file__).parents[1] / "src" / "graft" / "report" / "report.py"
    ).read_text()
    order = text.index("_narrative_section(inputs)")
    readiness = text.index("_readiness(inputs)")
    assert order < readiness, "narrative must render before readiness"

    rendered = __import__("graft.report.report", fromlist=["render"]).render(inputs)
    assert "## Narrative (LLM-generated)" in rendered
    assert "model prose here" in rendered
    assert "does not decide correctness" in rendered


def test_report_marks_narrative_as_model_generated():
    from graft.report.report import ReportInputs, render

    text = render(ReportInputs(narratives={"narration": "prose"}))
    assert "LLM-generated" in text
    assert "a language model" in text


def test_absent_narrative_says_so_and_is_not_a_blocker():
    from graft.report.report import ReportInputs, render

    text = render(ReportInputs())
    assert "No narrative generated" in text
    assert "does not affect the verdict" in text
