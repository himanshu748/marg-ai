"""Unsafe deployment configurations must fail before contacting AWS."""

import os
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        ({"MARG_DEPLOY_READ_ONLY": "false"}, "Writable deployment requires"),
        ({"MARG_API_TOKEN_SECRET_ARN": "a-secret-arn"}, "API token authentication requires"),
        ({"MARG_CERTIFICATE_ARN": "a-certificate-arn"}, "Set MARG_DASHBOARD_DOMAIN"),
        ({"MARG_AGENT_LLM": "invalid-provider"}, "MARG_AGENT_LLM must be"),
        ({"MARG_CPU_ARCHITECTURE": "invalid-architecture"}, "MARG_CPU_ARCHITECTURE must be"),
    ],
)
def test_deploy_rejects_unsafe_configuration_before_aws(tmp_path: Path, settings, message: str) -> None:
    marker = tmp_path / "aws-called"
    aws = tmp_path / "aws"
    aws.write_text('#!/bin/sh\ntouch "$AWS_TEST_MARKER"\nexit 1\n')
    aws.chmod(0o700)
    env = {name: value for name, value in os.environ.items() if not name.startswith("MARG_")}
    env.update(settings)
    env.update(PATH=f"{tmp_path}:{env.get('PATH', '')}", AWS_TEST_MARKER=str(marker))
    script = Path(__file__).parents[1] / "infra" / "deploy.sh"
    result = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True, timeout=5, check=False)
    assert result.returncode == 1
    assert message in result.stderr
    assert not marker.exists(), "Invalid configuration reached AWS before validation"


@pytest.mark.parametrize(("mode", "expected_gate"), [("mock", "0"), ("bedrock", "1")])
def test_template_provider_selection_sets_bedrock_gate(mode: str, expected_gate: str) -> None:
    # PyYAML ships with the cfn-lint tool required by deploy.sh.
    yaml = pytest.importorskip("yaml")

    class TemplateLoader(yaml.SafeLoader):
        pass

    def construct_intrinsic(loader, tag, node):
        value = loader.construct_sequence(node) if isinstance(node, yaml.SequenceNode) else loader.construct_scalar(node)
        return {tag: value}

    TemplateLoader.add_multi_constructor("!", construct_intrinsic)
    path = Path(__file__).parents[1] / "infra" / "cloudformation.yaml"
    template = yaml.load(path.read_text(), Loader=TemplateLoader)

    def resolve(expression):
        if not isinstance(expression, dict):
            return expression
        if "Ref" in expression:
            assert expression["Ref"] == "AgentLlm"
            return mode
        if "Equals" in expression:
            left, right = expression["Equals"]
            return resolve(left) == resolve(right)
        if "If" in expression:
            condition, when_true, when_false = expression["If"]
            return resolve(when_true if resolve(template["Conditions"][condition]) else when_false)
        raise AssertionError(f"Unexpected provider configuration: {expression}")

    container = template["Resources"]["TaskDefinition"]["Properties"]["ContainerDefinitions"][0]
    environment = {entry["Name"]: entry["Value"] for entry in container["Environment"]}
    assert resolve(environment["MARG_AGENT_LLM"]) == mode
    assert resolve(environment["MARG_BEDROCK_ENABLED"]) == expected_gate
