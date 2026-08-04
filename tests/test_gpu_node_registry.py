"""Worker ids must include the port, and hardware must come from the box."""

from __future__ import annotations

import json
import unittest
from unittest import mock

from orchestrator.gpu_node_registry import (
    SSHNodeSpec,
    canary_node,
    configured_nodes,
    node_spec_from_mapping,
    parse_gpu_query,
)


def _spec(**overrides) -> SSHNodeSpec:
    values = {
        "host": "111.172.214.101",
        "port": 32035,
        "user": "root",
        "credential_ref": "env:DEEPGRAPH_SSH_CREDENTIAL",
    }
    values.update(overrides)
    return SSHNodeSpec(**values)


class WorkerIdTests(unittest.TestCase):
    def test_the_id_includes_the_port_so_same_ip_nodes_coexist(self):
        from orchestrator.gpu_scheduler import ssh_worker_id

        a = ssh_worker_id("111.172.214.101", 32035, "0")
        b = ssh_worker_id("111.172.214.101", 32036, "0")

        self.assertEqual(a, "ssh:111.172.214.101:32035:gpu0")
        self.assertNotEqual(a, b)


class SpecTests(unittest.TestCase):
    def test_a_valid_node_passes(self):
        _spec().validate()

    def test_a_literal_credential_is_refused(self):
        with self.assertRaisesRegex(ValueError, "never a literal"):
            _spec(credential_ref="Phil@130901").validate()

    def test_missing_fields_are_refused(self):
        with self.assertRaises(ValueError):
            _spec(host="").validate()
        with self.assertRaises(ValueError):
            _spec(user="").validate()
        with self.assertRaises(ValueError):
            _spec(port=0).validate()
        with self.assertRaises(ValueError):
            _spec(credential_ref="").validate()


class GpuQueryTests(unittest.TestCase):
    def test_parses_real_nvidia_smi_output(self):
        gpus = parse_gpu_query(
            "0, NVIDIA A100-PCIE-40GB, 40960 MiB\n"
            "1, NVIDIA A100-PCIE-40GB, 40960 MiB"
        )

        self.assertEqual(len(gpus), 2)
        self.assertEqual(gpus[0]["gpu_model"], "NVIDIA A100-PCIE-40GB")
        self.assertEqual(gpus[0]["total_mem_gb"], 40.0)
        self.assertEqual(gpus[0]["device"], "0")

    def test_ignores_noise_and_headers(self):
        self.assertEqual(parse_gpu_query(""), [])
        self.assertEqual(parse_gpu_query("NO_NVIDIA_SMI"), [])
        self.assertEqual(parse_gpu_query("index, name, memory.total"), [])


class CanaryTests(unittest.TestCase):
    def _row(self, spec):
        return {
            "id": f"ssh:{spec.host}:{spec.port}:gpu",
            "metadata": json.dumps(
                {"ssh_host": spec.host, "ssh_port": spec.port, "ssh_user": spec.user}
            ),
        }

    def test_a_reachable_node_reports_its_real_gpus(self):
        completed = mock.Mock(
            returncode=0,
            stdout="0, NVIDIA A100-PCIE-40GB, 40960 MiB\n1, NVIDIA A100-PCIE-40GB, 40960 MiB",
        )
        gpus = canary_node(
            _spec(),
            run_remote=lambda worker, script: completed,
            worker_row=self._row,
        )

        self.assertEqual([g["gpu_model"] for g in gpus], ["NVIDIA A100-PCIE-40GB"] * 2)

    def test_an_unreachable_node_is_not_registered(self):
        completed = mock.Mock(returncode=255, stderr="Host key verification failed.", stdout="")
        with self.assertRaisesRegex(RuntimeError, "canary failed"):
            canary_node(_spec(), run_remote=lambda w, s: completed, worker_row=self._row)

    def test_a_node_without_gpus_is_not_registered(self):
        completed = mock.Mock(returncode=0, stdout="NO_NVIDIA_SMI")
        with self.assertRaisesRegex(RuntimeError, "no devices"):
            canary_node(_spec(), run_remote=lambda w, s: completed, worker_row=self._row)


class ConfiguredNodesTests(unittest.TestCase):
    def test_parses_a_list_of_nodes(self):
        raw = json.dumps(
            [
                {"host": "111.172.214.101", "port": 32035, "user": "root",
                 "credential_ref": "env:DEEPGRAPH_SSH_CREDENTIAL"},
                {"host": "111.172.214.101", "port": 32036, "user": "root",
                 "credential_ref": "env:DEEPGRAPH_SSH_CREDENTIAL"},
            ]
        )
        nodes = configured_nodes(raw)

        self.assertEqual([n.port for n in nodes], [32035, 32036])

    def test_bad_or_empty_config_yields_no_nodes(self):
        self.assertEqual(configured_nodes(""), [])
        self.assertEqual(configured_nodes("{not json"), [])
        self.assertEqual(configured_nodes(json.dumps([{"host": ""}])), [])
        # a literal credential in config is dropped, not honoured
        self.assertEqual(
            configured_nodes(json.dumps([{"host": "h", "port": 22, "user": "u",
                                          "credential_ref": "literalpw"}])),
            [],
        )

    def test_node_spec_from_mapping_defaults(self):
        spec = node_spec_from_mapping(
            {"host": "h", "port": 22, "user": "u",
             "credential_ref": "env:DEEPGRAPH_SSH_CREDENTIAL"}
        )
        self.assertEqual(spec.remote_base_dir, "/root/deepgraph-remote-worker")
        self.assertEqual(spec.python_bin, "python")


if __name__ == "__main__":
    unittest.main()
