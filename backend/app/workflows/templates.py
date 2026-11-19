"""Loads the ComfyUI API-format workflow templates and fills them in for a job.

The templates (``workflows/comfyui/wan22_ti2v_5b_{t2v,i2v}.json``) are exported from the
official Comfy-Org workflow by ComfyUI's own frontend (see ``workflows/comfyui/tools``).
A mapping file names every node and input VIDSGEN changes. At load time, every mapped
node id is checked for existence and the expected ``class_type``, so a re-exported
template whose ids moved fails loudly at startup instead of silently producing a wrong
graph.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..jobs.models import GenerationMode


class WorkflowError(Exception):
    """The workflow templates or mapping are missing, inconsistent or invalid."""


@dataclass(frozen=True)
class RequiredModel:
    folder: str
    file: str
    node_id: str
    input: str


@dataclass(frozen=True)
class GraphValues:
    """Provider-neutral values that go into the graph."""

    prompt: str
    negative_prompt: str
    width: int
    height: int
    num_frames: int
    fps: int
    steps: int
    guidance_scale: float
    seed: int
    filename_prefix: str
    image: str | None = None   # ComfyUI input-folder reference, e.g. "vidsgen/job_x.png" (i2v only)


class WorkflowTemplates:
    def __init__(self, directory: Path, mapping_file: str = "wan22_ti2v_5b.mapping.json"):
        self.directory = Path(directory)
        mapping_path = self.directory / mapping_file
        try:
            self.mapping: dict[str, Any] = json.loads(mapping_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise WorkflowError(f"Workflow mapping not found: {mapping_path}") from None
        except json.JSONDecodeError as e:
            raise WorkflowError(f"Workflow mapping is not valid JSON: {mapping_path}: {e}") from None

        # Exact expected file sizes (bytes) from the model manifest, if present.
        self.expected_sizes: dict[tuple[str, str], int] = {}
        manifest = self.directory / "models.yaml"
        if manifest.is_file():
            import yaml

            for f in (yaml.safe_load(manifest.read_text(encoding="utf-8")) or {}).get("files", []):
                folder, name = Path(f["comfyui_dir"]).name, Path(f["path"]).name
                self.expected_sizes[(folder, name)] = int(f["size_bytes"])

        self.templates: dict[GenerationMode, dict[str, Any]] = {}
        for mode in GenerationMode:
            name = self.mapping.get("templates", {}).get(mode.value)
            if not name:
                raise WorkflowError(f"Mapping has no template for mode {mode.value!r}")
            path = self.directory / name
            try:
                self.templates[mode] = json.loads(path.read_text(encoding="utf-8"))
            except FileNotFoundError:
                raise WorkflowError(f"Workflow template not found: {path}") from None
            except json.JSONDecodeError as e:
                raise WorkflowError(f"Workflow template is not valid JSON: {path}: {e}") from None
        self.validate()

    # ------------------------------------------------------------------ lookups
    @property
    def name(self) -> str:
        return self.mapping.get("name", "workflow")

    def node_id(self, role: str) -> str:
        try:
            return str(self.mapping["nodes"][role]["id"])
        except KeyError:
            raise WorkflowError(f"Mapping has no node role {role!r}") from None

    @property
    def default_negative_prompt(self) -> str:
        return self.mapping.get("default_negative_prompt", "")

    @property
    def required_models(self) -> list[RequiredModel]:
        return [RequiredModel(m["folder"], m["file"], self.node_id(m["node"]), m["input"])
                for m in self.mapping.get("required_models", [])]

    @property
    def node_classes(self) -> list[str]:
        classes = {n["class_type"] for g in self.templates.values() for n in g.values()}
        return sorted(classes)

    @property
    def progress_config(self) -> dict[str, Any]:
        return self.mapping.get("progress", {})

    def _applies(self, entry: dict[str, Any], mode: GenerationMode) -> bool:
        return mode.value in entry.get("modes", [m.value for m in GenerationMode])

    # ------------------------------------------------------------------ validation
    def validate(self) -> None:
        """Structural checks that must hold for every template/mapping combination."""
        problems: list[str] = []
        nodes = self.mapping.get("nodes", {})
        for mode, graph in self.templates.items():
            if not isinstance(graph, dict) or not graph:
                problems.append(f"{mode.value}: template is empty or not an API-format graph")
                continue
            bad = [nid for nid, node in graph.items()
                   if not isinstance(node, dict) or "class_type" not in node or not isinstance(node.get("inputs"), dict)]
            if bad:
                problems.append(f"{mode.value}: not in API format (nodes {', '.join(bad[:5])} lack class_type/inputs); "
                                "export it with the tools in workflows/comfyui/tools")
                continue
            for role, spec in nodes.items():
                node = graph.get(str(spec["id"]))
                if not self._applies(spec, mode):
                    if node is not None:
                        problems.append(f"{mode.value}: node {role} ({spec['id']}) must not be present in this mode")
                    continue
                if node is None:
                    problems.append(f"{mode.value}: missing node {role} (id {spec['id']})")
                elif node.get("class_type") != spec["class_type"]:
                    problems.append(f"{mode.value}: node {spec['id']} is {node.get('class_type')}, expected {spec['class_type']}")
            for param, spec in self.mapping.get("inputs", {}).items():
                if not self._applies(spec, mode):
                    continue
                node = graph.get(self.node_id(spec["node"]))
                if node is not None and spec["input"] not in node.get("inputs", {}):
                    problems.append(f"{mode.value}: node {spec['node']} has no input {spec['input']!r} for {param}")
            for m in self.required_models:
                node = graph.get(m.node_id)
                if node is not None and node["inputs"].get(m.input) != m.file:
                    problems.append(f"{mode.value}: node {m.node_id}.{m.input} is {node['inputs'].get(m.input)!r}, expected {m.file!r}")

        # Mode-specific wiring: t2v must not need an image; i2v must feed the uploaded image.
        link = self.mapping.get("i2v_link", {})
        if problems:  # wiring checks below assume well-formed graphs
            raise WorkflowError("Invalid ComfyUI workflow templates:\n  - " + "\n  - ".join(problems))
        if link:
            target, source = self.node_id(link["node"]), self.node_id(link["from"])
            t2v_inputs = self.templates.get(GenerationMode.t2v, {}).get(target, {}).get("inputs", {})
            i2v_inputs = self.templates.get(GenerationMode.i2v, {}).get(target, {}).get("inputs", {})
            if link["input"] in t2v_inputs:
                problems.append(f"t2v: node {target} must not have {link['input']!r} connected")
            if i2v_inputs.get(link["input"]) != [source, 0]:
                problems.append(f"i2v: node {target}.{link['input']} must be connected to node {source}")
            if any(n.get("class_type") == "LoadImage" for n in self.templates.get(GenerationMode.t2v, {}).values()):
                problems.append("t2v: template must not contain a LoadImage node")

        output = self.mapping.get("output_node")
        if not output:
            problems.append("mapping has no output_node")
        if problems:
            raise WorkflowError("Invalid ComfyUI workflow templates:\n  - " + "\n  - ".join(problems))

    # ------------------------------------------------------------------ building
    def build(self, mode: GenerationMode, values: GraphValues) -> dict[str, Any]:
        """Return a NEW graph (the template is never modified) with all values filled in."""
        if mode == GenerationMode.i2v and not values.image:
            raise WorkflowError("image-to-video requires an uploaded image reference")
        graph = copy.deepcopy(self.templates[mode])
        data = {
            "prompt": values.prompt,
            # A user-provided negative prompt replaces the official default entirely.
            "negative_prompt": values.negative_prompt.strip() or self.default_negative_prompt,
            "width": values.width,
            "height": values.height,
            "num_frames": values.num_frames,
            "fps": values.fps,
            "steps": values.steps,
            "guidance_scale": values.guidance_scale,
            "seed": values.seed,
            "image": values.image,
            "filename_prefix": values.filename_prefix,
        }
        for param, spec in self.mapping["inputs"].items():
            if not self._applies(spec, mode):
                continue
            graph[self.node_id(spec["node"])]["inputs"][spec["input"]] = data[param]
        return graph
