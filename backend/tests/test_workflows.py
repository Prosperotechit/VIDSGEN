"""Structural validation of the ComfyUI workflow templates and the parameter mapping.

These checks run offline against ``fixtures/comfyui/object_info.json`` (captured from a real
ComfyUI 0.46.0). They prove the graphs are well-formed for that ComfyUI version. They do NOT
prove video quality or that generation succeeds on a GPU; that is Phase 5.
"""

import copy
import json
import shutil
from pathlib import Path

import pytest
import yaml

from app.core.config import REPO_ROOT
from app.jobs.models import SUPPORTED_RESOLUTIONS, GenerationMode
from app.workflows import GraphValues, WorkflowError, WorkflowTemplates
from comfyui_support import OBJECT_INFO, validate_graph

WF_DIR = REPO_ROOT / "workflows" / "comfyui"


@pytest.fixture(scope="module")
def wt() -> WorkflowTemplates:
    return WorkflowTemplates(WF_DIR)


def values(**kw) -> GraphValues:
    base = dict(prompt="a red fox", negative_prompt="", width=832, height=480, num_frames=49, fps=24, steps=30,
                guidance_scale=5.0, seed=42, filename_prefix="vidsgen/job_abc", image=None)
    base.update(kw)
    return GraphValues(**base)


# ---------------------------------------------------------------- templates as exported
@pytest.mark.parametrize("mode", list(GenerationMode))
def test_exported_templates_match_comfyui_schema(wt, mode):
    assert validate_graph(wt.templates[mode]) == []


def test_templates_are_api_format(wt):
    for graph in wt.templates.values():
        assert all(set(n) >= {"class_type", "inputs"} for n in graph.values())
        assert "nodes" not in graph and "links" not in graph   # not the UI format


def test_t2v_needs_no_image_and_i2v_loads_one(wt):
    t2v, i2v = wt.templates[GenerationMode.t2v], wt.templates[GenerationMode.i2v]
    assert not any(n["class_type"] == "LoadImage" for n in t2v.values())
    assert "start_image" not in t2v["55"]["inputs"]
    assert i2v["56"]["class_type"] == "LoadImage"
    assert i2v["55"]["inputs"]["start_image"] == ["56", 0]
    # Apart from the image branch, both modes are the same official graph.
    stripped = copy.deepcopy(i2v)
    del stripped["56"]
    del stripped["55"]["inputs"]["start_image"]
    assert stripped == t2v


def test_official_5b_model_files_only(wt):
    manifest = yaml.safe_load((WF_DIR / "models.yaml").read_text())
    expected = {(Path(f["comfyui_dir"]).name, Path(f["path"]).name) for f in manifest["files"]}
    assert {(m.folder, m.file) for m in wt.required_models} == expected
    for graph in wt.templates.values():
        assert graph["37"]["inputs"]["unet_name"] == "wan2.2_ti2v_5B_fp16.safetensors"
        assert graph["38"]["inputs"] == {"clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors", "type": "wan", "device": "default"}
        assert graph["39"]["inputs"]["vae_name"] == "wan2.2_vae.safetensors"
        text = json.dumps(graph)
        assert "14B" not in text and "high_noise" not in text and "low_noise" not in text and "wan_2.1_vae" not in text


def test_output_is_mp4_h264(wt):
    for graph in wt.templates.values():
        save = graph["58"]["inputs"]
        assert (save["format"], save["format.codec"]) == ("mp4", "h264")


def test_provenance_matches_templates(wt):
    prov = json.loads((WF_DIR / "export_provenance.json").read_text())
    assert prov["official_workflow"]["path"] == "templates/video_wan2_2_5B_ti2v.json"
    assert prov["exported_with"]["comfyui_version"] == "0.46.0"
    assert sorted(prov["node_classes"]) == wt.node_classes
    assert set(wt.node_classes) <= set(OBJECT_INFO)


# ---------------------------------------------------------------- mapping / building
def test_every_vidsgen_value_range_fits_comfyui(wt):
    """All values the VIDSGEN API accepts produce graphs ComfyUI's schema accepts."""
    for mode in GenerationMode:
        for res, (w, h) in SUPPORTED_RESOLUTIONS.items():
            for frames, fps, steps, cfg, seed in [(17, 8, 1, 1.0, 0), (121, 30, 60, 15.0, 2**32 - 1), (49, 24, 30, 5.0, 123)]:
                g = wt.build(mode, values(width=w, height=h, num_frames=frames, fps=fps, steps=steps, guidance_scale=cfg,
                                          seed=seed, image="vidsgen/job_abc.png" if mode == GenerationMode.i2v else None))
                assert validate_graph(g) == [], (mode, res, frames)


def test_build_maps_every_parameter(wt):
    g = wt.build(GenerationMode.i2v, values(prompt="ocean waves", negative_prompt="blurry", width=1280, height=704,
                                            num_frames=121, fps=16, steps=25, guidance_scale=6.5, seed=7,
                                            filename_prefix="vidsgen/job_xyz", image="vidsgen/job_xyz.png"))
    assert g["6"]["inputs"]["text"] == "ocean waves"
    assert g["7"]["inputs"]["text"] == "blurry"
    assert (g["55"]["inputs"]["width"], g["55"]["inputs"]["height"], g["55"]["inputs"]["length"]) == (1280, 704, 121)
    assert g["57"]["inputs"]["fps"] == 16
    assert (g["3"]["inputs"]["steps"], g["3"]["inputs"]["cfg"], g["3"]["inputs"]["seed"]) == (25, 6.5, 7)
    assert g["56"]["inputs"]["image"] == "vidsgen/job_xyz.png"
    assert g["58"]["inputs"]["filename_prefix"] == "vidsgen/job_xyz"
    # Fixed official settings are untouched.
    assert (g["3"]["inputs"]["sampler_name"], g["3"]["inputs"]["scheduler"], g["3"]["inputs"]["denoise"]) == ("uni_pc", "simple", 1)
    assert g["48"]["inputs"]["shift"] == 8


def test_default_negative_prompt_is_the_official_one(wt):
    official = json.loads((WF_DIR / "wan22_ti2v_5b_t2v.json").read_text())["7"]["inputs"]["text"]
    assert wt.default_negative_prompt == official
    assert wt.build(GenerationMode.t2v, values(negative_prompt=""))["7"]["inputs"]["text"] == official
    assert wt.build(GenerationMode.t2v, values(negative_prompt="   "))["7"]["inputs"]["text"] == official


def test_user_negative_prompt_replaces_default(wt):
    g = wt.build(GenerationMode.t2v, values(negative_prompt="text, watermark"))
    assert g["7"]["inputs"]["text"] == "text, watermark"
    assert wt.default_negative_prompt not in json.dumps(g, ensure_ascii=False)


def test_build_does_not_modify_templates(wt):
    before = copy.deepcopy(wt.templates)
    wt.build(GenerationMode.i2v, values(image="vidsgen/x.png", prompt="changed"))
    assert wt.templates == before


def test_i2v_requires_image_reference(wt):
    with pytest.raises(WorkflowError, match="requires an uploaded image"):
        wt.build(GenerationMode.i2v, values(image=None))


# ---------------------------------------------------------------- failure modes (fail fast)
def _copy(tmp_path: Path) -> Path:
    d = tmp_path / "wf"
    shutil.copytree(WF_DIR, d, ignore=shutil.ignore_patterns("tools"))
    return d


def _edit(path: Path, fn) -> None:
    data = json.loads(path.read_text())
    fn(data)
    path.write_text(json.dumps(data))


@pytest.mark.parametrize("edit,message", [
    (lambda g: g.pop("3"), "missing node sampler"),
    (lambda g: g["58"].update(class_type="SaveImage"), "node 58 is SaveImage, expected SaveVideo"),
    (lambda g: g["38"]["inputs"].update(clip_name="umt5_xxl_fp16.safetensors"), "expected 'umt5_xxl_fp8_e4m3fn_scaled.safetensors'"),
    (lambda g: g["55"]["inputs"].pop("length"), "has no input 'length'"),
    (lambda g: g["55"]["inputs"].update(start_image=["56", 0]), "must not have 'start_image'"),
])
def test_invalid_t2v_template_fails_fast(tmp_path, edit, message):
    d = _copy(tmp_path)
    _edit(d / "wan22_ti2v_5b_t2v.json", edit)
    with pytest.raises(WorkflowError, match=message):
        WorkflowTemplates(d)


def test_i2v_without_image_wiring_fails_fast(tmp_path):
    d = _copy(tmp_path)
    _edit(d / "wan22_ti2v_5b_i2v.json", lambda g: g["55"]["inputs"].pop("start_image"))
    with pytest.raises(WorkflowError, match="must be connected to node 56"):
        WorkflowTemplates(d)


def test_missing_or_corrupt_files_fail_fast(tmp_path):
    d = _copy(tmp_path)
    (d / "wan22_ti2v_5b_i2v.json").unlink()
    with pytest.raises(WorkflowError, match="template not found"):
        WorkflowTemplates(d)
    d2 = _copy(tmp_path / "b")
    (d2 / "wan22_ti2v_5b.mapping.json").write_text("{not json")
    with pytest.raises(WorkflowError, match="not valid JSON"):
        WorkflowTemplates(d2)
    with pytest.raises(WorkflowError, match="mapping not found"):
        WorkflowTemplates(tmp_path / "nowhere")


def test_ui_format_template_is_rejected(tmp_path):
    d = _copy(tmp_path)
    (d / "wan22_ti2v_5b_t2v.json").write_text(json.dumps({"nodes": [], "links": [], "version": 0.4}))
    with pytest.raises(WorkflowError, match="not in API format"):
        WorkflowTemplates(d)


def test_validator_catches_real_schema_violations(wt):
    """The offline validator itself must catch the kinds of drift we care about."""
    g = wt.build(GenerationMode.t2v, values())
    bad = copy.deepcopy(g)
    bad["55"]["inputs"]["width"] = 833                       # not a multiple of 32
    bad["58"]["inputs"]["format.codec"] = "prores"           # not offered for mp4
    bad["57"]["inputs"]["frame_rate"] = 24                   # input that doesn't exist
    del bad["3"]["inputs"]["cfg"]                            # required input missing
    bad["8"]["inputs"]["vae"] = ["37", 0]                    # MODEL linked into a VAE input
    bad["57"]["inputs"]["color_space"] = "Rec709"            # V3 COMBO value not offered
    errors = "\n".join(validate_graph(bad))
    for expected in ["833 not on step 32", "'prores' not in", "unexpected input frame_rate",
                     "missing required input cfg", "link type MODEL != VAE", "'Rec709' not in"]:
        assert expected in errors
