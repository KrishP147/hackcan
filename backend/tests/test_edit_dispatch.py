import asyncio
from unittest.mock import AsyncMock, patch

from services import edit_dispatch


def _rule(**kw):
    from main import EditRule
    return EditRule(**{"edit_type": "recolor", "start_frame": 1, "end_frame": 3, **kw})


def test_deterministic_rule_hits_every_frame(tmp_project):
    for t in (1, 2, 3):
        (tmp_project / "frames" / f"frame_{t:04d}.jpg").write_bytes(b"")
    with patch.object(edit_dispatch, "_ensure_flows"), \
         patch.object(edit_dispatch, "_ensure_masks"), \
         patch.object(edit_dispatch, "_project_dir", return_value=tmp_project), \
         patch.object(edit_dispatch.project_manager, "get_status", return_value={}), \
         patch.object(edit_dispatch.local_edit_service, "apply_recolor") as rec:
        asyncio.run(edit_dispatch.run_edit_rule("pid", _rule(color="00FF00")))
        assert rec.call_count == 3


def test_replace_routes_to_propagation_engine(tmp_project):
    with patch.object(edit_dispatch, "_ensure_flows"), \
         patch.object(edit_dispatch, "_ensure_masks"), \
         patch.object(edit_dispatch, "_project_dir", return_value=tmp_project), \
         patch.object(edit_dispatch.project_manager, "get_status", return_value={}), \
         patch.object(edit_dispatch.replace_tool, "apply_replace_range",
                      new_callable=AsyncMock) as rep:
        asyncio.run(edit_dispatch.run_edit_rule(
            "pid", _rule(edit_type="replace", prompt="a dog")))
        rep.assert_awaited_once()
        args = rep.await_args.args
        assert "a dog" in args
