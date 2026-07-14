import asyncio
from unittest.mock import AsyncMock, patch

import torch

from services import synth_propagation_service


def test_stub_falls_back_to_backend_a(tmp_project):
    with patch.object(synth_propagation_service.replace_tool, "apply_replace_range",
                      new_callable=AsyncMock) as a:
        asyncio.run(synth_propagation_service.apply_replace_range(
            tmp_project, [1, 2], 1, "x", torch.device("cpu")))
        a.assert_awaited_once()
