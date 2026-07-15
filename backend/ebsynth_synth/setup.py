from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


ROOT = Path(__file__).resolve().parent

setup(
    name="frameshift-ebsynth-synth",
    ext_modules=[
        CUDAExtension(
            "ebsynth_synth._C",
            [
                str(ROOT / "bindings.cpp"),
                str(ROOT / "patchmatch.cu"),
                str(ROOT / "voting.cu"),
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math", "-lineinfo"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
