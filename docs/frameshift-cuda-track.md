# FrameShift — CUDA Track (Document 2 of 2)

The hand-written GPU work, built and validated on your own machine, **kept off the deploy critical path** so it never holds the product hostage. Two tracks in ascending difficulty: **Track A** a warp kernel (exact `grid_sample` oracle — the clean, low-risk resume piece), **Track B** the Ezsynth guided-synthesis kernel (the crown jewel; powers Backend B in Document 1).

---

## 0. The decoupling (the rule everything hangs on)

Production ships on torch; the kernel is a parallel track behind a flag, so neither goal blocks the other.

```python
# product-level selection
layers = ( synth_propagate_cuda(...)   if USE_SYNTH and USE_CUDA_KERNEL      # Track B kernel
      else synth_propagate_cpu(...)    if USE_SYNTH                          # pip ebsynth / Ezsynth ref
      else flow_propagate(...) )                                             # Doc 1 Backend A — always works

# op-level selection (warp)
out = warp_ops.warp(img, flow) if USE_CUDA_KERNEL else F.grid_sample(img, grid, ...)   # Track A
```

- **Deploy / production default:** `USE_CUDA_KERNEL = False`. Nothing depends on your `.so` compiling on Modal's `debian-slim`.
- **Dev / benchmark:** `True`, where you control CUDA/libtorch versions.

**What is and isn't proof.** For **Track A (warp)**, `grid_sample` *is* the same operation → identical output is a true correctness proof, and the benchmark line ("matched `grid_sample` numerically, N ms vs M ms at 720p") is exact. For **Track B (synthesis)**, `grid_sample` is *not* a drop-in — it's an exact oracle only for the degenerate voting sub-case (patch radius 0, given NNF). The full PatchMatch kernel is randomized, so it's validated by **PSNR/SSIM against the CPU Ezsynth reference** (quality match, not bytes), and its product-level fallback is **Backend A**, not `grid_sample`. Keep the two claims separate and both stay true.

---

# TRACK A — Warp kernel (exact `grid_sample` oracle)

The contained, low-risk piece. Reimplements the *apply* half of transport — a bilinear gather along a flow field — which torch already provides as `grid_sample`, giving you a perfect oracle and a one-line fallback.

## A.1 Find vs apply

Transport = **find** the correspondence (RAFT) + **apply** it (gather). `grid_sample` is *apply* only. This kernel reimplements *apply*: embarrassingly parallel (one thread per output pixel, no cross-thread coordination) — the right first kernel.

## A.2 Layout

```
flow_warp_ops/
  warp_kernels.cu     # backward_warp, compose_flow, fb_check
  bindings.cpp        # torch::Tensor wrappers + PYBIND11_MODULE
  setup.py            # CUDAExtension
```

## A.3 Backward-warp kernel

Match `grid_sample`'s convention (`align_corners=False`, pixel centers `+0.5`) exactly, or the diff won't be zero — that mismatch is the #1 way this wastes an afternoon.

```cpp
#include <cuda.h>
__device__ __forceinline__ float rd(const float* im,int c,int y,int x,int C,int H,int W){
    if(x<0||x>=W||y<0||y>=H) return 0.f;              // zeros padding
    return im[(c*H+y)*W+x];
}
__global__ void backward_warp_kernel(
        const float* __restrict__ img, const float* __restrict__ flow,
        float* __restrict__ out, int C,int H,int W){
    int x=blockIdx.x*blockDim.x+threadIdx.x, y=blockIdx.y*blockDim.y+threadIdx.y;
    if(x>=W||y>=H) return;
    float sx = x + flow[(0*H+y)*W+x];                 // absolute source coord
    float sy = y + flow[(1*H+y)*W+x];
    int x0=floorf(sx), y0=floorf(sy), x1=x0+1, y1=y0+1;
    float ax=sx-x0, ay=sy-y0;
    for(int c=0;c<C;++c){
        float v = (1-ax)*(1-ay)*rd(img,c,y0,x0,C,H,W)+ax*(1-ay)*rd(img,c,y0,x1,C,H,W)
                + (1-ax)*   ay *rd(img,c,y1,x0,C,H,W)+ax*   ay *rd(img,c,y1,x1,C,H,W);
        out[(c*H+y)*W+x]=v;
    }
}
void backward_warp_launch(const float* img,const float* flow,float* out,int C,int H,int W){
    dim3 b(16,16), g((W+b.x-1)/b.x,(H+b.y-1)/b.y);
    backward_warp_kernel<<<g,b>>>(img,flow,out,C,H,W);
}
```

`compose_flow` is the same gather applied to a flow field (builds `F_{N→A}`); `fb_check` is the per-pixel forward-backward consistency test → validity mask. Both per-pixel, same shape.

## A.4 Bindings + build

```cpp
#include <torch/extension.h>
void backward_warp_launch(const float*,const float*,float*,int,int,int);
torch::Tensor warp(torch::Tensor img,torch::Tensor flow){
    TORCH_CHECK(img.is_cuda()&&flow.is_cuda(),"cuda tensors required");
    img=img.contiguous(); flow=flow.contiguous();
    int C=img.size(0),H=img.size(1),W=img.size(2);
    auto out=torch::zeros_like(img);
    backward_warp_launch(img.data_ptr<float>(),flow.data_ptr<float>(),out.data_ptr<float>(),C,H,W);
    return out;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){ m.def("warp",&warp,"bilinear backward warp (CUDA)"); }
```

```python
# setup.py
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
setup(name="flow_warp_ops",
      ext_modules=[CUDAExtension("flow_warp_ops",["bindings.cpp","warp_kernels.cu"])],
      cmdclass={"build_ext": BuildExtension})
```

JIT alternative (`torch.utils.cpp_extension.load`) compiles on first import — still needs nvcc + host compiler present.

## A.5 Validation & the resume payoff

```python
import torch, torch.nn.functional as F
img  = torch.rand(3,720,1280,device="cuda")
flow = (torch.rand(2,720,1280,device="cuda")-0.5)*20
mine = flow_warp_ops.warp(img, flow)
ref  = warp_grid_sample(img[None], flow[None])[0]   # same align_corners convention
print((mine-ref).abs().max())                        # target ~0 (float rounding aside)
```

Non-zero max diff → almost always an `align_corners`/pixel-center mismatch or padding-mode difference. The diff *is* the proof. Benchmark both with `torch.cuda.Event` at 720p → the exact interview line: "hand-wrote a CUDA backward-warp kernel, verified numerically against `grid_sample`, measured X ms vs Y ms at 1280×720."

---

# TRACK B — Ezsynth guided-synthesis kernel (crown jewel)

Powers Backend B in Document 1. Reimplements the *find* half EbSynth uses — guided PatchMatch synthesis — which has **no torch primitive**, hence a custom kernel. Randomized, so validated by PSNR/SSIM against the CPU reference, not byte-equality.

## B.1 The energy

`S` = edited anchor, `S'` = original anchor, `T'` = original target, guides `G`. For each target patch `p`, PatchMatch finds source patch `q` minimizing:

```
E(p,q) = ‖ S'_patch(q) − T'_patch(p) ‖²              # appearance (match in ORIGINAL domain)
       + Σ_g λ_g · ‖ G_g^src(q) − G_g^tgt(p) ‖²        # edge, positional, temporal guides
```

Match in the guide/original domain; **copy pixels from the edited `S`** → an *edit* propagator, not a style-averager. Solved by alternating NNF search (PatchMatch) + voting (reconstruction), coarse-to-fine on a pyramid. Guides (appearance/edge/positional/temporal) are built on Backend A's cached RAFT flow — see Doc 1 §3.2.

## B.2 PatchMatch, and the GPU-hard part

Three moves: random **init** (or positional-guide seed); **propagation** (inherit neighbor's offset, shifted — good matches flood outward); **random search** (refine at shrinking radii). Converges in ~O(n).

**Propagation is the hard part on a GPU** — on CPU it's inherently sequential (scan order; each patch reads its neighbor's just-updated offset). Break the dependency with:
- **Jump-flood propagation** — `log(n)` fully-parallel passes at strides `n/2…1`. (Used below.)
- **Diagonal wavefronts / tiled** — anti-diagonals independent, or per-tile in shared memory then reconcile borders.

Getting propagation correct *and* parallel is the resume-defining systems problem. Reconstruction is embarrassingly parallel.

## B.3 Layout

```
ebsynth_synth/
  patchmatch.cu   # nnf init, jump-flood propagation, random search, patch-distance
  voting.cu       # reconstruction (patch voting) — generalizes grid_sample
  guides.cu       # edge/positional/temporal (or build in torch)
  pyramid.cpp     # coarse-to-fine EM host loop
  bindings.cpp    setup.py
```

NNF = `int2` per pixel (source x,y) + `float` best cost. Patch radius `r` (EbSynth default 2 → 5×5). Distance sums over the window across appearance + guide channels with weights `λ`.

## B.4 Patch distance

```cpp
__device__ float patch_dist(const float* Ssrc,const float* Ttgt,const float* lambda,int C,
                            int sx,int sy,int tx,int ty,int H,int W,int r,float best){
    float acc=0.f;
    for(int dy=-r;dy<=r;++dy) for(int dx=-r;dx<=r;++dx){
        int sxx=sx+dx,syy=sy+dy,txx=tx+dx,tyy=ty+dy;
        if(sxx<0||sxx>=W||syy<0||syy>=H||txx<0||txx>=W||tyy<0||tyy>=H){acc+=1e3f;continue;}
        for(int c=0;c<C;++c){float a=Ssrc[(c*H+syy)*W+sxx]-Ttgt[(c*H+tyy)*W+txx]; acc+=lambda[c]*a*a;}
        if(acc>=best) return acc;                    // early-out
    }
    return acc;
}
```

## B.5 Jump-flood propagation (one pass, fully parallel)

```cpp
__global__ void jump_flood_pass(int2* nnf,float* cost,
        const float* Ssrc,const float* Ttgt,const float* lambda,
        int C,int H,int W,int r,int step){
    int tx=blockIdx.x*blockDim.x+threadIdx.x, ty=blockIdx.y*blockDim.y+threadIdx.y;
    if(tx>=W||ty>=H) return; int t=ty*W+tx;
    int2 best=nnf[t]; float bestc=cost[t];
    const int off[8][2]={{-step,0},{step,0},{0,-step},{0,step},
                         {-step,-step},{step,step},{-step,step},{step,-step}};
    #pragma unroll
    for(int k=0;k<8;++k){
        int nx=tx+off[k][0], ny=ty+off[k][1];
        if(nx<0||nx>=W||ny<0||ny>=H) continue;
        int2 cand=nnf[ny*W+nx];
        int cx=cand.x+(tx-nx), cy=cand.y+(ty-ny);    // neighbor's source, shifted to me
        if(cx<0||cx>=W||cy<0||cy>=H) continue;
        float d=patch_dist(Ssrc,Ttgt,lambda,C,cx,cy,tx,ty,H,W,r,bestc);
        if(d<bestc){bestc=d;best=make_int2(cx,cy);}
    }
    nnf[t]=best; cost[t]=bestc;
}
```

Host loops `step=W/2…1` (halving); each launch fully parallel — no scan-order dependency. Interleave with random search.

## B.6 Random search

```cpp
__global__ void random_search(int2* nnf,float* cost,
        const float* Ssrc,const float* Ttgt,const float* lambda,
        int C,int H,int W,int r,unsigned seed){
    int tx=blockIdx.x*blockDim.x+threadIdx.x, ty=blockIdx.y*blockDim.y+threadIdx.y;
    if(tx>=W||ty>=H) return; int t=ty*W+tx;
    int2 best=nnf[t]; float bestc=cost[t];
    curandStatePhilox4_32_10_t st; curand_init(seed,t,0,&st);
    for(int radius=max(W,H); radius>=1; radius>>=1){
        int cx=best.x+(int)((curand_uniform(&st)*2-1)*radius);
        int cy=best.y+(int)((curand_uniform(&st)*2-1)*radius);
        cx=min(max(cx,0),W-1); cy=min(max(cy,0),H-1);
        float d=patch_dist(Ssrc,Ttgt,lambda,C,cx,cy,tx,ty,H,W,r,bestc);
        if(d<bestc){bestc=d;best=make_int2(cx,cy);}
    }
    nnf[t]=best; cost[t]=bestc;
}
```

## B.7 Voting / reconstruction (generalizes `grid_sample`)

Each output pixel averages the *edited* source pixels from every patch covering it. **Patch radius 0 → this collapses to a per-pixel gather along the NNF = `grid_sample`.** That degenerate case is your one exact oracle in Track B.

```cpp
__global__ void vote(const int2* nnf,const float* Sedit,float* out,float* wsum,
                     int C,int H,int W,int r){
    int tx=blockIdx.x*blockDim.x+threadIdx.x, ty=blockIdx.y*blockDim.y+threadIdx.y;
    if(tx>=W||ty>=H) return;
    for(int dy=-r;dy<=r;++dy) for(int dx=-r;dx<=r;++dx){
        int px=tx+dx,py=ty+dy; if(px<0||px>=W||py<0||py>=H) continue;
        int2 s=nnf[py*W+px]; int sx=s.x-dx, sy=s.y-dy;
        if(sx<0||sx>=W||sy<0||sy>=H) continue;
        for(int c=0;c<C;++c) atomicAdd(&out[(c*H+ty)*W+tx], Sedit[(c*H+sy)*W+sx]);
        atomicAdd(&wsum[ty*W+tx],1.f);
    }
}   // then out /= wsum
```

## B.8 EM loop + pyramid (host)

```
build gaussian pyramid of {Sedit, Ssrc, Ttgt, guides}
for level in coarse..fine:
    upsample NNF from previous level (or seed from positional guide)
    repeat K times (EM):
        for step in W/2..1: jump_flood_pass
        random_search
        vote -> current T; refresh temporal guide if used
```

Typical: 5×5 patches, ~6 EM iters/level, 4–5 pyramid levels.

## B.9 Bindings + build

```cpp
#include <torch/extension.h>
torch::Tensor synthesize(torch::Tensor Sedit,torch::Tensor Ssrc,torch::Tensor Ttgt,
                         torch::Tensor guides,torch::Tensor lambda,int patch_r,int iters);
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){ m.def("synthesize",&synthesize); }
```

```python
# setup.py
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
setup(name="ebsynth_synth",
      ext_modules=[CUDAExtension("ebsynth_synth",
                   ["bindings.cpp","patchmatch.cu","voting.cu","pyramid.cpp"])],
      cmdclass={"build_ext": BuildExtension})
```

## B.10 Validation & benchmark

- **Voting sub-case (exact):** patch radius 0 + a hand-made NNF → `vote` must equal `grid_sample` with the same coords. Proves the gather + convention (shared with Track A).
- **Full synthesis (quality):** run pip `ebsynth`/Ezsynth and your kernel on the same anchor + guides; compare reconstructions by **PSNR/SSIM** (not bytes — PatchMatch is randomized). Fix seeds where possible.
- **Benchmark:** full-clip propagation vs Backend A and vs CPU Ezsynth. Reference: Ezsynth-class runs ~100 frames at ~960×544, 2 keyframes on a mid GPU. Line: "implemented guided PatchMatch synthesis as a CUDA extension (jump-flood propagation + patch voting); matched Ezsynth reconstruction quality at N× CPU throughput."

## B.11 Integration behind Backend B

```python
def ebsynth_synthesize(anchor_edit, guides, masks):
    if USE_CUDA_KERNEL:
        return ebsynth_synth.synthesize(anchor_edit.S, guides.S_orig, guides.T_orig,
                                        guides.stack, guides.lambda_, patch_r=2, iters=6)
    return ezsynth_cpu(anchor_edit, guides, masks)     # reference
```

Output shape matches Doc 1 Backend A, so the production composite is unchanged regardless of backend or implementation.

---

## Shared: toolchain reality (why both tracks stay off the deploy path)

`setup.py build_ext` / `load()` invokes **nvcc** on the `.cu` and your **host C++ compiler** on the `.cpp`, links against *this machine's* libtorch + CUDA runtime, and emits a `.so`/`.pyd` bound to that ABI. Ship it to a box with a different CUDA minor version, torch build, or too-old driver → "undefined symbol" / "DLL load failed" (the errors on the Ezsynth issues tracker). To build at all you need: CUDA Toolkit (nvcc + headers), a matching host compiler, and a torch whose CUDA build matches the toolkit. Modal's `debian-slim` has none of it, and reproducing the version agreement in a container is the classic "works on my laptop, dies in the box." Hence: kernels local; Backend A / `grid_sample` ship.

---

## Build order (whole track)

1. **Track A** warp kernel + `compose_flow` + `fb_check`; JIT `load()`; get ~0 diff vs `grid_sample`; benchmark. *(Contained, safe, complete on its own — stop here and you already have a strong resume line.)*
2. **Track B voting kernel** first (easy) → validate radius-0 vs `grid_sample`; locks data layout + convention.
3. **Track B patch-distance + random search** (per-pixel).
4. **Track B jump-flood propagation** — the hard part; verify NNF quality vs CPU PatchMatch on a toy image.
5. **EM loop + pyramid**; add guides one at a time (appearance → edge → positional → temporal).
6. **Validate vs Ezsynth (PSNR/SSIM)** and benchmark; wire behind Backend B with Backend A as the always-on fallback.
