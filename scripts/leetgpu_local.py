#!/usr/bin/env python3
"""
Local dev harness for LeetGPU challenges (private-fork tool, CUDA only).

Runs a challenge entirely on the local GPU: it compiles a solution.cu with nvcc,
calls `solve` via ctypes with device pointers, and diffs the result against
challenge.py's `reference_impl` under the challenge tolerances. It also benchmarks
the perf case (shifted geometric mean over N runs), optionally head-to-head against
a reference.cu you paste a known-good solution into.

Timing is measured on the local GPU (e.g. an A40) -- directional only.

The `profile` command generates a standalone CUDA driver (profile.cu) plus raw
input-data blobs for the perf case, so you can build a pure CUDA binary and point
any profiler at it (ncu / nsys / compute-sanitizer) with no python or torch in the
process -- the only kernels it runs are solve()'s.

Run with the CUDA-enabled venv, e.g.:
    .venv-cuda/bin/python scripts/leetgpu_local.py setup challenges/easy/1_vector_add
    .venv-cuda/bin/python scripts/leetgpu_local.py test 1_vector_add
    .venv-cuda/bin/python scripts/leetgpu_local.py profile 1_vector_add
"""

import argparse
import ctypes
import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CHALLENGES_DIR = REPO_ROOT / "challenges"
WORKSPACE = REPO_ROOT / "workspace"

# Make `core.challenge_base` importable (challenge.py files import from it too).
sys.path.insert(0, str(CHALLENGES_DIR))
from core.challenge_base import (  # noqa: E402
    FullTensor,
    OutTensor,
    RandIntTensor,
    RandnTensor,
    RandTensor,
)

import torch  # noqa: E402

DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "float64": torch.float64,
    "int32": torch.int32,
    "int64": torch.int64,
    "int8": torch.int8,
    "uint8": torch.uint8,
    "uint32": getattr(torch, "uint32", torch.int32),
    "bool": torch.bool,
}


# --------------------------------------------------------------------------- #
# challenge loading + spec materialization
# --------------------------------------------------------------------------- #
def load_challenge(challenge_dir: Path, device: str = "cuda"):
    challenge_py = challenge_dir / "challenge.py"
    if not challenge_py.exists():
        raise FileNotFoundError(f"No challenge.py in {challenge_dir}")
    spec = importlib.util.spec_from_file_location("challenge", challenge_py)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        return module.Challenge(device=device)
    finally:
        sys.modules.pop("challenge", None)


def materialize(v, device: str):
    """Turn a perf-test spec object into a concrete tensor; pass tensors/scalars through."""
    if torch.is_tensor(v):
        return v
    if isinstance(v, OutTensor):
        return torch.empty(v.shape, dtype=DTYPES[v.dtype], device=device)
    if isinstance(v, FullTensor):
        return torch.full(v.shape, v.value, dtype=DTYPES[v.dtype], device=device)
    if isinstance(v, RandTensor):
        return torch.empty(v.shape, dtype=DTYPES[v.dtype], device=device).uniform_(v.low, v.high)
    if isinstance(v, RandnTensor):
        return torch.empty(v.shape, dtype=DTYPES[v.dtype], device=device).normal_(v.mean, v.std)
    if isinstance(v, RandIntTensor):
        return torch.randint(v.low, v.high, tuple(v.shape), device=device).to(DTYPES[v.dtype])
    return v  # raw scalar


# --------------------------------------------------------------------------- #
# ctypes signature helpers
# --------------------------------------------------------------------------- #
def is_pointer(ctype) -> bool:
    return isinstance(ctype, type) and issubclass(ctype, ctypes._Pointer)


def ensure_cuda_signature(name: str, signature: dict):
    """Reject challenges whose solve signature isn't a plain CUDA C ABI (e.g.
    challenges that pass torch tensors / nn.Modules) -- the CUDA harness can't run them."""
    for pname, (ct, _dir) in signature.items():
        scalar = isinstance(ct, type) and issubclass(ct, ctypes._SimpleCData)
        if not (is_pointer(ct) or scalar):
            raise SystemExit(
                f"'{name}' is not a CUDA-ABI challenge (param '{pname}' is {ct!r}); "
                "the local CUDA harness only supports pointer/scalar solve signatures."
            )


def bind_solve(lib_path: Path, signature: dict):
    lib = ctypes.CDLL(str(lib_path))
    solve = lib.solve
    solve.restype = None
    solve.argtypes = [
        ctypes.c_void_p if is_pointer(ct) else ct for (ct, _dir) in signature.values()
    ]
    return solve


def build_args(signature: dict, kwargs: dict):
    args = []
    for name, (ct, _dir) in signature.items():
        v = kwargs[name]
        args.append(ctypes.c_void_p(v.data_ptr()) if is_pointer(ct) else v)
    return args


# --------------------------------------------------------------------------- #
# nvcc compile
# --------------------------------------------------------------------------- #
def default_arch() -> str:
    major, minor = torch.cuda.get_device_capability()
    return f"{major}{minor}"


def driver_cuda_major() -> int | None:
    """Highest CUDA major version the installed driver supports (via libcuda), or None."""
    for name in ("libcuda.so.1", "libcuda.so"):
        try:
            drv = ctypes.CDLL(name)
        except OSError:
            continue
        v = ctypes.c_int()
        if drv.cuDriverGetVersion(ctypes.byref(v)) == 0:
            return v.value // 1000
    return None


def nvcc_version(path: str) -> tuple[int, int] | None:
    try:
        out = subprocess.run([path, "--version"], capture_output=True, text=True).stdout
    except OSError:
        return None
    m = re.search(r"release (\d+)\.(\d+)", out)
    return (int(m.group(1)), int(m.group(2))) if m else None


_NVCC_CACHE = None


def find_nvcc() -> str:
    """Pick an nvcc whose CUDA major version the driver supports.

    A toolkit newer than the driver (e.g. nvcc 13.x on a 12.4 driver) statically links
    a cudart the driver can't init, so kernels silently never run. CUDA minor-version
    compatibility only holds within a major series, so we require nvcc_major <= driver_major
    and pick the newest compatible candidate. Override with the NVCC env var.
    """
    global _NVCC_CACHE
    if _NVCC_CACHE is not None:
        return _NVCC_CACHE

    override = os.environ.get("NVCC")
    if override:
        _NVCC_CACHE = override
        return override

    candidates = []
    for cand in (shutil.which("nvcc"),
                 os.environ.get("CONDA_PREFIX") and f"{os.environ['CONDA_PREFIX']}/bin/nvcc"):
        if cand and cand not in candidates and Path(cand).exists():
            candidates.append(cand)

    driver_major = driver_cuda_major()
    scored = [(nvcc_version(c), c) for c in candidates]
    scored = [(v, c) for v, c in scored if v is not None]

    if driver_major is not None:
        compatible = [(v, c) for v, c in scored if v[0] <= driver_major]
        if compatible:
            v, c = max(compatible)
            if scored and max(scored)[0][0] > driver_major:
                print(f"note: driver supports CUDA {driver_major}.x; using nvcc {v[0]}.{v[1]} "
                      f"({c}) instead of a newer toolkit on PATH.")
            _NVCC_CACHE = c
            return c
        if scored:
            best_v, best_c = max(scored)
            print(f"WARNING: no nvcc <= CUDA {driver_major}.x found; falling back to "
                  f"{best_v[0]}.{best_v[1]} ({best_c}) — kernels may fail to launch. "
                  f"Set NVCC to a CUDA {driver_major}.x toolkit.")
            _NVCC_CACHE = best_c
            return best_c

    _NVCC_CACHE = candidates[0] if candidates else "nvcc"
    return _NVCC_CACHE


def compile_cu(src: Path, arch: str, out_dir: Path) -> Path:
    out = out_dir / f"{src.stem}.so"
    cmd = [
        find_nvcc(),
        "-O3",
        "-shared",
        "-Xcompiler",
        "-fPIC",
        f"-arch=sm_{arch}",
        str(src),
        "-o",
        str(out),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"nvcc failed for {src.name}:\n{proc.stderr.strip()}")
    return out


# --------------------------------------------------------------------------- #
# correctness
# --------------------------------------------------------------------------- #
def run_case(challenge, signature, solve, case, device):
    """Run one test case; return (passed, max_abs_err)."""
    base = {k: materialize(v, device) for k, v in case.items()}
    ref = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in base.items()}
    ker = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in base.items()}

    challenge.reference_impl(**ref)
    solve(*build_args(signature, ker))
    torch.cuda.synchronize()

    ok, worst = True, 0.0
    for name, (_ct, direction) in signature.items():
        if direction not in ("out", "inout"):
            continue
        k, r = ker[name], ref[name]
        if not torch.is_tensor(k) or k.numel() == 0:
            continue
        if k.is_floating_point():
            close = torch.allclose(k, r, atol=challenge.atol, rtol=challenge.rtol, equal_nan=True)
            err = (k.float() - r.float()).abs().max().item()
        else:
            close = torch.equal(k, r)
            err = (k - r).abs().max().item()
        ok = ok and close
        worst = max(worst, float(err))
    return ok, worst


def check_cases(challenge, signature, solve, cases, label, device) -> bool:
    all_ok, worst = True, 0.0
    n_pass = 0
    for case in cases:
        ok, err = run_case(challenge, signature, solve, case, device)
        all_ok = all_ok and ok
        n_pass += int(ok)
        worst = max(worst, err)
    tag = "PASS" if all_ok else "FAIL"
    print(f"  [{label:<11}] {tag}  ({n_pass}/{len(cases)} cases, max abs err {worst:.2e})")
    return all_ok


# --------------------------------------------------------------------------- #
# benchmark
# --------------------------------------------------------------------------- #
def shifted_geomean(times, shift: float) -> float:
    n = len(times)
    return math.exp(sum(math.log(t + shift) for t in times) / n) - shift


def time_solve(solve, args, warmup: int, runs: int):
    for _ in range(warmup):
        solve(*args)
    torch.cuda.synchronize()
    times = []
    for _ in range(runs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        solve(*args)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))  # ms
    return times


def benchmark(challenge, signature, sol_solve, ref_solve, device, runs, shift):
    perf = challenge.generate_performance_test()
    print(f"\nbenchmark (local GPU {torch.cuda.get_device_name()} — directional, NOT T4), "
          f"runs={runs}, sgm shift={shift}ms")

    def sgm_for(solve):
        kw = {k: materialize(v, device) for k, v in perf.items()}
        times = time_solve(solve, build_args(signature, kw), warmup=3, runs=runs)
        return shifted_geomean(times, shift), times

    sol_sgm, sol_times = sgm_for(sol_solve)
    runs_str = " ".join(f"{t:.3f}" for t in sol_times)
    print(f"  solution.cu    sgm {sol_sgm:.3f} ms   [{runs_str}]")
    if ref_solve is not None:
        ref_sgm, ref_times = sgm_for(ref_solve)
        runs_str = " ".join(f"{t:.3f}" for t in ref_times)
        print(f"  reference.cu   sgm {ref_sgm:.3f} ms   [{runs_str}]")
        if sol_sgm > 0:
            print(f"  speedup (ref/sol)  {ref_sgm / sol_sgm:.2f}x")


# --------------------------------------------------------------------------- #
# workspace resolution
# --------------------------------------------------------------------------- #
def resolve_workspace(name_or_path: str) -> Path:
    cand = WORKSPACE / name_or_path
    if (cand / "meta.json").exists():
        return cand
    p = Path(name_or_path)
    if (p / "meta.json").exists():
        return p.resolve()
    # Fall back to a bare challenge number/name (as `setup` accepts): map it to the
    # challenge folder, then look for the default-named workspace for that folder.
    try:
        folder = resolve_challenge_dir(name_or_path).name
    except FileNotFoundError:
        folder = None
    if folder is not None:
        by_folder = WORKSPACE / folder
        if (by_folder / "meta.json").exists():
            return by_folder
        hint = f" (matched challenge '{folder}', but no workspace at {by_folder})"
    else:
        hint = ""
    raise FileNotFoundError(
        f"No workspace for '{name_or_path}'. Run `setup` first (looked in {cand}){hint}."
    )


def workspace_challenge_dir(ws: Path) -> Path:
    meta = json.loads((ws / "meta.json").read_text())
    return (REPO_ROOT / meta["challenge_dir"]).resolve()


# --------------------------------------------------------------------------- #
# subcommands
# --------------------------------------------------------------------------- #
def _challenge_id(dirname: str) -> int:
    head = dirname.split("_", 1)[0]
    return int(head) if head.isdigit() else 1 << 30


def cmd_list(args):
    difficulties = [args.difficulty] if args.difficulty else ["easy", "medium", "hard"]
    for diff in difficulties:
        ddir = CHALLENGES_DIR / diff
        if not ddir.exists():
            continue
        rows = []
        for cdir in sorted(ddir.iterdir(), key=lambda p: _challenge_id(p.name)):
            if not (cdir / "challenge.py").exists():
                continue
            try:
                spec = importlib.util.spec_from_file_location("challenge", cdir / "challenge.py")
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                name = getattr(module.Challenge, "name", cdir.name)
            except Exception:
                name = "?"
            finally:
                sys.modules.pop("challenge", None)
            if args.search and args.search.lower() not in (cdir.name + name).lower():
                continue
            rows.append((cdir.name, name))
        if rows:
            print(diff)
            for folder, name in rows:
                print(f"  {folder:<38} {name}")
    return 0


def resolve_challenge_dir(name_or_path: str) -> Path:
    """Accept a full/relative path OR a bare folder name/number (scanning challenges/*)."""
    p = Path(name_or_path)
    if (p / "starter" / "starter.cu").exists():
        return p.resolve()
    matches = []
    for cdir in CHALLENGES_DIR.glob("*/*"):
        if not (cdir / "starter" / "starter.cu").exists():
            continue
        if cdir.name == name_or_path or str(_challenge_id(cdir.name)) == name_or_path:
            return cdir.resolve()
        if name_or_path.lower() in cdir.name.lower():
            matches.append(cdir)
    if len(matches) == 1:
        return matches[0].resolve()
    if len(matches) > 1:
        opts = ", ".join(sorted(m.name for m in matches))
        raise FileNotFoundError(f"'{name_or_path}' is ambiguous: {opts}")
    raise FileNotFoundError(
        f"No challenge '{name_or_path}' (looked under {CHALLENGES_DIR}/*/)."
    )


def cmd_setup(args):
    challenge_dir = resolve_challenge_dir(args.challenge_dir)
    starter = challenge_dir / "starter" / "starter.cu"
    if not starter.exists():
        raise FileNotFoundError(f"No starter.cu at {starter}")
    name = args.name or challenge_dir.name
    ws = WORKSPACE / name
    if ws.exists() and not args.force:
        raise FileExistsError(f"{ws} already exists (use --force to overwrite).")
    ws.mkdir(parents=True, exist_ok=True)
    shutil.copy(starter, ws / "solution.cu")
    shutil.copy(starter, ws / "reference.cu")
    meta = {"challenge_dir": os.path.relpath(challenge_dir, REPO_ROOT)}
    (ws / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Scaffolded {ws}")
    print(f"  edit  {ws / 'solution.cu'}   (your kernel)")
    print(f"  paste {ws / 'reference.cu'}  (optional known-good perf baseline)")
    print(f"  then  test {name}")


def cmd_test(args):
    require_cuda()
    ws = resolve_workspace(args.target)
    challenge_dir = workspace_challenge_dir(ws)
    challenge = load_challenge(challenge_dir, device="cuda")
    if getattr(challenge, "num_gpus", 1) > 1:
        print(f"skip: {challenge.name} needs {challenge.num_gpus} GPUs (v1 is single-GPU).")
        return 0
    signature = challenge.get_solve_signature()
    ensure_cuda_signature(challenge.name, signature)
    arch = args.arch or default_arch()

    print(f"challenge: {challenge.name}  (arch sm_{arch})")
    out_dir = ws
    sol_so = compile_cu(ws / "solution.cu", arch, out_dir)
    print("compiling solution.cu... ok")
    sol_solve = bind_solve(sol_so, signature)

    requested = args.cases.split(",")
    ok = True
    if "example" in requested:
        ok &= check_cases(challenge, signature, sol_solve,
                          [challenge.generate_example_test()], "example", "cuda")
    if "functional" in requested:
        ok &= check_cases(challenge, signature, sol_solve,
                          challenge.generate_functional_test(), "functional", "cuda")
    if "performance" in requested:
        ok &= check_cases(challenge, signature, sol_solve,
                          [challenge.generate_performance_test()], "performance", "cuda")

    # reference.cu: sanity-check + benchmark baseline (only if edited from starter)
    ref_solve = None
    starter_src = (challenge_dir / "starter" / "starter.cu").read_text()
    ref_src = ws / "reference.cu"
    if ref_src.exists() and ref_src.read_text() != starter_src:
        ref_so = compile_cu(ref_src, arch, out_dir)
        print("compiling reference.cu... ok")
        ref_solve = bind_solve(ref_so, signature)
        check_cases(challenge, signature, ref_solve,
                    [challenge.generate_example_test()], "ref:example", "cuda")

    if not args.no_bench:
        benchmark(challenge, signature, sol_solve, ref_solve, "cuda", args.runs, args.shift)

    print("\n" + ("ALL PASS" if ok else "FAILURES"))
    return 0 if ok else 1


# --------------------------------------------------------------------------- #
# profile: generate a standalone CUDA driver + input data blobs
# --------------------------------------------------------------------------- #
PTR_C_TYPE = {
    ctypes.c_float: "float",
    ctypes.c_double: "double",
    ctypes.c_uint16: "half",
    ctypes.c_int: "int",
    ctypes.c_uint: "unsigned int",
    ctypes.c_int64: "long long",
    ctypes.c_byte: "signed char",
    ctypes.c_uint8: "unsigned char",
}
SCALAR_C_TYPE = {
    ctypes.c_int: "int",
    ctypes.c_size_t: "size_t",
    ctypes.c_float: "float",
    ctypes.c_double: "double",
}


def _c_scalar_literal(ct, v) -> str:
    if ct in (ctypes.c_float, ctypes.c_double):
        return repr(float(v)) + ("f" if ct is ctypes.c_float else "")
    return str(int(v))


def cmd_profile(args):
    """Emit workspace/<name>/profile/{<in>.bin, profile.cu, build.sh}: a standalone
    CUDA driver wired to the challenge's solve signature, reading frozen perf-case
    inputs from disk. Build it and point any profiler at the resulting binary."""
    ws = resolve_workspace(args.target)
    challenge_dir = workspace_challenge_dir(ws)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    challenge = load_challenge(challenge_dir, device=device)
    signature = challenge.get_solve_signature()
    ensure_cuda_signature(challenge.name, signature)
    perf = challenge.generate_performance_test()

    outdir = ws / "profile"
    outdir.mkdir(exist_ok=True)

    decl_params, body_lines, call_names = [], [], []
    need_fp16 = False

    for name, (ct, direction) in signature.items():
        v = perf[name]
        if is_pointer(ct):
            ctype = PTR_C_TYPE.get(ct._type_)
            if ctype is None:
                raise SystemExit(f"unsupported pointer element for '{name}': {ct._type_}")
            need_fp16 = need_fp16 or ctype == "half"
            t = materialize(v, device)
            nbytes = t.numel() * t.element_size()
            decl_params.append(f"{'const ' if direction == 'in' else ''}{ctype}* {name}")
            if direction in ("in", "inout"):
                (outdir / f"{name}.bin").write_bytes(
                    t.detach().cpu().contiguous().numpy().tobytes()
                )
                body_lines.append(
                    f'    {ctype}* {name} = ({ctype}*)dev_from_file("{name}.bin", {nbytes}ULL);'
                )
            else:
                body_lines.append(f"    {ctype}* {name} = ({ctype}*)dev_empty({nbytes}ULL);")
        else:
            ctype = SCALAR_C_TYPE.get(ct)
            if ctype is None:
                raise SystemExit(f"unsupported scalar type for '{name}': {ct}")
            decl_params.append(f"{ctype} {name}")
            body_lines.append(f"    {ctype} {name} = {_c_scalar_literal(ct, v)};")
        call_names.append(name)

    includes = "#include <cuda_runtime.h>\n"
    if need_fp16:
        includes += "#include <cuda_fp16.h>\n"
    includes += "#include <cstdio>\n#include <cstdlib>\n"
    nl = "\n"

    cu = f"""\
{includes}
// Standalone profiling driver for challenge "{challenge.name}" (perf test case).
// Generated by leetgpu_local.py -- edit freely. Inputs are read from the .bin files
// in this directory; scalars are baked in below. Only solve()'s kernels run here, so
// `ncu ./profile` / `nsys profile ./profile` capture exactly the target kernel.

extern "C" void solve({", ".join(decl_params)});

static void* dev_from_file(const char* path, size_t nbytes) {{
    FILE* f = fopen(path, "rb");
    if (!f) {{ fprintf(stderr, "cannot open %s\\n", path); exit(1); }}
    void* host = malloc(nbytes);
    if (fread(host, 1, nbytes, f) != nbytes) {{ fprintf(stderr, "short read %s\\n", path); exit(1); }}
    fclose(f);
    void* dev = nullptr;
    cudaMalloc(&dev, nbytes);
    cudaMemcpy(dev, host, nbytes, cudaMemcpyHostToDevice);
    free(host);
    return dev;
}}

static void* dev_empty(size_t nbytes) {{
    void* dev = nullptr;
    cudaMalloc(&dev, nbytes);
    cudaMemset(dev, 0, nbytes);
    return dev;
}}

#ifndef REPS
#define REPS {args.reps}
#endif

int main() {{
{nl.join(body_lines)}
    for (int r = 0; r < REPS; ++r) {{
        solve({", ".join(call_names)});
    }}
    cudaDeviceSynchronize();
    return 0;
}}
"""
    (outdir / "profile.cu").write_text(cu)

    build = f"""\
#!/usr/bin/env bash
# Build the standalone profiling binary. Override arch with:  ARCH=sm_121 ./build.sh
# NVCC defaults to a driver-compatible toolkit; override with:  NVCC=/path/to/nvcc ./build.sh
set -euo pipefail
cd "$(dirname "$0")"
ARCH="${{ARCH:-native}}"
NVCC="${{NVCC:-{find_nvcc()}}}"
"$NVCC" -O3 -arch=$ARCH profile.cu ../solution.cu -o profile
echo "built ./profile"
echo "  ncu --set full ./profile     # deep counters"
echo "  nsys profile ./profile       # timeline"
"""
    build_path = outdir / "build.sh"
    build_path.write_text(build)
    build_path.chmod(0o755)

    dumped = [f"{n}.bin" for n in call_names if (outdir / f"{n}.bin").exists()]
    print(f"generated profiling bundle in {outdir}/")
    print(f"  profile.cu   driver for '{challenge.name}' (perf case)")
    print(f"  {', '.join(dumped)}   frozen input data")
    print("  build.sh     nvcc profile.cu ../solution.cu -o profile")
    print(f"\nnext:  cd {outdir} && ./build.sh && ncu --set full ./profile")
    return 0


def cmd_check(args):
    """Smoke-test the local setup end to end: torch+CUDA, challenges/core import,
    an nvcc whose CUDA major the driver supports, and an actual compile -> launch ->
    readback of a trivial kernel. Run this right after cloning + creating the venv."""
    ok = True

    def line(tag, msg):
        print(f"  [{tag:<6}] {msg}")

    print(f"setup check (python {sys.version.split()[0]}, {os.uname().machine})")

    line("torch", torch.__version__)
    if not torch.cuda.is_available():
        line("cuda", "NOT available -- use the CUDA venv (.venv-cuda-<arch>/bin/python)")
        return 1
    cap = torch.cuda.get_device_capability()
    line("cuda", f"available -- {torch.cuda.get_device_name()} (sm_{cap[0]}{cap[1]})")
    line("core", f"challenges/core importable ({CHALLENGES_DIR})")

    nvcc = find_nvcc()
    nv = nvcc_version(nvcc)
    drv = driver_cuda_major()
    if nv is None:
        line("nvcc", f"NOT found (looked for '{nvcc}') -- install a CUDA toolkit")
        return 1
    compat = drv is None or nv[0] <= drv
    drv_str = f"driver supports CUDA {drv}.x" if drv is not None else "driver version unknown"
    warn = "" if compat else "  <-- TOO NEW for the driver, kernels may not launch"
    line("nvcc", f"{nvcc} (CUDA {nv[0]}.{nv[1]}); {drv_str}{warn}")
    ok = ok and compat

    src = (
        "__global__ void add_one(float* x, int n) {\n"
        "    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
        "    if (i < n) x[i] += 1.0f;\n"
        "}\n"
        'extern "C" void solve(float* x, int n) {\n'
        "    add_one<<<(n + 255) / 256, 256>>>(x, n);\n"
        "}\n"
    )
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "smoke.cu").write_text(src)
        try:
            so = compile_cu(td / "smoke.cu", default_arch(), td)
        except RuntimeError as e:
            line("kernel", f"nvcc compile FAILED\n{e}")
            return 1
        signature = {
            "x": (ctypes.POINTER(ctypes.c_float), "inout"),
            "n": (ctypes.c_int, "in"),
        }
        solve = bind_solve(so, signature)
        n = 1024
        t = torch.zeros(n, device="cuda")
        solve(ctypes.c_void_p(t.data_ptr()), n)
        torch.cuda.synchronize()
        launched = bool(torch.allclose(t, torch.ones_like(t)))
        if launched:
            line("kernel", "compiled, launched, and read back correctly")
        else:
            line("kernel", "compiled but kernel did NOT run (nvcc/driver mismatch?)")
        ok = ok and launched

    print("\n" + ("READY" if ok else "SETUP INCOMPLETE"))
    return 0 if ok else 1


def require_cuda():
    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA torch is not available in this environment. Use the CUDA venv, e.g.\n"
            "  .venv-cuda/bin/python scripts/leetgpu_local.py ..."
        )


# --------------------------------------------------------------------------- #
def main() -> int:
    p = argparse.ArgumentParser(description="Local LeetGPU challenge harness (CUDA).")
    sub = p.add_subparsers(dest="cmd", required=True)

    ls = sub.add_parser("list", help="List available challenges.")
    ls.add_argument("difficulty", nargs="?", choices=["easy", "medium", "hard"], default=None)
    ls.add_argument("--search", default=None, help="Filter by substring (folder or name).")
    ls.set_defaults(func=cmd_list)

    s = sub.add_parser("setup", help="Scaffold a workspace from a challenge dir.")
    s.add_argument("challenge_dir")
    s.add_argument("--name", default=None, help="Workspace name (default: challenge folder name).")
    s.add_argument("--force", action="store_true")
    s.set_defaults(func=cmd_setup)

    t = sub.add_parser("test", help="Compile solution.cu and check + benchmark it.")
    t.add_argument("target", help="Workspace name or path.")
    t.add_argument("--cases", default="example,functional,performance")
    t.add_argument("--no-bench", action="store_true")
    t.add_argument("--runs", type=int, default=5)
    t.add_argument("--shift", type=float, default=1.0, help="sgm shift in ms.")
    t.add_argument("--arch", default=None, help="Override SM arch (e.g. 86).")
    t.set_defaults(func=cmd_test)

    pr = sub.add_parser("profile", help="Generate a standalone CUDA profiling driver + data.")
    pr.add_argument("target", help="Workspace name or path.")
    pr.add_argument("--reps", type=int, default=1, help="Kernel launches per run (baked as REPS).")
    pr.set_defaults(func=cmd_profile)

    ck = sub.add_parser("check", help="Smoke-test the local setup (torch+CUDA, nvcc, kernel launch).")
    ck.set_defaults(func=cmd_check)

    args = p.parse_args()
    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())
