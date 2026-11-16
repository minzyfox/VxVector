# Vx issue drafts (November 14, 2026)

These drafts come from Step 3, L2 k-means on the Apple GPU, and from a later timing run that crashed.
Each draft is ready to paste into a GitHub issue in vx-lang/Vx.

All reproductions were run with `vxc 0.0.4 (423a10db)`, which is Vx `main` and the VxVector pin, on an Apple M2 Pro running macOS 27.2.1.
Each is a complete file. The compile command is `vxc --host default --machine fleet/m4-uma.vx <file>.vx`, with `--action emit-obj -o <file>.o` where noted.
The Mac had its display on, with an editor and a terminal open.

Before you file one, search vx-lang/Vx again. Searches on November 14 found nothing for "Impacting Interactivity", Metal aborts, CPU fallback, `is_finite`, or `libvx_npu_dispatch`.
Related issues are named in each draft.

| # | Label | Title | Ready |
| --- | --- | --- | --- |
| 1 | bug | Metal: a launch that macOS stops for interactivity aborts the program | yes |
| 2 | enhancement | Metal: split a long `across` launch into short ones | yes, after 1 |
| 3 | enhancement | Metal: let C code ask whether kernels run on the GPU, and require it | yes |
| 4 | bug | `f32::is_finite` cannot be called in an `across` loop | yes |
| 5 | bug | A function that is not `pub` is a global symbol of the object | yes |
| 6 | enhancement | A stable path to `libvx_npu_dispatch.dylib` for C programs | yes |
| 7 | enhancement | fleet: machine files for M1, M2, and M3 Macs | low value until #245 |
| 8 | bug | Apple GPU: a call through a function pointer gets no wait before C reads GPU memory | yes, follow-up to #3153 |
| 9 | bug | `emit-obj` requires every `extern` function to exist when vxc runs | yes |
| 10 | enhancement | On unified memory, let the CPU write GPU memory in place | as a comment on #2690 |

Drafts 1 and 2 matter most for VxVector. Draft 1 turns a crash into an error that VxVector can handle. Draft 2 removes the cause.
Drafts 8 to 10 came from Step 5, spherical centers (record), and were run on November 14 at `423a10db`. Draft 8 is a silent wrong result. VxVector avoids it only by calling `vx_metal_wait()` itself.

---

## 1. bug: Metal: a launch that macOS stops for interactivity aborts the program

On a Mac, the GPU also draws the screen. macOS can stop a command buffer that keeps the GPU too long. It reports the error `kIOGPUCommandBufferCallbackErrorImpactingInteractivity`.
The Vx runtime then prints `FATAL` and calls `abort()` (`finish` in `runtime/metal_dispatch.mm`). The program cannot handle the error or try again.

The error is transient. The same launch usually succeeds when it runs again. The reproduction below failed in 5 of 7 runs, at a different round each time.

```vx
// Each launch keeps the Apple GPU busy for about 75 ms on an M2 Pro: 32768
// lanes, and each lane adds 32768 products.
import core::num;

pub fn spin(out : &mut Tensor<f32, [?], Memory::HBM>, x : &Tensor<f32, [?, ?], Memory::HBM>, c : &Tensor<f32, [?, ?], Memory::HBM>, n : i32, k : i32, d : i32) -> void {
  spawn on(Topology::Device) {
    for chunk in 0..(n + 255) / 256 across blocks {
      for lane in 0..256 across lanes {
        let row = chunk * 256 + lane;
        if row < n {
          let mut best : f32 = 0.0;
          for j in 0..k {
            let mut s : f32 = 0.0;
            for e in 0..d {
              let t = x[row][e] - c[j][e];
              s = s + t * t;
            }
            if j == 0 || s < best {
              best = s;
            }
          }
          out[row] = best;
        }
      }
    }
  }
  return;
}

fn main() -> i32 {
  let n = 32768;
  let k = 256;
  let d = 128;
  let mut x = transfer(Tensor<f32, [32768, 128]>::fill(0.5), Memory::HBM);
  let c = transfer(Tensor<f32, [256, 128]>::fill(0.25), Memory::HBM);
  let mut out = transfer(Tensor<f32, [32768]>::fill(0.0), Memory::HBM);
  for round in 0..40 {
    spin(&mut out, &x, &c, n, k, d);
    let host = transfer(&out, Memory::CPU_DRAM);
    if round % 10 == 0 {
      println!("round {} value {}", round, host[0]);
    }
  }
  println!("done");
  return 0;
}
```

With `VX_DISPATCH_VERBOSE=1`, a failing run ends with:

```
[Vx Metal] waited for 1 kernel, 74.9 ms on the GPU
[Vx Metal] FATAL: the GPU did not finish vx_npu_kernel_0: Impacting Interactivity (0000000e:kIOGPUCommandBufferCallbackErrorImpactingInteractivity)
Program was killed by signal 6 (SIGABRT, often a failed assert)
```

A library hits the same abort when C calls it. In VxVector, a k-means library built with `--action emit-obj` aborted its host process at 32,768 rows, 256 centers, and 128 dimensions, about 80 ms per launch.

**Proposal.**

1. **Retry a launch that can run again.** A launch can run again when no tensor that it writes is also one that it reads, so its inputs are unchanged. The payload already marks read-only tensor parameters (`params[i].back() == 'r'`). It would also need to mark parameters that a kernel writes without reading. On `ImpactingInteractivity`, the runtime could rerun such a launch a few times before it gives up.
2. **Return the error instead of aborting.** When a retry is not possible or does not succeed, the function that made the launch should fail in a way the caller can see. Two options: a `Result` for Vx callers, or a `vx_metal_last_error()` for C callers of an `emit-obj` object. Either way, a server process that uses a Vx library survives a GPU fault.

Related: #3113 (closed) changed when a batch commits. #2781 (open) puts a function's launches into one command buffer. Draft 2 shows why #2781 does not make this worse.

---

## 2. enhancement: Metal: split a long `across` launch into short ones

macOS judges each launch, not each command buffer. The program of draft 1, changed to make eight launches of 4,096 rows instead of one launch of 32,768, did not fail in 5 of 5 runs.
Its eight launches still went into one batch, and the batch still took the same time on the GPU, 74 ms.

The change to the program of draft 1:

```diff
-pub fn spin(..., n : i32, k : i32, d : i32) -> void {
+pub fn spin(..., base : i32, count : i32, k : i32, d : i32) -> void {
   spawn on(Topology::Device) {
-    for chunk in 0..(n + 255) / 256 across blocks {
+    for chunk in 0..(count + 255) / 256 across blocks {
       for lane in 0..256 across lanes {
-        let row = chunk * 256 + lane;
-        if row < n {
+        let row = base + chunk * 256 + lane;
+        if row < base + count {
 ...
-    spin(&mut out, &x, &c, n, k, d);
+    for part in 0..8 {
+      spin(&mut out, &x, &c, part * 4096, 4096, k, d);
+    }
```

| Version | Launches per round | GPU time per wait | Failed runs |
| --- | --- | --- | --- |
| One launch | 1 | about 75 ms | 5 of 7 |
| Eight launches | 8, in one batch | about 74 ms | 0 of 5 |

Every program that runs long kernels on a Mac must split them by hand today, with its own offsets.

**Proposal.** A loop over blocks in an `across` kernel already steps over the grid (`grid_loops=1`), so the runtime may launch fewer blocks than the loop asks for.
With a first-block offset as a kernel argument, the runtime could launch the blocks in several pieces. It could size each piece from the GPU time of the last launch of that kernel, which it measures already, to stay under a budget such as 10 ms.
Each piece is a separate dispatch, so macOS can let the display run between them. The kernel's result does not change, because the blocks are independent.

---

## 3. enhancement: Metal: let C code ask whether kernels run on the GPU, and require it

When Metal is off (`VX_METAL=0`, or `vx_metal_set_enabled(0)`), `vx_metal_dispatch` returns 0 and the runtime runs the CPU copy of the kernel. The caller is not told.
A library that promises GPU execution cannot tell a GPU run from a CPU run.

VxVector works around it in two ways:
- It allocates with `vx_metal_alloc`, which returns NULL when Metal is off or no device exists, and refuses to run then.
- Its tests parse `VX_DISPATCH_VERBOSE` output for `[Vx Metal] ... grid` lines, and for `is copied` and `Falling back`.

**Proposal.**
- `int vx_metal_available(void)`: 1 when a Metal device exists and this thread sends kernels to it.
- A way to require the GPU, such as `vx_metal_set_required(1)` or `VX_METAL=require`. Then a kernel that cannot run on the GPU fails as in draft 1, instead of running its CPU copy.
- Counters that C can read without stderr: launches sent to Metal, CPU fallbacks, copied tensors, and GPU time. For example, `vx_metal_stats(struct vx_metal_stats *)`.

---

## 4. bug: `f32::is_finite` cannot be called in an `across` loop

```vx
import core::num;

pub fn mark(out : &mut Tensor<i32, [?], Memory::HBM>, x : &Tensor<f32, [?], Memory::HBM>, n : i32) -> void {
  spawn on(Topology::Device) {
    for b in 0..(n + 255) / 256 across blocks {
      for t in 0..256 across lanes {
        let i = b * 256 + t;
        if i < n {
          if x[i].is_finite() {
            out[i] = 1;
          } else {
            out[i] = 0;
          }
        }
      }
    }
  }
  return;
}
```

`vxc --host default --machine fleet/m4-uma.vx is_finite.vx --action emit-obj -o is_finite.o`:

```
Error[E6032] at is_finite.vx:9:19: the call to `f32$is_finite` may write through an argument, and the checker does not read its body. If `f32$is_finite` writes no memory, give it `ensures writes_only()`. ...
```

`is_finite` writes no memory, but the user cannot add the contract to a standard-library function. VxVector writes its own `finite_f32`, which tests `value - value == 0.0`.

**Proposal.** Give `ensures writes_only()` to the pure methods of the number types in `core::num`, such as `is_finite`, `is_nan`, `is_infinite`, `abs`, `sqrt`, `min`, and `max`. A test could call each one in an `across` loop.

---

## 5. bug: A function that is not `pub` is a global symbol of the object

```vx
fn finite(value : f32) on Topology::Device -> bool
ensures writes_only()
{
  return value - value == 0.0;
}

pub fn mark(out : &mut Tensor<i32, [?], Memory::HBM>, x : &Tensor<f32, [?], Memory::HBM>, n : i32) -> void {
  spawn on(Topology::Device) {
    for b in 0..(n + 255) / 256 across blocks {
      for t in 0..256 across lanes {
        let i = b * 256 + t;
        if i < n {
          if finite(x[i]) {
            out[i] = 1;
          } else {
            out[i] = 0;
          }
        }
      }
    }
  }
  return;
}
```

After `--action emit-obj -o helper.o`, `nm helper.o` shows:

```
0000000000000440 T _finite
00000000000005fc T __mlir_finite
```

`finite` is not `pub`, but the object exports it. It has the same name as the C library's `finite()`, so a program that links this object can call the wrong one.
VxVector's objects also export `_i64::From$i32$from`, the runtime kernel names `_vx_npu_kernel_N`, and their `__mlir_` and `__vx_ciface_` wrappers.
Every object numbers its kernels from 0, so two objects with Apple GPU kernels cannot be linked into one library. Linking `helper.o` with another such object:

```
$ clang -shared gpu_kmeans.o helper.o libvx_npu_dispatch.dylib -o both.dylib
duplicate symbol '__mlir__vx_ciface_vx_npu_kernel_0' in:
    helper.o
    gpu_kmeans.o
duplicate symbol '__vx_ciface_vx_npu_kernel_0' in:
    helper.o
...
```

**Proposal.** Give functions that are not `pub` internal linkage, and the kernels and wrappers too, or names unique to the module. Export only `pub` functions.

---

## 6. enhancement: A stable path to `libvx_npu_dispatch.dylib` for C programs

A C program that links an object with Apple GPU kernels needs `libvx_npu_dispatch.dylib`, as `tests/integration_test/apple_gpu_kernels_called_from_c.rs` shows.
The library exists only at `target/<profile>/build/vxc-<hash>/out/libvx_npu_dispatch.dylib`. After a few rebuilds there are several, with different hashes.
The test finds it through the `NPU_SHARED_LIB_PATH` that `build.rs` sets for Cargo, which a build outside Cargo cannot read.

VxVector finds the right one by searching the `vxc` binary for the path that `vxc` embeds:

```sh
grep -a -o '/[^"[:cntrl:]]*libvx_npu_dispatch\.dylib' target/debug/vxc | sort -u
```

**Proposal.** Copy the library to `target/<profile>/libvx_npu_dispatch.dylib` beside `libvx_std_core.dylib`, or add `vxc --print-runtime-path`. The install could ship it in `lib/` too.

---

## 7. enhancement: fleet: machine files for M1, M2, and M3 Macs

`fleet/m4-uma.vx` is the only Apple machine file. On an M2 Pro with 16 GiB, it declares 32 GiB of `HBM` and 10 GPU cores, where the machine has 16 GiB and 16 cores.
Admission and capacity checks therefore accept a program that would need more memory than the machine has.

VxVector wrote a local `machines/m2-pro-16g.vx` from `sysctl` and `system_profiler` values. The kernels compiled with it gave the same Metal source and the same object, byte for byte, as with `m4-uma.vx`.
So the machine file affects the checks, not the code.

**Proposal.** Either:
- add files for the M1, M2, and M3 families; or
- document in `fleet/README.md` how to write one for a local Mac, naming the `sysctl` and `system_profiler` fields to read.

Memory size differs between Macs of one chip, so the second option may serve better. A `vxc --machine host` that reads these values when it runs would serve best.

This draft matters little for VxVector today. Its kernels take shapes `[?, ?]`, which Vx does not check against any machine file (W1029). A comment on #245, bounded dynamic shapes, may help more. VxVector's sizes come from SQL when it runs, and a memory budget bounds them.

---

## 8. bug: Apple GPU: a call through a function pointer gets no wait before C reads GPU memory

Follow-up to #3153. #3153 waits before a call to a C function that gets a pointer into GPU memory. It marks the `extern` declaration (`vx.gets_device_pointer`), so a call through a function pointer is not marked.
A function-pointer type can name GPU memory, `fn(*mut f32, *const f32 in Memory::HBM, i64) -> *mut f32`, and C can pass `memcpy` as one. The call then gets no `VX_CTRL_WAIT`, and C reads the memory before the kernel has written it.

`kernels.vx`:

```vx
// A kernel writes dst, then dst goes to `copy` through a function pointer.
pub fn fill_then_copy(dst : &mut Tensor<f32, [?], Memory::HBM>, n : i32, rounds : i32,
    host : *mut f32,
    copy : fn(*mut f32, *const f32 in Memory::HBM, i64) -> *mut f32) -> void {
  spawn on(Topology::Device) {
    for g in 0..(n + 255) / 256 across blocks {
      for l in 0..256 across lanes(256) {
        let i = g * 256 + l;
        if i < n {
          let mut v : f32 = 0.0;
          for r in 0..rounds {
            v = v + 1.0;
          }
          dst[i] = v;
        }
      }
    }
  }
  copy(host, dst.as_ptr(), (n as i64) * 4);
  return;
}
```

`host.c`:

```c
#include <stdint.h>
#include <stdio.h>
#include <string.h>
void fill_then_copy(float *, float *, int64_t, int64_t, int64_t, int32_t, int32_t, float *,
                    void *(*)(void *, const void *, size_t));
void *vx_metal_alloc(size_t);
int main(void) {
    enum { N = 1 << 20 };
    float *dst = vx_metal_alloc(N * sizeof(float));
    static float host[N];
    for (int trial = 0; trial < 5; trial++) {
        memset(dst, 0, N * sizeof(float));
        memset(host, 0xff, sizeof host);
        fill_then_copy(dst, dst, 0, N, 1, N, 20000, host, memcpy);
        int stale = 0;
        for (int i = 0; i < N; i++) stale += host[i] != 20000.0f;
        printf("trial %d: %d of %d copied values are stale (first %g, last %g)\n", trial, stale, N, host[0], host[N - 1]);
    }
    return 0;
}
```

```
$ vxc --host default --machine fleet/m4-uma.vx kernels.vx --action emit-obj -o kernels.o
$ cc host.c kernels.o <target/debug/build/vxc-*/out/libvx_npu_dispatch.dylib> -o host && ./host
trial 0: 1048576 of 1048576 copied values are stale (first 0, last 0)
...
trial 4: 1048576 of 1048576 copied values are stale (first 0, last 0)
```

Each value should be 20000. A direct call works. Declared as `extern "C" { fn memcpy(dst : *mut f32, src : *const f32 in Memory::HBM, bytes : i64) -> *mut f32; }`, the same program copies 0 stale values in every trial. The kernel itself writes all 1,048,576 values: C reads `dst` after `fill_then_copy` returns, and every value is 20000.

The LLVM dialect shows the difference. Direct call:

```
%60 = llvm.call @vx_plugin_control(%18, %33) : (i32, !llvm.ptr) -> i32
%61 = llvm.call @memcpy(%arg7, %57, %59) : (!llvm.ptr, !llvm.ptr, i64) -> !llvm.ptr
```

Through the function pointer, the only wait comes after the call:

```
%58 = llvm.call %arg8(%arg7, %55, %57) : !llvm.ptr, (!llvm.ptr, !llvm.ptr, i64) -> !llvm.ptr
%59 = llvm.call @vx_plugin_control(%18, %37) : (i32, !llvm.ptr) -> i32
```

**Proposal.** Wait before an indirect call whose function type has a pointer into memory other than host memory, as #3153 does for a marked declaration. Or reject such a function type where a wait cannot be inserted.

VxVector met this with a function pointer for exactly this purpose: draft 9 keeps it from declaring the C function as `extern`. It calls `vx_metal_wait()` itself before the indirect call.

---

## 9. bug: `emit-obj` requires every `extern` function to exist when vxc runs

Related: #1652 (`run-jit` cannot link a C library), #1827. This one is about `emit-obj`, which links nothing.

`vxc --action emit-obj` builds the object in a JIT session that resolves each symbol. So an `extern` function must already be loaded in `vxc` itself, as libc's `memcpy` is. A library cannot call a C function that the program linking it will define, which is the usual case for a callback or a hook.

`hook.vx`:

```vx
// A library calls a C function that the program it is linked into defines.
extern "C" {
  fn vxvector_hook(value : i32) -> i32;
}

pub fn twice_hook(value : i32) -> i32 {
  return unsafe { vxvector_hook(value) } * 2;
}
```

```
$ vxc hook.vx --action emit-obj -o hook.o
JIT session error: Symbols not found: [ _vxvector_hook ]
Could not compile twice_hook:
  Failed to materialize symbols: { (main, { _twice_hook, __mlir_twice_hook }) }
$ echo $?
134
```

It is the same with `--host default --machine fleet/m4-uma.vx`. No object is written, and vxc aborts with status 134.

**Proposal.** In `emit-obj`, leave an `extern` symbol undefined in the object, for the linker to resolve, as a C compiler does. Resolve symbols only for `run-jit`.

Today the workaround is to take the C function as a function-pointer argument. On the Apple GPU, that runs into draft 8.

---

## 10. enhancement: On unified memory, let the CPU write GPU memory in place

Post as a comment on #2690. #2690 lets the CPU read a GPU tensor without a copy: `transfer(t, Memory::CPU_DRAM)` of a borrowed tensor waits for the GPU when the machine file marks the edge `: shared` and the result is only read. The CPU cannot write the tensor in place.

VxVector needs a write. Its spherical k-means normalizes the centers between two kernels in float64 (`normalize` in `src/kmeans.vx`), which the Apple GPU cannot do. It costs k × dims operations, so it suits the CPU, and the memory is shared.
Today a Vx function cannot write GPU memory. So VxVector writes this step in C (`vxvector_gpu_normalize_rows` in `src/gpu_l2.c`) and passes it as a function pointer (drafts 8 and 9).

**Proposal.** On an edge marked `: shared`, let a function borrow a GPU tensor mutably on the CPU, after waiting for the launches that use it. Writing back with `transfer` would copy, k × dims values here. That is small, but not what a shared edge needs.

VxVector has not tried the read-only path of #2690 for this step. It needs the write.

