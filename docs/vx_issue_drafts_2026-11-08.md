# Vx issue drafts (November 8, 2026)

These drafts come from B2 in the [usage review](vx_usage_review_2026-11-08.md):
making `nearest` and `train` in `src/kmeans.vx` `unsafe fn` with pointer contracts.
Each draft is ready to paste into a GitHub issue in vx-lang/Vx.

All reproductions were run with `vxc 0.0.3 (31296d42)`, which is Vx `main` and the
VxVector pin. Each is a complete file. The commands are
`vxc <file>.vx --action emit-mlir` for errors and `vxc <file>.vx --audit` for trusted
clauses.

Before you file one, search vx-lang/Vx again. Each draft lists the related issues
found on November 8. Most drafts are about the raw-pointer facts, so each one
says "Part of #2829", the tracking issue for the limits of v1.

| # | Label | Title | Ready |
| --- | --- | --- | --- |
| 1 | bug | Raw-pointer facts: a `let` with no written type in a loop body ends the facts of the loop | yes, or as a comment on #2836 |
| 2 | bug | Raw-pointer facts: a pointer clause in a nested loop invariant counts as a call that can free memory | yes, follow-up to #2837 |
| 3 | bug | Raw-pointer facts: an offset that contains a call has no facts, and W1007 asks for that call | yes |
| 4 | enhancement | Raw-pointer facts: `!overlaps` for two buffers of different lengths | yes |
| 5 | enhancement | Prover: a fact that names a variable ends with the scope of the variable | yes |
| 6 | enhancement | Prover: model integer division | yes |
| 7 | bug | Raw-pointer facts: `!frees` is not inferred for a parameter that only reaches `!frees` callees | no: reduce first |
| 8 | bug | Prover: a null check in an `\|\|` condition with many terms loses the facts of the other terms | no: reduce first |
| 9 | enhancement | Prover: facts about the number fields of a struct | yes |

Already filed: [#2852](https://github.com/vx-lang/Vx/pull/2852) (open PR) keeps
the bound of a call result after a later call. VxVector works around it in
`centroid_distance`. Nothing new to file for it.

---

## 1. bug: Raw-pointer facts: a `let` with no written type in a loop body ends the facts of the loop

Part of #2829. Related: #2836, limit 9. The loop rule reads the body before the
checker checks it, so it does not know the types. #2836 covers a method call. This
draft covers a `let` whose value is a path call, such as `i64::from(..)`, which
the scan counts as a possible drop. #2836 says there is no plan to change its
limit, so this draft can also be posted there as a comment, asking whether this
case can be fixed on its own. A path call to a known impl, or a call that returns
a number, cannot drop.

Before the checker checks a loop body, the loop rule scans the body for events
(`loop_body_event` in `src/hir/check/pointer_facts.rs`). For a `let`, the scan
asks whether the type of the value drops. When the scan cannot find that type,
it uses `Type::Unknown`, and it counts the end of the scope of the `let` as an
event that can free memory. That event ends every fact about other memory that
the loop carries.

The scan cannot find the type of a call through a trait, such as `i64::from(..)`.
So `let r = i64::from(x);` in a loop body ends the facts about every pointer
parameter, although an `i64` has no drop.

Reproduction:

```vx
import core::convert;
import core::ptr;

fn row_of(i: i32) -> i32 { return i; }

unsafe fn read(p: *const f32, n: i32) -> f32
requires valid(p, n)
requires initialized(p, n)
{
  return unsafe { p[0] };
}

// The fact about p comes from before the loop. The `let r` in the body has no
// written type, and the loop scan counts the end of its scope as a drop.
unsafe fn untyped(p: *const f32, n: i32) -> f32
requires n > 0
requires valid(p, n)
requires initialized(p, n)
{
  let mut total: f32 = 0.0;
  for i in 0..n {
    let r = i64::from(row_of(i));
    total = total + unsafe { read(p, n) };
  }
  return total;
}

// The same body with `let r: i64` keeps the fact.
unsafe fn typed(p: *const f32, n: i32) -> f32
requires n > 0
requires valid(p, n)
requires initialized(p, n)
{
  let mut total: f32 = 0.0;
  for i in 0..n {
    let r: i64 = i64::from(row_of(i));
    total = total + unsafe { read(p, n) };
  }
  return total;
}

fn main() -> i32 { return 0; }
```

Current: `--audit` lists two trusted clauses, both in `untyped`:

```
untyped_let_is_a_drop.vx:23:30: a call in `unsafe` code trusts `initialized(p, n)` of `read`, which the compiler cannot prove
untyped_let_is_a_drop.vx:23:30: a call in `unsafe` code trusts `valid(p, n)` of `read`, which the compiler cannot prove
```

When the fact is needed for a loop invariant or an `ensures`, the note names the
`let`: "a fact that proves it ended at the end of the scope of `row`, which drops
its value, in a pass of the loop".

Expected: `untyped` and `typed` get the same facts. The scan can resolve a call
through a trait whose impl is known, as `i64::from` from `core::convert`. Or the
scan can leave the decision to the body check for these values. An unknown type
can still count as an event when the type can really drop.

VxVector workaround: each `let` in a loop body of `src/kmeans.vx` has a written
type.

---

## 2. bug: Raw-pointer facts: a pointer clause in a nested loop invariant counts as a call that can free memory

Part of #2829. Follow-up to #2837 (closed), which added pointer clauses in loop
invariants. The examples in #2837 have a single loop. This bug appears when one
loop is inside another.

The loop rule scans the expressions of the body for calls that can free memory.
The scan also visits the invariant clauses of a nested loop. A clause such as
`invariant initialized(q, i)` reads as a call to a function named `initialized`
with the argument `q`, a `*mut`. No function states `!frees` for it, so the scan
counts it as an event, and the outer loop drops the facts about other memory.

An invariant runs no code. The scan already skips the arguments of `assume` for
the same reason.

Reproduction:

```vx
import core::ptr;

unsafe fn read(p: *const f32, n: i32) -> f32
requires valid(p, n)
requires initialized(p, n)
{
  return unsafe { p[0] };
}

// The inner loop's invariant `initialized(q, i)` counts as a call to
// `initialized` that can free memory through q, so the outer loop loses the
// facts about p.
unsafe fn outer(p: *const f32, q: *mut i32, n: i32, rounds: i32) -> f32
requires n > 0
requires valid(p, n)
requires initialized(p, n)
requires valid(q, n)
{
  let mut total: f32 = 0.0;
  for round in 0..rounds {
    let mut i = 0;
    while i < n invariant i >= 0 && i <= n invariant initialized(q, i) {
      unsafe { q[i] = 0; }
      i = i + 1;
    }
    total = total + unsafe { read(p, n) };
  }
  return total;
}

fn main() -> i32 { return 0; }
```

Current:

```
nested_invariant_is_a_call.vx:26:30: a call in `unsafe` code trusts `initialized(p, n)` of `read`, which the compiler cannot prove
nested_invariant_is_a_call.vx:26:30: a call in `unsafe` code trusts `valid(p, n)` of `read`, which the compiler cannot prove
```

In VxVector the note was: "a fact that proves it ended at the call to
'initialized', which can free memory through `counts`, in a pass of the loop".

Expected: no trusted clause. The scan skips the pointer clauses `valid`,
`initialized`, `overlaps` and `frees` in a loop invariant, as it skips `assume`.

VxVector workaround: the loops that zero `counts` and `sums` moved into
`zero_counts` and `zero_sums`, which state `ensures initialized(..)`. Their loop
invariants are no longer inside the Lloyd iteration loop.

---

## 3. bug: Raw-pointer facts: an offset that contains a call has no facts, and W1007 asks for that call

Part of #2829.

`raw_pointer_facts.md` section 3 says that `offset(p, k)` has the base of `p`
and the offset of `p` plus `k`. When `k` contains a call, as in
`offset(p, i64::from(c) * i64::from(m))`, the pointer gets no usable offset,
although the prover knows `i64::from(v) == v` from the `ensures` in
`core::convert`. The same call works in a length.

The cast `(c as i64) * (m as i64)` works, but each cast gets W1007, which asks
for `i64::from(..)`. So the lint asks for the form that the checker cannot use.

Reproduction:

```vx
import core::convert;
import core::ptr;

unsafe fn fill(d: *mut f32, n: i64) -> void
requires valid(d, n)
ensures initialized(d, n)
{
  for i in 0i64..n invariant initialized(d, i) {
    unsafe { d[i] = 0.0; }
  }
}

// Fails: the offset contains calls to i64::from.
unsafe fn with_from(p: *mut f32, k: i32, m: i32) -> void
requires k > 0 && m > 0
requires valid(p, (k as i64) * (m as i64))
ensures initialized(p, (k as i64) * (m as i64))
{
  let mut c = 0;
  while c < k invariant c >= 0 && c <= k invariant initialized(p, (c as i64) * (m as i64)) {
    unsafe { fill(offset_mut(p, i64::from(c) * i64::from(m)), i64::from(m)); }
    c = c + 1;
  }
}

// Passes, with warning W1007 on each cast.
unsafe fn with_as(p: *mut f32, k: i32, m: i32) -> void
requires k > 0 && m > 0
requires valid(p, (k as i64) * (m as i64))
ensures initialized(p, (k as i64) * (m as i64))
{
  let mut c = 0;
  while c < k invariant c >= 0 && c <= k invariant initialized(p, (c as i64) * (m as i64)) {
    unsafe { fill(offset_mut(p, (c as i64) * (m as i64)), i64::from(m)); }
    c = c + 1;
  }
}

fn main() -> i32 { return 0; }
```

Current:

```
Warning[W1007] at offset_with_call.vx:34:36: this cast only widens, so it cannot lose a value; `i64::from(..)` from `core::convert` says so
Warning[W1007] at offset_with_call.vx:34:49: this cast only widens, so it cannot lose a value; `i64::from(..)` from `core::convert` says so
Error[E8009] at offset_with_call.vx:20:52: this loop invariant `initialized(p, c as i64 * m as i64)` may not hold after a pass through the loop body: the compiler cannot prove it from what it knows at the end of the body
```

Also checked: `let start = i64::from(c) * i64::from(m);` followed by
`offset_mut(p, start)` fails the same way inside the loop (see draft 5). A helper
with `ensures return == (row as i64) * (dims as i64)` in the offset fails too.

Expected: `with_from` passes. The offset of a pointer uses the same model of
calls as a length: the result of a call with a value `ensures` is a fresh name
with the facts of that `ensures`. Until then, W1007 does not fire on a cast inside
the offset of a pointer function.

VxVector workaround: each function widens once (`let width = i64::from(dims);`),
counts with `i64` locals, and writes offsets as products of locals, such as
`seeded * width`.

---

## 4. enhancement: Raw-pointer facts: `!overlaps` for two buffers of different lengths

Part of #2829. Related: #1166 (disjoint ranges of one tensor).

`!overlaps(p, q, n)` says that the first `n` elements at `p` and at `q` share no
byte. It has one length for both. A C caller often passes separate buffers of
different sizes: a data matrix of `rows * dims` elements and a scratch buffer of
`k * dims`. No clause can say that.

- `!overlaps(small, data, k * dims)` covers only the first `k` rows of `data`. A
  copy from a later row cannot be proved.
- `!overlaps(small, data, rows * dims)` is false when `rows > k`. It claims that
  memory past the end of `small` is apart from `data`. Two allocations that sit
  next to each other break it, such as two `palloc` chunks in PostgreSQL.

Reproduction:

```vx
import core::ptr;

unsafe fn copy(dst: *mut f32, src: *const f32, n: i64) -> void
requires valid(dst, n)
requires valid(src, n)
requires initialized(src, n)
requires !overlaps(dst, src, n)
{
  for i in 0i64..n {
    unsafe { dst[i] = src[i]; }
  }
}

// data holds two rows of width w, and row holds one. They are separate
// buffers, but `!overlaps(row, data, w)` covers only the first row of data,
// and `!overlaps(row, data, 2 * w)` would also claim w elements past the end
// of row.
unsafe fn copy_second_row(row: *mut f32, data: *const f32, w: i64) -> void
requires w > 0i64
requires valid(row, w)
requires valid(data, 2i64 * w)
requires initialized(data, 2i64 * w)
requires !overlaps(row, data, w)
{
  unsafe { copy(row, offset(data, w), w); }
}

fn main() -> i32 { return 0; }
```

Current:

```
overlaps_two_lengths.vx:25:12: a call in `unsafe` code trusts `!overlaps(dst, src, n)` of `copy`, which the compiler cannot prove
```

Requested: a clause with a length for each pointer, such as
`!overlaps(p, n, q, m)`: the `n` elements at `p` and the `m` elements at `q` share
no byte. Its fact is `Apart(bp, 0, n, bq, 0, m)`, and the goal of section 7 checks
each range against its own length. The 3-argument form stays as the case
`n == m`.

VxVector use case: `run_restart` copies sampled rows of `data` into `current`.
These two copies are the only trusted clauses left outside the C boundary.

---

## 5. enhancement: Prover: a fact that names a variable ends with the scope of the variable

Part of #2829. Related: #2796 ("a fact ends with the scope of its variable"),
#1358.

When the scope of a variable ends, the prover drops each fact that names it.
That is sound, but it drops what the fact says about the variables that stay.
After `batch_start = batch_end;`, the prover knows `batch_start == batch_end`
and `batch_end >= 0`. At the end of the body, `batch_end` leaves scope, and the
prover no longer knows `batch_start >= 0`.

The same happens to pointer facts. A fact such as
`Init(p, start, start + w)` ends with `let start`, and so does a loop invariant
that names the counter of a `for` loop after the loop.

Reproduction:

```vx
fn batches(samples: i32, batch_size: i32) -> i32
requires samples > 0 && batch_size > 0
{
  let mut batch_start = 0;
  while batch_start < samples invariant batch_start >= 0 {
    let mut batch_end = samples;
    if batch_size < samples - batch_start {
      batch_end = batch_start + batch_size;
    }
    batch_start = batch_end;
  }
  return batch_start;
}

fn main() -> i32 { return 0; }
```

Current:

```
Error[E8009] at fact_ends_with_scope.vx:5:53: this loop invariant `batch_start >= 0` may not hold after a pass through the loop body: the compiler cannot prove it from what it knows at the end of the body
```

The same loop passes when `let mut batch_end = 0;` moves before the loop.

Requested: at the end of a scope, the prover keeps the facts with the variable
replaced by a fresh name, the same way an assignment keeps the old value as
`$old_N`. The fresh name is then unconstrained except by those facts. A pointer
fact whose range names the variable can do the same.

VxVector workaround: `batch_end` is declared outside its loop, and the loops whose
facts matter count with `while` and a counter declared before the loop.

---

## 6. enhancement: Prover: model integer division

Related: #833 (integer division by zero semantics).

The prover does not model `/`. It prints "the prover ignores a condition here: it
cannot reason about the `Div` operator", and it cannot prove even simple facts
about a quotient.

Reproduction:

```vx
fn spread(center: i64, samples: i64, k: i64) -> i64
requires center >= 0i64 && samples > 0i64 && k > 0i64
ensures return >= 0i64
{
  return center * samples / k;
}

fn main() -> i32 { return 0; }
```

Current:

```
Warning in div_not_modeled.vx: the prover ignores a condition here: it cannot reason about the `Div` operator
Error[E8001] at div_not_modeled.vx:5:3: Function 'spread' cannot prove postcondition (ensures) at compile time
```

Requested: model `/` for integers with Vx's truncating semantics, as #2899 did
for `%`. With a nonzero divisor, the prover can state
`a == (a / b) * b + a % b` and `|a % b| < |b|`, with the sign of `a % b` the sign
of `a`. That is enough for `0 <= a / b <= a` when `a >= 0` and `b > 0`.

VxVector use case: the evenly spaced start positions in `spaced_position`. Today
it adds `+ samples` and a second `% samples`, which change nothing at run time,
so that the prover can see that the result is not negative.

---

## 7. bug (reduce first): Raw-pointer facts: `!frees` is not inferred for a parameter that only reaches `!frees` callees

Not ready to file. It does not reproduce in a small program yet.

A smaller case in VxVector now: `sample_objective` has the same problem. It is one
`for` loop that passes `offset(data, row * width)` to `nearest` and
`centroid_distance`, and it needs a written `ensures !frees(data)`. Without it,
`run_restart` gets E8012: "its body can free memory through `data`: it gives
`data` to 'sample_objective', which has no `!frees` for that parameter". Copies
of that loop with a plain `read` in place of `nearest` and `centroid_distance` do
infer `!frees`, so those two callees are the next place to look.

What was seen first: in VxVector's `src/kmeans.vx`, `run_restart` passes `data`
(a `*const f32`) only to `offset`, and the results go only to `copy_vector`,
`nearest`, `accumulate_point` and `centroid_distance`. Probes show that each of
those gets `!frees` for that parameter. But without a written
`ensures !frees(data)` on `run_restart`, each call of `run_restart` ends the
facts of its caller: the audit goes from 2 trusted clauses inside the kernel to 15.
With the clause written, the checker accepts it, so the body does meet it.
`current`, `sums` and `counts` get `!frees` inferred.

Ruled out in small programs, which all infer `!frees`: a constant or a variable
offset, an offset in a `for` loop, a pointer kept in a typed `let` first, and a
function that returns a struct. Removing either the seeding loop or the
objective loop of `run_restart` alone does not fix it.

To reduce it: start from `run_restart` and remove statements one at a time
while the probe still loses the facts. Look for a method call on a value that the
inference follows. #2836 says that the inference stops there. If the cause is
#2836, comment there with the use case instead of filing.

---

## 8. bug (reduce first): Prover: a null check in an `||` condition with many terms loses the facts of the other terms

Not ready to file. It does not reproduce in a small program yet.

What was seen: in `vxvector_kmeans_v1`, this guard returns early:

```vx
if points == null<f32>() || nodes == null_mut<f32>() || ... ||
  rows <= 0 || leaf_count > rows ||
  root_count < 0 || root_count > leaf_count ||
  i64::from(root_count) + i64::from(leaf_count) >= 2147483647 ||
  sampling_factor <= 0 || batch_size <= 0 || iterations <= 0 ||
  restarts <= 0 || metric < 0 || metric > 2 || seed < 0 {
  return 1;
}
```

After it, the calls of `train` get E8006 for `k > 0 && dims > 0 && rows >= k` and
for `sampling_factor > 0 && ... && seed >= 0`. Splitting the null checks into their
own `if` fixes it. With only `points == null<f32>()` left in the condition, two
of the four errors stay. Removing the `i64::from` term does not help.

Ruled out in small programs, which all pass: one null check with three integer
terms, one null check with one term, and a call to `i64::from` in a disjunct.

To reduce it: copy `vxvector_kmeans_v1` with the joined condition, replace
`train` with a function that has only its integer `requires`, and remove terms
and statements while the error stays. The loops between the guard and the call,
which read `points` and write `nodes`, are one candidate.

---

## 9. enhancement: Prover: facts about the number fields of a struct

Related: #2832, which asks the same question for a pointer field and needs a
decision on type invariants. This draft covers only number fields, and needs no
new syntax.

The prover does not use facts about the fields of a struct. A `requires` on a
field does not help the body prove an `ensures` on that field. A caller does not
know the fields of a struct literal it just built.

Reproduction:

```vx
import core::convert;

struct Sampler {
  rows: i32,
  offset: i32,
  stride: i32,
}

fn sample_row(sampler: Sampler, position: i32) -> i32
requires sampler.rows > 0 && sampler.offset >= 0 && sampler.stride >= 0 && position >= 0
ensures return >= 0 && return < sampler.rows
{
  let rows: i32 = sampler.rows;
  let offset: i32 = sampler.offset;
  let stride: i32 = sampler.stride;
  let index = i64::from(offset) + i64::from(position) * i64::from(stride);
  return (index % i64::from(rows)) as i32;
}

fn needs_rows(n: i32) -> i32 requires n > 0 { return n; }

fn caller(rows: i32) -> i32
requires rows > 0
{
  let sampler = Sampler { rows: rows, offset: 0, stride: 1 };
  return needs_rows(sampler.rows);
}

fn main() -> i32 { return 0; }
```

Current:

```
Error[E8001] at s2.vx:17:3: Function 'sample_row' cannot prove postcondition (ensures) at compile time
Error[E8006] at s2.vx:26:10: this call to 'needs_rows' may not meet its precondition `n > 0`: the compiler cannot prove it from what is known here
```

The same function with `rows`, `offset` and `stride` as parameters proves its
`ensures`. Passing the struct as `&Sampler`, or reading the fields directly
without the locals, fails the same way.

Requested: a struct of numbers that is a local variable or a parameter gets one
prover name per field, as a number variable does. A `requires` or `ensures` on
`s.f` becomes a fact on that name. A struct literal gives `s.f == value`. An
assignment to `s.f`, or a call that gets `&mut s`, changes the name as it does for
a number variable (#2620).

VxVector use case: C4 in the usage review asks for at most eight arguments per
function. `train` and `run_restart` have 15. A `Sampler { rows, samples, offset,
stride }` and a `LloydOptions { iterations, batch_size, spherical }` would bring
them under eight. Today each of those fields carries a fact that a proof needs,
so grouping them would turn proved clauses into trusted ones.
