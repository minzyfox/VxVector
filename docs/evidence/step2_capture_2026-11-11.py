"""Record data-path sample rows and plan results from a built library.

Usage: python step2_capture_2026-11-11.py LIBRARY OUTPUT.json. Run it on the
libraries before and after a change, and compare the two files.
"""
import ctypes as c, itertools, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tests'))
import test_data_path as t

lib = c.CDLL(sys.argv[1])
lib.vxvector_plan_cpu_v1.argtypes = [c.POINTER(t.Options), c.POINTER(t.Plan), c.POINTER(t.Error)]
lib.vxvector_run_cpu_v1.argtypes = [c.POINTER(t.Options), t.Next, c.c_void_p, c.POINTER(t.Result), c.POINTER(t.Error)]
lib.vxvector_free_cpu_result_v1.argtypes = [c.POINTER(t.Result)]

samples = {}
for count, k, sf, seed in itertools.product((1, 2, 3, 7, 30, 101, 1000), (1, 2, 5), (1, 3, 7), (0, 1, 42, 2**31 - 1)):
    if k > count:
        continue
    request = t.options(row_count=count, leaf_count=k, sampling_factor=sf, seed=seed, batch_size=3, iterations=1, restarts=1)
    reader = t.Reader([float(i), 0.0] for i in range(count))
    result, error = t.Result(), t.Error()
    status = lib.vxvector_run_cpu_v1(c.byref(request), reader.callback, None, c.byref(result), c.byref(error))
    assert status == 0, error.message
    samples[f'{count},{k},{sf},{seed}'] = list(result.sample_rows[:result.plan.sample_count])
    lib.vxvector_free_cpu_result_v1(c.byref(result))

plans = {}
fields = dict(row_count=(0, 1, 30, 2**31 - 1, 2**40, 2**63 - 1, 2**63), dims=(0, 1, 2, 2**31 - 1),
              leaf_count=(0, 1, 2, 31, 2**31 - 2), root_count=(-1, 0, 1, 3), sampling_factor=(0, 1, 3, 2**31 - 1),
              batch_size=(0, 1, 4, 2**31 - 1), memory_budget_bytes=(0, 100, 2**20, 2**64 - 1))
for values in itertools.product(*fields.values()):
    request = t.options(**dict(zip(fields, values)))
    plan, error = t.Plan(), t.Error()
    status = lib.vxvector_plan_cpu_v1(c.byref(request), c.byref(plan), c.byref(error))
    plans[','.join(map(str, values))] = [status, error.message.decode(), plan.sample_count, plan.batch_rows,
                                         plan.node_count, plan.hash_slots, plan.buffer_bytes]
for name, bad in (('iterations', 0), ('restarts', 0), ('metric', 3), ('metric', -1), ('spherical', 2), ('seed', -1)):
    plan, error = t.Plan(), t.Error()
    status = lib.vxvector_plan_cpu_v1(c.byref(t.options(**{name: bad})), c.byref(plan), c.byref(error))
    plans[f'{name}={bad}'] = [status, error.message.decode(), plan.sample_count, plan.buffer_bytes]
Path(sys.argv[2]).write_text(json.dumps(dict(samples=samples, plans=plans)))
print(len(samples), 'samples', len(plans), 'plans')
