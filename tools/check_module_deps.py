#!/usr/bin/env python3
# provenance: original
# Copyright (c) 2026 John Lipka and the Pelias contributors
"""Boundary checks of masterwork 3.4 for the product tree (standard library only).

    python3 tools/check_module_deps.py            exit 0 when every check passes, 1 otherwise

BC1  the module graph is acyclic and every include between modules is an edge allowed by 14.3; no product module
     includes from tests/ or third_party/
BC3  product files include only repository, standard-library, GPU-runtime, intrinsics and (inside os/ and
     hal/host_alloc_*) OS headers
BC4  line 1 of every source file is a provenance line and the copyright line follows
BC6  OS headers only in src/os/ and src/hal/host_alloc_*
"""
import os, re, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
SRC = os.path.join(ROOT, 'src')
INC = os.path.join(ROOT, 'include')
EXTS = ('.h', '.hpp', '.cc', '.cpp', '.cu', '.cuh')

# 14.3, allowed includes per module (a module may always include itself and include/pelias)
ALLOWED = {
    'os': set(), 'hal': {'os'}, 'quant': {'hal'}, 'plan': {'hal', 'io'}, 'bank': {'hal'}, 'bus': {'os', 'hal', 'quant'},
    'io': {'os', 'hal', 'quant'}, 'tok': {'io'}, 'kern': {'hal', 'quant'}, 'model': {'hal', 'quant', 'io', 'kern'},  # model -> kern: GP3-10 fast CPU kernels
   
    'ingest': {'hal', 'quant', 'io'}, 'data': {'io', 'tok'}, 'stream': {'hal', 'bus', 'model'},
    'sched': {'hal', 'kern', 'model', 'stream'}, 'train': {'hal', 'data', 'model', 'sched'},
    'export': {'hal', 'quant', 'io', 'model', 'bank'}, 'verify': {'hal', 'io', 'quant'}, 'cpu': {'os', 'hal', 'quant', 'bus'},
    'gpu': {'hal', 'quant', 'bank', 'bus', 'kern'}, 'engine': {'os', 'hal', 'plan', 'bus', 'bank', 'quant', 'io', 'model', 'kern', 'cpu', 'gpu'},
    'server': {'os', 'io', 'tok', 'engine', 'web', 'flow'}, 'web': {'os', 'io'}, 'project': {'io'},
    'startup': {'os', 'hal', 'plan', 'io', 'project'}, 'intake': {'io', 'tok', 'data'}, 'warn': {'io', 'data', 'verify', 'intake'},
    'flow': {'plan', 'startup', 'project', 'intake', 'warn', 'verify', 'help'}, 'help': set(), 'cli': None,  # None = everything
}
# the masterwork names plan -> hal; plan also reads the tensor table through io (14.3 lists io -> os, hal, quant), so
# plan -> io is an edge added at step 2 and recorded in BUILD_LOG (acyclic: io does not depend on plan)
OS_HEADERS = {'windows.h', 'winsock2.h', 'dlfcn.h', 'fcntl.h', 'unistd.h', 'sys/mman.h', 'sys/stat.h', 'sys/types.h', 'sys/resource.h',
              'pthread.h', 'malloc.h', 'ws2tcpip.h', 'io.h', 'direct.h', 'signal.h', 'setjmp.h', 'sched.h', 'sys/socket.h', 'netinet/in.h',
              'arpa/inet.h', 'netdb.h', 'poll.h', 'sys/epoll.h', 'sys/uio.h', 'sys/time.h', 'time.h'}
GPU_HEADERS = {'cuda_runtime.h', 'cuda.h', 'cuda_bf16.h', 'cuda_fp16.h', 'cuda_fp8.h', 'cooperative_groups.h', 'device_launch_parameters.h', 'vulkan/vulkan.h', 'nvml.h'}
INTRINSICS = {'immintrin.h', 'emmintrin.h', 'xmmintrin.h', 'arm_neon.h', 'intrin.h', 'x86intrin.h'}
STD_RE = re.compile(r'^[a-z_0-9]+$')  # <vector>, <cstdint>, <string_view> ...
INC_RE = re.compile(r'^\s*#\s*include\s*([<"])([^>"]+)[>"]')
problems = []


def module_of(path):
    rel = os.path.relpath(path, SRC).replace('\\', '/')
    return rel.split('/')[0]


def walk(d):
    for dp, dn, fn in os.walk(d):
        dn[:] = [x for x in dn if x not in ('build', 'third_party')]
        for f in fn:
            if f.endswith(EXTS):
                yield os.path.join(dp, f)


def check_provenance(path, lines):
    if not lines or not lines[0].startswith(('// provenance:', '# provenance:')):
        problems.append(f'BC4 {os.path.relpath(path, ROOT)}: line 1 is not a provenance line')
        return
    # the provenance line may continue on indented comment lines; the copyright line follows them (within 5 lines)
    for k in range(1, min(6, len(lines))):
        if 'Copyright (c) 2026 John Lipka and the Pelias contributors' in lines[k]:
            return
        if not lines[k].lstrip().startswith(('//', '#')):
            break
    problems.append(f'BC4 {os.path.relpath(path, ROOT)}: the copyright line does not follow the provenance line')


edges = {}
for path in list(walk(SRC)) + list(walk(INC)):
    with open(path, encoding='utf-8') as f:
        lines = f.read().split('\n')
    check_provenance(path, lines)
    rel = os.path.relpath(path, ROOT).replace('\\', '/')
    in_src = path.startswith(SRC)
    mod = module_of(path) if in_src else 'include'
    os_ok = in_src and (mod == 'os' or rel.startswith('src/hal/host_alloc_'))
    for n, line in enumerate(lines, 1):
        m = INC_RE.match(line)
        if not m:
            continue
        kind, name = m.group(1), m.group(2)
        if kind == '<':
            if name in OS_HEADERS:
                if not os_ok:
                    problems.append(f'BC6 {rel}:{n}: OS header <{name}> outside src/os/ and src/hal/host_alloc_*')
            elif name in GPU_HEADERS or name in INTRINSICS or STD_RE.match(name):
                pass
            else:
                problems.append(f'BC3 {rel}:{n}: header <{name}> is not standard, GPU-runtime, intrinsics or OS')
        else:
            if name.startswith(('tests/', 'third_party/', '../tests', '../third_party')):
                problems.append(f'BC1 {rel}:{n}: product file includes "{name}"')
                continue
            if name.startswith('pelias/'):
                continue
            target = name.split('/')[0]
            if in_src and target != mod:
                edges.setdefault(mod, set()).add(target)
                allowed = ALLOWED.get(mod)
                if allowed is not None and target not in allowed:
                    problems.append(f'BC1 {rel}:{n}: {mod} -> {target} is not an edge of 14.3')

# acyclic
def has_cycle():
    state = {}
    def visit(u):
        state[u] = 1
        for v in edges.get(u, ()):
            if state.get(v) == 1:
                return [u, v]
            if state.get(v) is None:
                r = visit(v)
                if r:
                    return [u] + r
        state[u] = 2
        return None
    for u in list(edges):
        if state.get(u) is None:
            r = visit(u)
            if r:
                return r
    return None

cyc = has_cycle()
if cyc:
    problems.append('BC1 cycle: ' + ' -> '.join(cyc))
for m in sorted(edges):
    print(f'{m} -> {", ".join(sorted(edges[m]))}')
for p in problems:
    print('PROBLEM', p)
print('RESULT:', 'all checks passed' if not problems else f'{len(problems)} problem(s)')
sys.exit(1 if problems else 0)
