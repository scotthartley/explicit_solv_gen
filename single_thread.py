"""Pin the numeric stack to one thread per process -- before numpy loads.

Import this **first**, above every other import, in any module that imports
numpy. The OpenMP runtime numpy pulls in reads these variables once, when it
loads, and setting them afterwards is silently ignored: the process keeps
whatever thread count it started with, and nothing reports that the request
was dropped.

`solvate_md` set them at its own module top, which was correct only when
`solvate_md` was the first thing imported. It was not: `ensemble` imports
numpy, ASE and `report` before it reaches `solvate_md`, and `n_sweep` imports
`ensemble`. So every sweep ran its parent process multi-threaded. Measured on
one n = 2 scoring job, 65.8 s wall with 230 s of system time thrashing,
against 26.8 s wall and 0.4 s of system time with the variable exported in the
shell instead -- a 2.5x tax paid by anything that scored in the parent
process, which until the scoring grid existed was all of it.

Spawned workers were never affected, and the asymmetry is the tell: a child
inherits an environment in which the parent has already set the variable, so
it is in place before *that* process imports numpy at all.

One thread per process is what this pipeline wants regardless. It parallelises
over jobs -- MD runs, then scoring runs -- which is coarser-grained and scales
better than threading a single 25-atom SCF, and N workers each spawning N
threads is how a machine ends up spending more time scheduling than computing.

Set rather than defaulted: a shell that exports OMP_NUM_THREADS=8 would
otherwise get 8 threads in each of 18 workers.
"""

import os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_var] = "1"
