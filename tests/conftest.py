import os

# Keep OpenMP/BLAS thread pools small so the CPU tests run under tight process limits.
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")
