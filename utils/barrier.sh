#!/bin/bash
python -c "from mpi4py import MPI; comm=MPI.COMM_WORLD; print(f\"{comm.rank} waiting\")comm.Barrier(); print(f\"Done from {comm.rank}\")"
