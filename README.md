## Benchmark: run multiple jobs in parallel and measure the total throughput

```bash
./benchmark hlt.py -r 4 -j 2 -t 32 -s 24 -k resources.json DQMIO.root
```

This will run `cmsRun hlt.py` multiple times in parallel, and measure the total throughput across all jobs.
