## System architecture

The CPU architecture can be looked at with `lscpu` or `numactl -H`.

The available GPUs can be listed with `nvidia-smi` or `lspci`.

An overview can be given by
```
./compute_info
```

## Generate workflows

Run
```
cmsenv
./workflow.sh
```
to generate `step3.py`, `step4.py` and `profile.py`.

`step3.py` and `profile.py` relie on `sourceFromPixelRaw_cff.py` to replace the ROOT Source with a DAQ Source, configured to read the input data in .raw format.

## Measuring the I/O throughput

Copy the input data to a RAM disk:
```
mkdir -p /dev/shm/fwyzard/store/pixelraw/Run2018D/JetHT/RAW/v1/000/321/177/00000
cp ~/data//store/pixelraw/Run2018D/JetHT/RAW/v1/000/321/177/00000/* /dev/shm/fwyzard/store/pixelraw/Run2018D/JetHT/RAW/v1/000/321/177/00000/
```

Update `sourceFromPixelRaw_cff.py` to point to the .raw files on the RAM disk.
Good parameters for achieving a high trhouput with he DAQ source, in in evironment with multiple input files of roughly ~230 MB each, are
```
    process.source.eventChunkBlock  = cms.untracked.uint32( 240 )
    process.source.eventChunkSize   = cms.untracked.uint32( 240 )
    process.source.maxBufferedFiles = cms.untracked.uint32( 8 )
    process.source.numBuffers       = cms.untracked.uint32( 8 )
```

The configuration file `readFromPixelRaw.py` can be used to measure the I/O throughput.
With a single job:
```
./multiRun.py readFromPixelRaw.py
```


```
for N in `seq 4`; do CUDA_VISIBLE_DEVICES=0 numactl -N 0 cmsRun profile.py 2>&1 | ./single_throughput.py; done
```

## Benchmark: run N jobs with T threads and S streams over N GPUs

The most efficient way to run seems to be to run N jobs, each one using a dedicated GPU.
The script
```
./multiRun.py CONFIG
```
can be used to run a "warm up" job to make sure all binaries, data and conditions are cached, followed by any number of different configurations.
The logs are automatically analysed to give the total throughput for each set of jobs.

