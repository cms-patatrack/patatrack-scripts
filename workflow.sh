#! /bin/bash

# create the Pixel-only workflow for running on CPU over 2018 MC samples with "ideal" conditions
runTheMatrix.py -j 0 -t 16 --command='--conditions auto:phase1_2018_design' -l 10824.5

cp 10824.5_*/step3_*.py profile_pixel-only_CPU.py
cat >> profile_pixel-only_CPU.py << @EOF

# load the CUDA service, but disable it for running on CPU
process.CUDAService = cms.Service("CUDAService",
    enabled = cms.untracked.bool(False)
)

# customise the configuration for profiling the Pixel-only workflow on CPU
from RecoPixelVertexing.Configuration.customizePixelTracksSoAonCPU import customizePixelTracksSoAonCPUForProfiling
process = customizePixelTracksSoAonCPUForProfiling(process)

# load data using the DAQ source
del process.source
process.load('sourceFromPixelRaw_cff')

# the raw data do not have the random number state
del process.RandomNumberGeneratorService.restoreStateLabel

# build triplets and run the broken line fit
process.caHitNtupletCUDA.minHitsPerNtuplet = 3
process.caHitNtupletCUDA.includeJumpingForwardDoublets = True
process.caHitNtupletCUDA.useRiemannFit = False

# report CUDAService messages
process.MessageLogger.categories.append("CUDAService")

# print the summary
process.options.wantSummary = cms.untracked.bool( True )
@EOF


# create the Pixel-only workflow for running on GPU over 2018 MC samples with "ideal" conditions:
runTheMatrix.py -j 0 -t 16 --command='--conditions auto:phase1_2018_design' -l 10824.502

cp 10824.502_*/step3_*.py profile_pixel-only_GPU.py
cat >> profile_pixel-only_GPU.py << @EOF

# customise the configuration for profiling the Pixel-only workflow on GPU
from RecoPixelVertexing.Configuration.customizePixelTracksForProfiling import customizePixelTracksForProfilingGPUOnly
process = customizePixelTracksForProfilingGPUOnly(process)

# load data using the DAQ source
del process.source
process.load('sourceFromPixelRaw_cff')

# the raw data do not have the random number state
del process.RandomNumberGeneratorService.restoreStateLabel

# build triplets and run the broken line fit
process.caHitNtupletCUDA.minHitsPerNtuplet = 3
process.caHitNtupletCUDA.includeJumpingForwardDoublets = True
process.caHitNtupletCUDA.useRiemannFit = False

# report CUDAService messages
process.MessageLogger.categories.append("CUDAService")

# print the summary
process.options.wantSummary = cms.untracked.bool( True )
@EOF
