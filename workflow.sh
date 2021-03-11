#! /bin/bash

# change --mc to --data
# change conditions to 102X_dataRun2_HLT_v2
# change number of events to 4200
# remove Validation sequences
# for profiling, add one of
#   --customise RecoPixelVertexing/Configuration/customizePixelTracksForProfiling.customizePixelTracksForProfiling
#   --customise RecoPixelVertexing/Configuration/customizePixelTracksForProfiling.customizePixelTracksForProfilingDisableConversion
#   --customise RecoPixelVertexing/Configuration/customizePixelTracksForProfiling.customizePixelTracksForProfilingDisableTransfer
# add the customisations below

# step 3
cmsDriver.py step3 \
    --data \
    --era Run2_2018 \
    --geometry DB:Extended \
    --conditions 102X_dataRun2_HLT_v2 \
    -s RAW2DIGI:RawToDigi_pixelOnly,RECO:reconstruction_pixelTrackingOnly,DQM:@pixelTrackingOnlyDQM \
    --procModifiers gpu \
    -n 4200 \
    --nThreads 8 \
    --runUnscheduled \
    --filein file:step2.root \
    --fileout file:step3.root \
    --datatier GEN-SIM-RECO,DQMIO \
    --eventcontent RECOSIM,DQM \
    --python_filename step3.py \
    --no_exec

cat >> step3.py <<@EOF

# load data using the DAQ source
del process.source
process.load('sourceFromPixelRaw_cff')

# do not run the Riemann fit
process.pixelTracksHitQuadruplets.doRiemannFit = False

# print a message every 100 events
process.MessageLogger.cerr.FwkReport.reportEvery = 100

# print the summary
process.options.wantSummary = cms.untracked.bool( True )
@EOF


# step 4
cmsDriver.py step4 \
    --data \
    --scenario pp \
    --era Run2_2018 \
    --geometry DB:Extended \
    --conditions 102X_dataRun2_HLT_v2 \
    -s HARVESTING:@pixelTrackingOnlyDQM \
    -n 4200 \
    --filetype DQM \
    --filein file:step3_inDQM.root \
    --fileout file:step4.root \
    --python_filename step4.py \
    --no_exec

# profiling
cmsDriver.py profile \
    --data \
    --era Run2_2018 \
    --geometry DB:Extended \
    --conditions 102X_dataRun2_HLT_v2 \
    -s RAW2DIGI:RawToDigi_pixelOnly,RECO:reconstruction_pixelTrackingOnly,DQM:@pixelTrackingOnlyDQM \
    --procModifiers gpu \
    --customise RecoPixelVertexing/Configuration/customizePixelTracksForProfiling.customizePixelTracksForProfilingDisableTransfer \
    -n 4200 \
    --nThreads 8 \
    --runUnscheduled \
    --filein file:step2.root \
    --fileout file:step3.root \
    --datatier GEN-SIM-RECO,DQMIO \
    --eventcontent RECOSIM,DQM \
    --python_filename profile.py \
    --no_exec

cat >> profile.py <<@EOF

# load data using the DAQ source
del process.source
process.load('sourceFromPixelRaw_cff')

# do not run the Riemann fit
process.pixelTracksHitQuadruplets.doRiemannFit = False

# print the summary
process.options.wantSummary = cms.untracked.bool( True )
@EOF
