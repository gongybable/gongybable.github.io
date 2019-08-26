# Machine Learning System

## Model Training
1. **static model** 
    - Trained offline - train the model exactly once and then use that trained model for a while.
    - Easier to build and test.

2. **dynamic model**
    - Trained online - data is continually entering the system and we're incorporating that data into the model through continuous updates.
    - Adapt to changing data.

## Model Inference (Making Predictions)
1. **offline inference** - meaning that you make all possible predictions in a batch, using a MapReduce or something similar. You then write the predictions to an SSTable or Bigtable, and then feed these to a cache/lookup table.
    - Pro: Don’t need to worry much about cost of inference.
    - Pro: Can likely use batch quota or some giant MapReduce.
    - Pro: Can do post-verification of predictions before pushing.
    - Con: Can only predict things we know about — bad for long tail.
    - Con: Update latency is likely measured in hours or days.

2. **online inference** - meaning that you predict on demand, using a server.
    - Pro: Can make a prediction on any new item as it comes in — great for long tail.
    - Con: Compute intensive, latency sensitive—may limit model complexity.
    - Con: Monitoring needs are more intensive.

##