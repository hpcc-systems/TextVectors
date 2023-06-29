/*##############################################################################
## HPCC SYSTEMS software Copyright (C) 2019 HPCC Systems.  All rights reserved.
############################################################################## */
IMPORT $.^.Types;
IMPORT std.System.Thorlib;
IMPORT $ as int;
IMPORT Std;

nNodes := Thorlib.nodes();
node := Thorlib.node();

slice := Types.slice;
sliceExt := Types.sliceExt;
CSlice := Types.CSlice;
TrainingDat := Types.TrainingDat;
NegRec := Types.NegativeRec;

/**
  * Synchronous Gradient Descent.
  * <p>Perform distributed training of a neural network using Synchronous Batch Gradient Descent.
  * <p>Weights are carried as slices, which are managed by the Weights module.
  * @param shape The shape of the neural network expressed as a set of dimensions.  For example,
  *        [100, 10, 100] means that there are 100 weights in the input layer, 10 in the hidden layer
  *        and 100 in the output layer.
  * @param trainToLoss Stop training when the average Loss level is reached.  This is the recommended method
  *        of controlling training duration. The Default .05 (i.e. 5 percent loss) seems to give
  *        good results.  
  * @param numEpochs The maximum number of times to loop through the full set of training data. Each
  *                  pass through the training data is known as an Epoch.  Default 0 (recommended)
                     means no fixed
  *                  number of epochs.  Will use trainToLoss and noProgressEpochs to terminate.
  * @param miniBatchSize The number of training samples to train on before re-synchronizing the weights
  *                   across nodes.  The larger the number, the faster it will run, but a number too
  *                   large may make convergence difficult.  1000 seems to be a good number.
  * @param lr The maximum learning rate to use.  .1 seems to be a good default level.
  * @param negSamp The number of negative samples to use for each main word during training.
  * @param noProgressEpochs The number of epochs without progress before early termination of training.
  */
EXPORT SGD(SET OF INTEGER4 shape, REAL trainToLoss=.05, UNSIGNED numEpochs=0, UNSIGNED miniBatchSize=1000, REAL lr=.1, UNSIGNED negSamp=10,
                  UNSIGNED noProgressEpochs = 5) := MODULE
  SHARED COMPRESS_UPDATES := IF(nNodes > 1, TRUE, FALSE);
  SHARED w := int.Weights(shape);  // Module for managing weights.
  /**
    * Experimental. Factor by which to reduce the learning rate
    * for each epoch in which no forward progress on loss reduction
    * has been made. Values less than 1 imply that we will
    * take smaller steps if we can't make progress with larger
    * steps.
    */
  SHARED LR_Progress_Factor := .75;
  /**
    * Experimental. Factor by which to reduce the batch size
    * for each epoch in which no forward progress on loss reduction
    * has been made. Values less than 1 imply that we will
    * take synchronize more often if we can't make progress with larger
    * intervals.
    */
  SHARED Batch_Size_Prog_Factor := .75;

  /**
    * Randomize the order of a set of training samples.
    */
  SHARED RandomizeSamples(DATASET(trainingDat) dIn) := FUNCTION
    tempRec := RECORD(trainingDat)
      UNSIGNED4 rnd;
    END;
    d0 := PROJECT(dIn, TRANSFORM(tempRec, SELF.rnd := RANDOM(), SELF := LEFT), LOCAL);
    d1 := SORT(d0, rnd, LOCAL);
    dOut := PROJECT(d1, trainingDat);
    return dOut;
  END;

  /**
    * Pairwise addition of two sets of weights.
    * <p>Returns [w1[1] + w2[1], ... ,w1[N] + w2[N]].
    */
  SHARED SET OF REAL8 addWeights(SET OF REAL8 w1, SET OF REAL8 w2, UNSIGNED4 numweights) := EMBED(C++)
    #body
    __lenResult = (size32_t) (numweights * sizeof(double));
    double *wout = (double*) rtlMalloc(__lenResult);
    __isAllResult = false;
    __result = (void *) wout;
    double *ww1 = (double *) w1;
    double *ww2 = (double *) w2;
    for (uint32_t i = 0; i < numweights; i++)
    {
      wout[i] = ww1[i] + ww2[i];
    }
  ENDEMBED;
  /**
    * Calculate progress at reducing average loss.
    * Loss is an unknown approximately exponentially decaying function
    * w.r.t time (i.e. number of epochs).
    * We would like Progress to be approximately linear with respect to
    * time.
    * So we calculate a scaled time-linear version of loss (0 < loss < 1), and then
    * treat progress as 1 - loss.
    */
  SHARED calcProgress(REAL avgLoss) := FUNCTION
    // Scale loss into range [0,1].  The highest non-spurious loss should be .5.
    // The lowest interesting loss is trainToLoss, because that's where we stop.
    // Protect the bounds to make sure we don't get a loss > .5.
    avL := MIN(.5, avgLoss);
    // Scaled Loss at .5 should be 1, and at trainToLoss should be 0.
    scaledL(REAL4 x) := MAX(0, (x - trainToLoss) / (.5 - trainToLoss));
    // The multiplier is heuristic since we don't know the actual exponential
    // This has the effect of moving up on the logarithm curve, where the slope is shallower.
    logMultiplier := 150;
    logL(REAL4 x) := LN(1 + logMultiplier * scaledL(x));
    // Scaled Log Loss [0,1]. But reduce the range to below 15.
    // That is because the first part of the log reduction curve is nearly vertical
    // and we can't compensate for it.  ScaledLL will be .99 until we get down to
    // 15.
    minLinear := .15;
    scaledLL(REAL4 x) := MIN(1, (LogL(x)) / logL(minLinear));
    linLoss := scaledLL(avL);
    // Always report some progress (.01) even when we're above trainToLoss * 2.
    // Square the progress to account for the reducing effect of Learning Rate which
    // is reduced with progress.
    altProgress := (.5 - avL) * .1;
    progress := MAX(.01 , POWER(1 - linLoss, 2), altProgress);
    return progress;
  END;
  /**
    *  Take in a set of weight slices to be updated on this node plus a set of updates (deltas)
    *  to apply to those weights (multiple slices).
    *  Roll up the results by adding the base weights and updates for each slice to produce
    *  a new set of weights for each slice.
    *  Then replicate the weight slices back to all nodes for the next round of processing.
    *  Note: There will typically be one update for each node, plus the shared (i.e. replicated)
    *  base-weights.  Before calling this rollup, update slices should be distributed by sliceId
    *  and sorted (locally) by sliceId.  At the end of this rollup, the updates have been
    *  applied to produce a new set of weights that are replicated to all nodes (i.e.
    *  synchronized.  Sort is always by sliceId locally.
    */
  SHARED rollUpdates(DATASET(SliceExt) inWeights, DATASET(SliceExt) updates) := FUNCTION
    combined := SORT(inWeights+updates, sliceId, LOCAL);
    SliceExt doRollup(SliceExt l, SliceExt r) := TRANSFORM
      SELF.weights := addWeights(l.weights, r.weights, w.sliceSize);
      SELF.loss := l.loss + r.loss;
      // To avoid premature stopping, use the max loss for all nodes and
      // the max epoch of minimum loss.
      SELF.minLoss := MAX(l.minLoss, r.minLoss);
      SELF.minEpoch := MAX(l.minEpoch, r.minEpoch);
      SELF := l;
    END;
    outWeights := ROLLUP(combined, doRollup(LEFT, RIGHT), sliceId, LOCAL);
    outWeightsD := w.distributeAllSlices(outWeights);
    outWeightsS := SORT(outWeightsD, sliceId, LOCAL);
    RETURN outWeightsS;
  END;
  /**
    * Same as rollUpdates above, but taking a set of compressed slices.
    * <p>Decompresses the slices and then calls rollUpdates.
    */
  SHARED rollUpdatesC(DATASET(SliceExt) inWeights, DATASET(CSlice) updates) := FUNCTION
    uncompUpdates := w.decompressWeights(updates);
    RETURN rollUpdates(inWeights, uncompUpdates);
  END;
  /**
    * Determine if training is complete, either because we have reached our convergence goal,
    * run all requested Epochs, or because training has stalled.
    */
  SHARED BOOLEAN isConverged(DATASET(SliceExt) slices, UNSIGNED c, UNSIGNED trainSize) := FUNCTION
    isStalled := c - slices(nodeId = node AND sliceId = 1)[1].minEpoch > noProgressEpochs;
    isFinished := numEpochs > 0 AND c > numEpochs;
    firstW := slices(nodeId = node AND sliceId = 1)[1]; 
    loss := firstW.loss;
    avgLoss := loss / (trainSize * (1 + negSamp));
    isConverged := IF(loss > 0, avgLoss < trainToLoss, FALSE);
    RETURN isStalled OR isFinished OR isConverged;
  END;
  /**
    * Determines whether an epoch is complete by comparing the records processed
    * with the training set size.
    */
  SHARED BOOLEAN isEpochDone(DATASET(SliceExt) slices, UNSIGNED4 trainSize) := FUNCTION
    batchPos := slices(nodeId = node AND sliceId = 1)[1].batchPos;
    //batchPos := ASSERT(slices(nodeId = node AND sliceId = 1), FALSE, 'batchPos = ' + batchPos)[1].batchPos;
    done := batchPos >= trainSize;
    RETURN done;
  END;
  /**
    * Train the network and return weights duplicated on each node.
    * <p>Returns weights  duplicated on all nodes using the
    * SliceExt format.  If you want a single set of weight slices rather than a
    * replicated set of extended slices, use Train(...) below.
    */
  EXPORT DATASET(SliceExt) Train_Dupl(DATASET(trainingDat) trainData) := FUNCTION
    // Initialize the weights to random values
    initWeights := w.initWeights;
    // Get the size of each slice of the weights (i.e. number of weights)
    sliceSize := w.sliceSize;
    // Copy the weights to all nodes as SliceExt records.
    initWeightsExt := w.toSliceExt(initWeights);
		// Get the size of the local segment of the training data.
    trainSize := TABLE(trainData, {cnt := COUNT(GROUP)}, LOCAL)[1].cnt;
    // LOOP for each Epoch (e.g. 1000)
    DATASET(sliceExt) doEpoch(DATASET(sliceExt) inWeights, UNSIGNED epochNum) := FUNCTION
      R_train := RandomizeSamples(trainData); // Local operation
      zWeights := PROJECT(inWeights, TRANSFORM(RECORDOF(LEFT),
                            SELF.loss := 0,
                            SELF.batchPos := 1,
                            SELF := LEFT), LOCAL);
      noProgress := epochNum - zWeights[1].minEpoch - 1;
      maxNoProgress := MAX(zWeights[1].maxNoProg, noProgress);
      epochLR := IF(maxNoProgress > 0, lr * POWER(LR_Progress_Factor, maxNoProgress), lr);
      epochBatchSize := (UNSIGNED4)IF(maxNoProgress > 0, miniBatchSize * POWER(Batch_Size_Prog_Factor, maxNoProgress),
                            miniBatchSize);
      batchSize := MIN(trainSize, epochBatchSize);
      // LOOP for each mini-batch
      DATASET(sliceExt) doBatch(DATASET(sliceExt) inWeights2, UNSIGNED batchNum) := FUNCTION
        // Walk through the randomized samples taking batchSize at a time.
        firstW := inWeights2(nodeId = node AND sliceId = 1)[1];
        batchPos := firstW.batchPos;
        loss := firstW.loss;
        B_train := CHOOSEN(R_train, batchSize, batchPos, LOCAL);
        nTrainRecs := TABLE(B_train, {cnt := COUNT(GROUP)}, LOCAL)[node + 1].cnt;
        // Do the gradient descent, and return the weight updates
        // Decrease learning rate as we proceed so that we can better converge.
        avgLoss := loss / (nNodes * batchSize * batchNum * (1 + negSamp));
        progress := calcProgress(avgLoss);
        adjLR := MAX((1 - progress), .1) * epochLR;
        // Train the neural network and get updated weights.
        tempPrint := Std.System.Log.addWorkunitInformation('batchPos = ' + batchPos +
                ', rtrain = ' + COUNT(R_train) +
                ', weightSlots = ' +  w.nWeightSlots +
                ', sliceSize = ' + sliceSize + ', inWeights2 = ' + COUNT(inWeights2) + ', nWeights = ' + COUNT(inWeights2[1].weights));
        //DATASET(sliceExt) wUpdates := WHEN(int.svTrainNN(inWeights2, B_train, sliceSize, w.nWeightSlots,
        //      shape[1], shape[2], nTrainRecs, adjLR, negSamp), tempPrint); // C++
        DATASET(sliceExt) wUpdates := int.svTrainNN(inWeights2, B_train, sliceSize, w.nWeightSlots,
              shape[1], shape[2], nTrainRecs, adjLR, negSamp); // C++
        // Distribute the updates by sliceId for rollup.  Compress the updates if needed.
        wUpdatesC := w.compressWeights(wUpdates);
        wUpdatesDC := DISTRIBUTE(wUpdatesC(LENGTH(cweights) > 0), sliceId);
        wUpdatesD := DISTRIBUTE(wUpdates, sliceId);
        // Now apply the updates on the nodes assigned by sliceId, and then re-replicate to all nodes.
        newWeightsC := rollUpdatesC(inWeights2(sliceId % nNodes = node), wUpdatesDC);
        newWeightsN := rollUpdates(inWeights2(sliceId % nNodes = node), wUpdatesD);
        newWeights0 := IF(COMPRESS_UPDATES, newWeightsC, newWeightsN);
        // Continue the loop with replicated weights.
        newWeights1 := PROJECT(newWeights0, TRANSFORM(RECORDOF(LEFT),
                                                      SELF.batchPos := batchPos + nTrainRecs,
                                                      SELF := LEFT), LOCAL);
        firstW2 := newWeights1(nodeId = node AND sliceId = 1)[1];
        loss2 := firstW2.loss;
        status := Std.System.Log.addWorkunitInformation('Status: Initial Loss = ' +
                  ROUND(loss2 / (nNodes * batchSize * (1 + negSamp)), 6));
        newWeights := IF(epochNum = 1 AND batchNum = 1, WHEN(newWeights1, status), newWeights1);
        RETURN newWeights;
      END;
      epochWeights0 := LOOP(zWeights, TRUE, NOT isEpochDone(ROWS(LEFT), trainSize) , doBatch(ROWS(LEFT), COUNTER));
      //epochWeights0 := LOOP(zWeights, nBatches, doBatch(ROWS(LEFT), COUNTER));
      // Mark the loss information in each slice.
      isBest(SliceExt rec) := rec.loss < rec.minLoss;
      epochWeights := PROJECT(epochWeights0, TRANSFORM(RECORDOF(LEFT),
                                    SELF.minLoss := IF(isBest(LEFT), LEFT.loss, LEFT.minLoss),
                                    SELF.minEpoch := IF(isBest(LEFT), epochNum, LEFT.minEpoch),
                                    SELF.maxNoProg := maxNoProgress,
                                    SELF := LEFT), LOCAL);
      firstW := epochWeights(nodeId = node AND sliceId = 1)[1];
      loss := firstW.loss;
      avgLoss := loss / (COUNT(trainData) * (1 + negSamp));
      progress := calcProgress(avgLoss);
      adjLR := MAX((1 - progress), .1) * epochLR;
      minEpoch := firstW.minEpoch;
      status := Std.System.Log.addWorkunitInformation('Status: ' + 'Epoch = ' + epochNum +
                                ', Progress = ' + (DECIMAL5_2) (progress * 100) +
                                '%, Loss = ' + (DECIMAL6_6)avgLoss +
                                ', minEpoch = ' + minEpoch +
                                ', LR = ' + (DECIMAL6_6)adjLR +
                                ', batchSize = ' + batchSize);
      RETURN WHEN(epochWeights, status);
    END;
    finalWeights := LOOP(InitWeightsExt, TRUE, NOT isConverged(ROWS(LEFT), COUNTER, COUNT(trainData)), doEpoch(ROWS(LEFT), COUNTER));
    firstW := finalWeights(nodeId = node AND sliceId = 1)[1];
    loss := firstW.loss;
    status := Std.System.Log.addWorkunitInformation('Status: Final Loss = ' +
                    ROUND(loss / (COUNT(trainData) * (1 + negSamp)), 6));
    RETURN WHEN(finalWeights, status);
  END;

  /**
    * Train the network and return a set of weights distributed among the HPCC nodes
    * by SliceId.
    * If you want a set of weights that are DUPLICATED on each node (i.e. for LOCAL processing
    * on each node), use Train_Dupl above.
    */
  EXPORT DATASET(Slice) Train(DATASET(trainingDat) trainData) := FUNCTION
    // Train the model and get weights replicated on every node.
    finalWeights := Train_Dupl(trainData);
    // At this point, weights are synchronized and duplicated on each node
    // Filter the weights so that there is only one copy and it is (effectively) distributed
    // by sliceId.
    finalWeightsDedup := w.fromSliceExt(finalWeights);
    RETURN finalWeightsDedup;
  END;
END;
