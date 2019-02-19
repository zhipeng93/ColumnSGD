package pku.mllibFP.classfication

import org.apache.spark.{TaskContext}
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.internal.Logging
import org.apache.spark.mllib.linalg.{DenseVector, SparseVector}
import org.apache.spark.rdd.RDD
import pku.mllibFP.util._

import scala.util.Random

/**
  * @param inputRDD      : dataRDD, each partition contains only one element, Array[IndexedDataPoint]
  * @param numFeatures
  * @param numPartitions : number of partitions for the model, e.g., number of tasks per stage
  * @param regParam
  * @param stepSize      : step size for batch
  * @param numIterations
  * @param miniBatchSize
  */
class MLP(@transient inputRDD: RDD[ArrayWorkSet[WorkSet]],
          numFeatures: Int,
          numPartitions: Int,
          regParam: Double,
          stepSize: Double,
          numIterations: Int,
          miniBatchSize: Int,
          numClasses: Int) extends Serializable with Logging {
  val layers = Array(numFeatures, 1000, numClasses)
  /**
    * [executed on executors]
    * generate the model, cache the data, compute the labels also new the intermediateResults Array
    *
    * @param inputRDD
    * @return modelRDD, an combination of data and model.
    **/
  def generateModel(inputRDD: RDD[ArrayWorkSet[WorkSet]]): RDD[(ArrayWorkSet[WorkSet], Array[Array[Array[Double]]], Array[Array[Array[Double]]])] = {
    // generate model
    inputRDD.mapPartitions {
      iter => {
        val model: Array[Array[Array[Double]]] = new Array[Array[Array[Double]]](layers.length - 1)

        // first layer. The input is partitioned by hash or range. Thus, the the first layer may contains more neurons than specified.
        val layerId = 0
        model(layerId) = Array.ofDim(layers(layerId) / numPartitions + 1, layers(layerId + 1))
        // later layers. The input is partitioned by range. Also, the number of neurons in hidden layers must be the number specified.
        // Otherwise it is wrong because each neuron is connected to all other neurons in the input.
        for (layerId <- 1 until (layers.length - 1)) {
          if (TaskContext.getPartitionId() < numPartitions - 1) {
            // the model size for each layer is #neuron / numPartitions + 1
            model(layerId) = Array.ofDim(layers(layerId) / numPartitions + 1, layers(layerId + 1))
          }
          else {
            // last worker
            val last_worker_model_size = layers(layerId) - (numPartitions - 1) * (layers(layerId) / numPartitions + 1)
            assert(last_worker_model_size > 0,
              s"Zero elements in the last worker, increase the #neuron in Layer:${layerId} or decrease #workers")
            model(layerId) = Array.ofDim(last_worker_model_size, layers(layerId + 1))

          }
        }

        // cache all feature maps in memory
        val intermediateResults: Array[Array[Array[Double]]] = new Array[Array[Array[Double]]](layers.length)
        // the size of intermediate result for each hidden layer is #neuron * batchSize.
        // In practice, we only need to 1/#neuron fraction of it. But here, we cache all of them for convienice.
        // no intermediate result for the input layer
        for (layerId <- 1 until (layers.length)) {
          intermediateResults(layerId) = Array.ofDim(layers(layerId), miniBatchSize)
        }
        Iterator((iter.next(), model, intermediateResults))
      }
    }
  }

  /**
    * [executed on executors] compute the intermediate results to be gathered to the driver[not gathered yet].
    * They could be dot product for linear models, (w, V) for factorization machine, (w1, ..., wk) for MLR.
    *
    * @param model
    * @param arrayWorkSet
    * @param seed
    * @param layerId : compute the intermediate result using model from LayerId. LayerId \in [1, layers.length - 1]
    *                **the input is from layerId - 1.**
    *                We use model(id) and hidden(id) to compute hidden(id + 1)
    * @return
    */
  def computeInterResultsFirstLayer(model: Array[Array[Array[Double]]], arrayWorkSet: ArrayWorkSet[WorkSet],
                                    seed: Int, layerId: Int): Array[Array[Double]] = {
    // only handles first hidden layer
    assert(layerId == 1, s"You can only call computeInterResultsFirstLayer() in the first layer due to its libsvm input")
    val neuronSize: Int = layers(layerId)
    val result: Array[Array[Double]] = Array.ofDim[Double](neuronSize, miniBatchSize)
    val model_this_layer: Array[Array[Double]] = model(layerId - 1)
    // size should be #neu (layerId -1) / numPart * neu(layerID)
    assert(model_this_layer.length == layers(layerId - 1) / numPartitions + 1 && model_this_layer(0).length == neuronSize, s"Shape mismatch: " +
      s"model in layer_${layerId} is X * ${model_this_layer(0).length} != X * ${neuronSize}")
    val rand = new Random(seed)
    for (id_batch <- 0 until miniBatchSize) {
      arrayWorkSet.getRandomLabeledPartDataPoint(rand).features match {
        case sp: SparseVector => {
          val indices = sp.indices
          val values = sp.values
          for (id_neuron <- 0 until neuronSize) {
            for (idx <- 0 until indices.length) {
              assert(indices(idx) < model_this_layer.length, s"indices(idx): ${indices(idx)} !< ${model_this_layer.length}")
              assert(id_neuron < model_this_layer(0).length, s"id_neruon: ${id_neuron} !< ${model_this_layer(0).length}")
              result(id_neuron)(id_batch) += values(idx) * model_this_layer(indices(idx))(id_neuron)
            }
          }
        }
        case dp: DenseVector => {
          throw new ColumnMLDenseVectorException
        }
      }
    }
    result
  }

  /**
    *
    * @param model
    * @param lastLayerInput : we use the broadcast intermediate results, but only use one partition of it.
    *                       Here the lastLayerInput is different from ArrayWorkSet in two aspects:
    *                       (1) sparse (libsvm) or dense (Array[Double])
    *                       (2) use part or whole
    *                       Here use
    * @param layerId        : layerId ranges from 1
    * @param workerId       the workerId. This worker handles model with range (workerId * (#neuron / numWorker + 1), (workerId + 1))
    * @return
    */
  def computeInterResultsLaterLayers(model: Array[Array[Array[Double]]], lastLayerInput: Array[Array[Double]],
                                     layerId: Int, workerId: Int): Array[Array[Double]] = {
    // handles next layers, from 2 to the last one
    assert(layerId > 1 && layerId <= layers.length - 1,
      s"You can only call computeInterResultsLaterLayers() in the first layer due to its dense input")
    val neuronSize: Int = layers(layerId)
    val lastNeuronSize: Int = layers(layerId - 1)
    assert(lastLayerInput.length == lastNeuronSize && lastLayerInput(0).length == miniBatchSize,
      s"Shape mismatch: the input from last layer (layer_${layerId - 1}) is ${lastLayerInput.length} * ${lastLayerInput(0).length}" +
        s"!= ${lastNeuronSize}*${miniBatchSize}")

    val result: Array[Array[Double]] = Array.ofDim[Double](neuronSize, miniBatchSize)
    val model_this_layer: Array[Array[Double]] = model(layerId - 1)

    val inputStartIndex = workerId * (lastNeuronSize / numPartitions + 1)
    var inputEndIndex = inputStartIndex + lastNeuronSize / numPartitions + 1
    if (inputEndIndex > lastNeuronSize)
      inputEndIndex = lastNeuronSize
    assert(inputStartIndex < lastNeuronSize,
      s"Zero elements in the last worker in Layer:${layerId}. Please decrease #workers or increase #neurons.")
    assert(model_this_layer.length == inputEndIndex - inputStartIndex,
      s"the size of model partition in Layer:${layerId - 1} is ${model_this_layer.length} * ${model_this_layer(0).length} " +
        s"!= ${inputEndIndex - inputStartIndex} * ${neuronSize}")

    for (id_batch <- 0 until miniBatchSize) {
      for (id_neuron <- 0 until (neuronSize)) {
        // compute result(id_neuron)(id_batch)
        for (tmpId <- inputStartIndex until (inputEndIndex)) {
          // note: although lastLayerInput is not all needed, we still broadcast all of them.
          // this does not hurt the communication a lot since we have to collect the same size.
          // The difference is that 1*cost or 2*cost
          result(id_neuron)(id_batch) += lastLayerInput(tmpId)(id_batch) * model_this_layer(tmpId - inputStartIndex)(id_neuron)
        }
      }
    }

    result
  }

  /**
    * update models in layer[layerId], also collect the grad to the front hidden layer, i.e., layer[layerId - 1]
    *
    * @param model
    * @param inputGrad
    * @param layerId
    * @param workerId
    * @return
    */
  def updateModelAndCollectGradLaterLayers(model: Array[Array[Array[Double]]], inputGrad: Array[Array[Double]],
                                           intermediateResult: Array[Array[Array[Double]]],
                                           layerId: Int, workerId: Int): Array[Array[Double]] = {
    // handles next layers, from 2 to the last one
    assert(layerId > 1 && layerId <= layers.length - 1,
      s"You can only call computeInterResultsLaterLayers() in the first layer due to its dense input")
    val neuronSize: Int = layers(layerId)
    val lastNeuronSize: Int = layers(layerId - 1)
    assert(inputGrad.length == neuronSize && inputGrad(0).length == miniBatchSize,
      s"Shape mismatch when backward in layer ${layerId}: ${inputGrad.length} * ${inputGrad(0).length}" +
        s"!= ${neuronSize}*${miniBatchSize}")

    val model_this_layer: Array[Array[Double]] = model(layerId - 1)

    val inputStartIndex = workerId * (lastNeuronSize / numPartitions + 1)
    var inputEndIndex = inputStartIndex + lastNeuronSize / numPartitions + 1
    if (inputEndIndex > lastNeuronSize)
      inputEndIndex = lastNeuronSize
    assert(inputStartIndex < lastNeuronSize,
      s"Zero elements in the last worker in Layer:${layerId}. Please decrease #workers or increase #neurons.")
    assert(model_this_layer.length == inputEndIndex - inputStartIndex,
      s"Shape mismatch when backward in layer ${layerId}: ${model_this_layer.length} * ${model_this_layer(0).length} " +
        s"!= ${inputEndIndex - inputStartIndex} * ${neuronSize}")

    // note: Here: W_{h_{i-1}, h_i}^T * X_{h_{i-1}} = Y_{h_i}
    // dl / dx = W * (dl / dy), dl / dw = (X * (dl / dy)^T)
    // 1. first compute outgrad, 2. update model. Note: the order matters.
    val outputGrad: Array[Array[Double]] = Array.ofDim[Double](lastNeuronSize, miniBatchSize)

    for (id_batch <- 0 until (miniBatchSize)) {
      for (tmp_id <- inputStartIndex until (inputEndIndex)) {
        for (id_neuron <- 0 until (neuronSize)) {
          outputGrad(tmp_id)(id_batch) += model_this_layer(tmp_id - inputStartIndex)(id_neuron) * inputGrad(id_neuron)(id_batch)
        }
      }
    }

    // update model
    val lastLayerInput: Array[Array[Double]] = intermediateResult(layerId - 1)
    // update model
    for (id_batch <- 0 until (miniBatchSize)) {
      for (tmp_id <- inputStartIndex until (inputEndIndex)) {
        for (id_neuron <- 0 until (neuronSize)) {
          model_this_layer(tmp_id - inputStartIndex)(id_neuron) -= stepSize * lastLayerInput(tmp_id)(id_batch) * inputGrad(id_neuron)(id_batch)
        }
      }
    }

    outputGrad
  }

  def updateModelFirstLayer(model: Array[Array[Array[Double]]], inputGrad: Array[Array[Double]],
                            arrayWorkSet: ArrayWorkSet[WorkSet], seed: Int, layerId: Int): Unit = {

    val rand = new Random(seed)
    val model_this_layer: Array[Array[Double]] = model(layerId - 1)
    val neuronSize = layers(layerId)

    // update model
    for (id_batch <- 0 until (miniBatchSize)) {
      val tmp_data_point = arrayWorkSet.getRandomLabeledPartDataPoint(rand)
      tmp_data_point.features match {
        case sp: SparseVector => {
          val indices = sp.indices
          val values = sp.values
          for (id_neuron <- 0 until (neuronSize)) {
            // update model_this_layer(indices(idx))(id_neuron)
            for (iid <- 0 until (indices.length)) {
              model_this_layer(indices(iid))(id_neuron) -= stepSize * inputGrad(id_neuron)(id_batch) * values(iid)
            }
          }
        }
        case dp: DenseVector => {
          throw new ColumnMLDenseVectorException
        }
      }
    }
  }

  def miniBatchSGD(): Unit = {
    val start_loading = System.currentTimeMillis()
    val modelRDD: RDD[(ArrayWorkSet[WorkSet], Array[Array[Array[Double]]], Array[Array[Array[Double]]])] = generateModel(inputRDD)
    modelRDD.cache()
    modelRDD.setName("modelRDD")

    // collect labels to the driver, orangized as grouped labels
    val tmp: Array[(Int, Array[Double])] = modelRDD.mapPartitionsWithIndex(
      (pid, iter) => {
        val arrayWorkSet: ArrayWorkSet[WorkSet] = iter.next._1
        val numWorkSets = arrayWorkSet.length()
        val worker_partition_num = numWorkSets / numPartitions + 1
        val start = worker_partition_num * pid
        val end = math.min(worker_partition_num + start, numWorkSets)
        if (end > start) {
          val partitionIds: Array[Int] = new Array[Int](end - start)
          for (id <- 0 until (partitionIds.length)) {
            partitionIds(id) = id + start
          }
          arrayWorkSet.getLabels(partitionIds).toIterator
        }
        else
          Array.empty[(Int, Array[Double])].toIterator
      }
    ).collect()

    val labels_tmp: Array[Array[Double]] = new Array[Array[Double]](tmp.length)
    for (id <- 0 until (labels_tmp.length)) {
      labels_tmp(tmp(id)._1) = tmp(id)._2
    }
    val labels = new ArrayLabels[Double](labels_tmp)

    logInfo(s"ghand=loading:${(System.currentTimeMillis() - start_loading) / 1000.0}")

    var start_time = System.currentTimeMillis()
    var iter_id: Int = 0
    var cur_seed = 0

    while (iter_id < numIterations) {
      start_time = System.currentTimeMillis()
      // broadcast from coefficients last iteration
      cur_seed = 42 + iter_id

      var tmpBCInterResult: Broadcast[Array[Array[Double]]] = null
      var tmpInterResult: Array[Array[Double]] = null

      // forward pass: the input layer is sparse libsvm, x` are dense vectors

      // first layer
      tmpInterResult = modelRDD.mapPartitions(iter => {
        val first_ele = iter.next()
        val arrayWorkSet: ArrayWorkSet[WorkSet] = first_ele._1
        val model: Array[Array[Array[Double]]] = first_ele._2
        val result: Array[Array[Double]] = computeInterResultsFirstLayer(model, arrayWorkSet, cur_seed, 1)
        Iterator(result)
      }).reduce(aggregateResult)

      for (layerId <- 2 until (layers.length)) {
        sigmoid(tmpInterResult)
        assert(tmpInterResult.length == layers(layerId - 1) && tmpInterResult(0).length == miniBatchSize,
          s"Shape Mismatch")
        tmpBCInterResult = modelRDD.sparkContext.broadcast(tmpInterResult)

        tmpInterResult = modelRDD.mapPartitions(iter => {
          val first_ele = iter.next()
          val model: Array[Array[Array[Double]]] = first_ele._2
          val intermediateResults: Array[Array[Array[Double]]] = first_ele._3
          // we are computing layer[layerId], and the input is from layer[layerId - 1]. In NN, this is called featuremap.
          deepCopy2DimArray(tmpBCInterResult.value, intermediateResults(layerId - 1))
          val lastLayerInput: Array[Array[Double]] = intermediateResults(layerId - 1)

          val result: Array[Array[Double]] = computeInterResultsLaterLayers(model, lastLayerInput, layerId, TaskContext.getPartitionId())
          Iterator(result)
        }).reduce(aggregateResult)

        tmpBCInterResult.destroy()
      }

      // note that, we do not apply sigmoid for the last layer output.
      softmax(tmpInterResult)
      val batchLoss = computeBatchLoss(tmpInterResult, labels, miniBatchSize, cur_seed)
      logInfo(s"ghandBatchLoss:${batchLoss}")

      // backward pass
      // for each layer h_i, broadcast dl/dh_i, collect dl/dh_{i-1}
      var layerId = layers.length - 1
      while (layerId > 1) {
        // deal with layers except the first layer
        if (layerId == layers.length - 1) { // last layer
          diffSoftmax(tmpInterResult, labels, cur_seed)
        }
        else {
          diffSigmoid(tmpInterResult)
        }
        tmpBCInterResult = modelRDD.sparkContext.broadcast(tmpInterResult)
        // update model and collect grads of the front layer
        tmpInterResult = modelRDD.mapPartitions(iter => {
          val first_ele = iter.next()
          val model: Array[Array[Array[Double]]] = first_ele._2
          val intermediateResults = first_ele._3
          //          val model_this_layer = model(layerId - 1)
          val inputGrad = tmpBCInterResult.value
          // update model
          val outGrad: Array[Array[Double]] = updateModelAndCollectGradLaterLayers(model, inputGrad,
            intermediateResults, layerId, TaskContext.getPartitionId())

          Iterator(outGrad)
        }).reduce(aggregateResult)
        tmpBCInterResult.destroy()

        layerId -= 1
      }

      // deal with the first layer. not need to collect dl/dh_{i-1}
      diffSigmoid(tmpInterResult)
      tmpBCInterResult = modelRDD.sparkContext.broadcast(tmpInterResult)
      modelRDD.mapPartitions(iter => {
        val first_ele = iter.next()
        val arrayWorkSet = first_ele._1
        val model: Array[Array[Array[Double]]] = first_ele._2
        //          val model_this_layer = model(layerId - 1)
        val inputGrad = tmpBCInterResult.value
        // update model
        updateModelFirstLayer(model, inputGrad, arrayWorkSet, cur_seed, layerId)
        Iterator(-1)
      }).count()

      tmpBCInterResult.destroy()

      logInfo(s"ghandFP=DriverTime=trainTime:" +
        s"${(System.currentTimeMillis() - start_time) / 1000.0}")
      iter_id += 1
    }

  }

  def aggregateResult(array1: Array[Array[Double]], array2: Array[Array[Double]]): Array[Array[Double]] = {
    assert(array1.length == array2.length)
    var k: Int = 0
    while (k < array1.length) {
      var i = 0
      while (i < array1(0).length) {
        array1(k)(i) += array2(k)(i)
        i += 1
      }
      k += 1
    }
    array1
  }


  /**
    * [executed on driver] compute the batch loss using the lastLayer results gathered from executors.
    *
    * @param lastLayerResults : #classes * batch size. The softmax-ed input.
    * @param labels
    * @param seed
    * @return
    */
  def computeBatchLoss(lastLayerResults: Array[Array[Double]], labels: ArrayLabels[Double],
                       seed: Int): Double = {
    val rand = new Random(seed)
    var batchLoss: Double = 0
    assert(lastLayerResults.length == layers(layers.length - 1),
      s"Shape mismatch: The output #neuron:${lastLayerResults.length} != #Classes:${layers(layers.length - 1)}")
    for (id_batch <- 0 until miniBatchSize) {
      val tmp_label = labels.getRandomLabel(rand).toInt
      batchLoss += -math.log(lastLayerResults(tmp_label)(id_batch))
    }
    batchLoss / miniBatchSize
  }

  /**
    * apply softmax to the given array
    *
    * @param lastLayerResults : #neuron * batchsize
    */
  def softmax(lastLayerResults: Array[Array[Double]]): Unit = {
    val neuronSize = lastLayerResults.length
    assert(neuronSize == numClasses, s"numClass mismatch")
    val tmpBatchSize = lastLayerResults(0).length
    assert(tmpBatchSize == miniBatchSize, s"batch size mismatch")
    val norm: Array[Double] = new Array[Double](tmpBatchSize)

    for (id_neuron <- 0 until neuronSize) {
      for (id_batch <- 0 until tmpBatchSize) {
        norm(id_batch) += math.exp(lastLayerResults(id_neuron)(id_batch))
      }
    }

    for (id_neuron <- 0 until neuronSize) {
      for (id_batch <- 0 until tmpBatchSize) {
        lastLayerResults(id_neuron)(id_batch) =
          math.exp(lastLayerResults(id_neuron)(id_batch)) / norm(id_batch)
      }
    }

  }

  def diffSoftmax(softmaxedResult: Array[Array[Double]], labels: ArrayLabels[Double], seed: Int): Unit = {
    val rand = new Random(seed)
    for (id_batch <- 0 until softmaxedResult(0).length) {
      val tmp_label = labels.getRandomLabel(rand).toInt
      softmaxedResult(tmp_label)(id_batch) -= 1.0
    }
  }

  def sigmoid(input: Array[Array[Double]]): Unit = {
    for (i <- 0 until (input.length)) {
      for (j <- 0 until (input(0).length)) {
        input(i)(j) = 1.0 / (1 + Math.exp(-input(i)(j)))
      }
    }
  }

  /**
    *
    * @param input : neuronSize * batchsize
    *              it is already sigmoid-ed.
    */
  def diffSigmoid(input: Array[Array[Double]]): Unit = {
    for (i <- 0 until (input.length)) {
      for (j <- 0 until (input(0).length)) {
        // diff(sigmoid(x)) / partial(x) = (1-sigmoid(x)) * sigmoid(x)
        input(i)(j) = input(i)(j) * (1 - input(i)(j))
      }
    }
  }

  def deepCopy2DimArray(srcArray: Array[Array[Double]], dstArray: Array[Array[Double]]): Unit = {
    assert(srcArray.length == dstArray.length && srcArray(0).length == dstArray(0).length)
    for (i <- 0 until (srcArray.length)) {
      System.arraycopy(srcArray(i), 0, dstArray(i), 0, srcArray(i).length)
    }
  }
}
