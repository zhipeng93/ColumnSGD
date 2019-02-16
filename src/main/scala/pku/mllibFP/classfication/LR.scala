package pku.mllibFP.classfication

import org.apache.spark.mllib.linalg.{DenseVector, SparseVector}
import pku.mllibFP.util._
import org.apache.spark.rdd.RDD

import scala.util.Random

/**
  * in LR, the model size is 1*m, the intermediate result is 1*m, i.e., the dot product,
  * the loss is logistic loss.
  *
  * @param inputRDD      : dataRDD, each partition contains only one element, Array[IndexedDataPoint]
  * @param numFeatures
  * @param numPartitions : number of partitions for the model, e.g., number of tasks per stage
  * @param regParam
  * @param stepSize      : step size for batch
  * @param numIterations
  * @param miniBatchSize
  */

class LR(@transient inputRDD: RDD[ArrayWorkSet[WorkSet]],
         numFeatures: Int,
         numPartitions: Int,
         regParam: Double,
         stepSize: Double,
         numIterations: Int,
         miniBatchSize: Int) extends BaseFPModel(inputRDD, numFeatures, numPartitions,
  regParam, stepSize, numIterations, miniBatchSize) {

  override def iniInterResult(): Unit = {
    // initialize intermediate results
    intermediateResults = Array.ofDim[Double](1, miniBatchSize)
  }

  override def generateModel(inputRDD: RDD[ArrayWorkSet[WorkSet]]): RDD[(ArrayWorkSet[WorkSet],
    Array[Array[Double]])] = {
    // generate model
    inputRDD.mapPartitions {
      iter => {
        val model: Array[Array[Double]] = Array.ofDim[Double](1, (numFeatures / numPartitions + 1) * 2)
        // duplicate model size for backup computation
        Iterator((iter.next(), model))
      }
    }
  }


  override def computeBatchLoss(interResults: Array[Array[Double]], labels: ArrayLabels[Double],
                                batchSize: Int, seed: Int): Double = {
    val rand = new Random(seed)
    var batchLoss: Double = 0

    for (id_batch <- 0 until batchSize) {
      val label_scaled = 2 * labels.getRandomLabel(rand) - 1
      batchLoss += MLUtils.log1pExp(-label_scaled * interResults(0)(id_batch))
    }
    batchLoss / batchSize
  }


  override def computeInterResults(model: Array[Array[Double]], arrayWorkSet: ArrayWorkSet[WorkSet],
                                   batchSize: Int, new_seed: Int): Array[Array[Double]] = {
    val result: Array[Array[Double]] = Array.ofDim[Double](1, batchSize)
    val rand = new Random(new_seed)
    for (id_batch <- 0 until batchSize) {
      arrayWorkSet.getRandomLabeledPartDataPoint(rand).features match {
        case sp: SparseVector => {
          val indices = sp.indices
          val values = sp.values
          for (idx <- 0 until indices.length) {
            result(0)(id_batch) += values(idx) * model(0)(indices(idx)) / 2
            // due to bachup computation
          }
        }
        case dp: DenseVector => {
          throw new ColumnMLDenseVectorException
        }
      }
    }
    result
  }

  override def updateModel(model: Array[Array[Double]], arrayWorkSet: ArrayWorkSet[WorkSet],
                           interResults: Array[Array[Double]], batchSize: Int,
                           last_seed: Int, iterationId: Int): Unit = {
    val rand = new Random(last_seed)

//    val gradient: Array[Double] = new Array[Double](model(0).length)

    for (id_batch <- 0 until batchSize) {
      val tmp_data_point = arrayWorkSet.getRandomLabeledPartDataPoint(rand)
      val label_scaled = 2 * tmp_data_point.label - 1
      val coeff = -label_scaled / (1 + math.exp(label_scaled * interResults(0)(id_batch)))
      tmp_data_point.features match {
        case sp: SparseVector => {
          val indices = sp.indices
          val values = sp.values
          for(iid <- 0 until(indices.length)){
//            gradient(indices(iid)) += coeff * values(iid)
            model(0)(indices(iid)) -= stepSize / batchSize * coeff * values(iid)
          }
        }
        case dp: DenseVector => {
          throw new ColumnMLDenseVectorException
        }
      }
    }
//    for(iid <- 0 until(model(0).length)){
//      model(0)(iid) -= stepSize * gradient(iid) / batchSize
//    }
  }

}
