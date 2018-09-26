package pku.mllibFP.classfication

import org.apache.spark.mllib.linalg.{DenseVector, SparseVector}
import pku.mllibFP.util._
import org.apache.spark.rdd.RDD

import scala.util.Random

class SVM(@transient inputRDD: RDD[ArrayWorkSet[WorkSet]],
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
    inputRDD.mapPartitions{
      iter => {
        val model: Array[Array[Double]] = Array.ofDim[Double](1, numFeatures / numPartitions + 1)
        Iterator((iter.next(), model))
      }
    }
  }

  override def computeBatchLoss(interResults: Array[Array[Double]], labels: ArrayLabels[Double],
                                batchSize: Int, seed: Int): Double = {
    val rand = new Random(seed)
    var batchLoss: Double = 0

    for(id_batch <- 0 until batchSize){
      val label_scaled = 2 * labels.getRandomLabel(rand) - 1
      val margin = label_scaled * interResults(0)(id_batch)
      if(margin < 1)
        batchLoss += 1 - margin
    }
    batchLoss / batchSize
  }


  override def computeInterResults(model: Array[Array[Double]], arrayWorkSet: ArrayWorkSet[WorkSet],
                                   batchSize: Int, new_seed: Int): Array[Array[Double]] = {
    val result: Array[Array[Double]] = Array.ofDim[Double](1, batchSize)
    val rand = new Random(new_seed)
    for(id_batch <- 0 until batchSize){
      arrayWorkSet.getRandomLabeledPartDataPoint(rand).features match {
        case sp: SparseVector => {
          val indices = sp.indices
          val values = sp.values
          for(idx <- 0 until indices.length){
            result(0)(id_batch) += values(idx) * model(0)(indices(idx))
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
                           interResults: Array[Array[Double]],
                           batchSize: Int, last_seed: Int, iterationId: Int): Unit ={
    val rand = new Random(last_seed)
//    val gradient: Array[Double] = new Array[Double](model(0).length) // dimension of the local model.

    for(id_batch <- 0 until batchSize){
      val tmp_data_point = arrayWorkSet.getRandomLabeledPartDataPoint(rand)
      val label_scaled = 2 * tmp_data_point.label - 1
      val margin = label_scaled * interResults(0){id_batch}
      if(margin < 1){
        // update model
        val coeff = -label_scaled
        tmp_data_point.features match{
          case sp: SparseVector => {
            val indices = sp.indices
            val values = sp.values
            for(iid <- 0 until(indices.length)){
//              gradient(indices(iid)) += coeff * values(iid)
              model(0)(indices(iid)) -= stepSize / batchSize * coeff * values(iid)
            }

          }
          case dp: DenseVector => {
            throw new ColumnMLDenseVectorException
          }
        }
      }
      else{
        // do nothing
      }
    }
//    for(iid <- 0 until(model(0).length)){
//      model(0)(iid) -= stepSize * gradient(iid) / batchSize
//    }

  }

}
