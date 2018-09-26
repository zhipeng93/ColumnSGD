package pku.mllibFP.classfication

import org.apache.spark.mllib.linalg.{DenseVector, SparseVector, Vector}
import pku.mllibFP.util._
import org.apache.spark.rdd.RDD

import scala.util.Random

class MLR(@transient inputRDD: RDD[ArrayWorkSet[WorkSet]],
          numFeatures: Int,
          numPartitions: Int,
          regParam: Double,
          stepSize: Double,
          numIterations: Int,
          miniBatchSize: Int,
          modelK: Int) extends BaseFPModel(inputRDD, numFeatures, numPartitions,
  regParam, stepSize, numIterations, miniBatchSize) {

  override def iniInterResult(): Unit = {
    // initialize intermediate results
    intermediateResults = Array.ofDim[Double](modelK, miniBatchSize)
  }

  override def generateModel(inputRDD: RDD[ArrayWorkSet[WorkSet]]): RDD[(ArrayWorkSet[WorkSet],
    Array[Array[Double]])] = {
    // generate model
    inputRDD.mapPartitions{
      iter => {
        val model: Array[Array[Double]] = Array.ofDim[Double](modelK, numFeatures / numPartitions + 1)
        Iterator((iter.next(), model))
      }
    }
  }


  override def computeInterResults(model: Array[Array[Double]], arrayWorkSet: ArrayWorkSet[WorkSet],
                                   batchSize: Int, new_seed: Int): Array[Array[Double]] = {
    val result: Array[Array[Double]] = Array.ofDim[Double](modelK, batchSize)
    val rand = new Random(new_seed)
    for(id_batch <- 0 until batchSize){
      arrayWorkSet.getRandomLabeledPartDataPoint(rand).features match {
        case sp: SparseVector => {
          val indices = sp.indices
          val values = sp.values
          for(id_model <- 0 until modelK){
            for(idx <- 0 until indices.length) {
              result(id_model)(id_batch) += values(idx) * model(id_model)(indices(idx))
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

  override def computeBatchLoss(interResults: Array[Array[Double]], labels: ArrayLabels[Double],
                                batchSize: Int, seed: Int): Double = {
    val rand = new Random(seed)
    var batchLoss: Double = 0
    val norm: Array[Double] = new Array[Double](batchSize)
    for(id_model <- 0 until modelK){
      for(id_batch <- 0 until batchSize){
        norm(id_batch) += math.exp(interResults(id_model)(id_batch))
      }
    }

    for(id_batch <- 0 until batchSize){
      val tmp_label = labels.getRandomLabel(rand).toInt
      batchLoss += - math.log(math.exp(interResults(tmp_label)(id_batch)) / norm(id_batch))
    }
    batchLoss / batchSize
  }


  override def updateModel(model: Array[Array[Double]], arrayWorkSet: ArrayWorkSet[WorkSet],
                           interResults: Array[Array[Double]],
                           batchSize: Int, last_seed: Int, iterationId: Int): Unit ={
    val rand = new Random(last_seed)
    // calculte the norm
    val norm: Array[Double] = new Array[Double](batchSize)
    for(id_model <- 0 until modelK){
      for(id_batch <- 0 until batchSize){
        norm(id_batch) += math.exp(interResults(id_model)(id_batch))
      }
    }
    // update the model
    for(id_batch <- 0 until batchSize){
      val tmp_data_point = arrayWorkSet.getRandomLabeledPartDataPoint(rand)
      val label = tmp_data_point.label
      // use one data point to update the model (k sub-models)
      tmp_data_point.features match {
        case sp: SparseVector => {
          val indices = sp.indices
          val values = sp.values
          for(id_model <- 0 until modelK){
            var coeff = math.exp(interResults(id_model)(id_batch)) / norm(id_batch)
            if(label == id_model) {
             coeff -= 1
            }
            for(idx <- 0 until indices.length){
              model(id_model)(indices(idx)) -= stepSize / batchSize * coeff * values(idx)
            }
          }
        }
        case dp: DenseVector => {
          throw new ColumnMLDenseVectorException
        }
      }
    }
  }

}
