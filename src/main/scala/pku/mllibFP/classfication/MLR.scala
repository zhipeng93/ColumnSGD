package pku.mllibFP.classfication

import org.apache.spark.mllib.linalg.{DenseVector, SparseVector, Vector}
import pku.mllibFP.util.{ColumnMLDenseVectorException, LabeledPartDataPoint, MLUtils, WorkSet}
import org.apache.spark.rdd.RDD

import scala.util.Random

class MLR(@transient inputRDD: RDD[WorkSet],
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

  override def generateModel(inputRDD: RDD[WorkSet]): RDD[(WorkSet,
    Array[Array[Double]])] = {
    // generate model
    inputRDD.mapPartitions{
      iter => {
        val model: Array[Array[Double]] = Array.ofDim[Double](modelK, numFeatures / numPartitions + 1)
        Iterator((iter.next(), model))
      }
    }
  }


  override def computeInterResults(model: Array[Array[Double]], workSet: WorkSet,
                                   batchSize: Int, new_seed: Int): Array[Array[Double]] = {
    val result: Array[Array[Double]] = Array.ofDim[Double](modelK, batchSize)
    val rand = new Random(new_seed)
    val num_data_points = workSet.getNumDataPoints()
    for(id_batch <- 0 until batchSize){
      val id_global = rand.nextInt(num_data_points)
      workSet.getLabeledPartDataPoint(id_global).features match {
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

  override def computeBatchLoss(interResults: Array[Array[Double]], labels: Array[Double],
                                batchSize: Int, seed: Int): Double = {
    val rand = new Random(seed)
    var batchLoss: Double = 0
    val num_data_points = labels.length
    val norm: Array[Double] = new Array[Double](batchSize)
    for(id_model <- 0 until modelK){
      for(id_batch <- 0 until batchSize){
        norm(id_batch) += math.exp(interResults(id_model)(id_batch))
      }
    }

    for(id_batch <- 0 until batchSize){
      val id_global = rand.nextInt(num_data_points)
      batchLoss += - math.log(math.exp(interResults(labels(id_global).toInt)(id_batch)) / norm(id_batch))
    }
    batchLoss / batchSize
  }


  override def updateModel(model: Array[Array[Double]], workSet: WorkSet, interResults: Array[Array[Double]],
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
    val num_data_points = workSet.getNumDataPoints()
    for(id_batch <- 0 until batchSize){
      val id_global = rand.nextInt(num_data_points)
      val tmp_data_point = workSet.getLabeledPartDataPoint(id_global)
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
