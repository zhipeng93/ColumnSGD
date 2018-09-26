package pku.mllibFP.classfication

import org.apache.spark.mllib.linalg.{DenseVector, SparseVector}
import pku.mllibFP.util._
import org.apache.spark.rdd.RDD

import scala.util.Random

class FM(@transient inputRDD: RDD[ArrayWorkSet[WorkSet]],
         numFeatures: Int,
         numPartitions: Int,
         regParam: Double,
         stepSize: Double,
         numIterations: Int,
         miniBatchSize: Int,
         modelK: Int) extends BaseFPModel(inputRDD, numFeatures, numPartitions,
  regParam, stepSize, numIterations, miniBatchSize) {

  override def iniInterResult(): Unit = {
    // dotProduct = sum_i (w_i * x_i - 0.5 * \sum_f (v_{if}^2 * x_i^2)), S_f  = V_f * x
    intermediateResults = Array.ofDim[Double](modelK + 1, miniBatchSize)
  }

  override def generateModel(inputRDD: RDD[ArrayWorkSet[WorkSet]]): RDD[(ArrayWorkSet[WorkSet],
    Array[Array[Double]])] = {
    // generate model
    inputRDD.mapPartitions{
      iter => {
        val init_model: Array[Array[Double]] = Array.ofDim[Double](modelK + 1, numFeatures / numPartitions + 1)
        val rand = new Random()
        for(i <- 1 until init_model.length){
          for(j <- 0 until init_model(0).length){
            init_model(i)(j) = rand.nextGaussian() * 0.01 // ~(0, 0.01)
//             init_model(i)(j) = 0.01 // this is purely LR.
          }
        }
        Iterator((iter.next(), init_model))
      }
    }
  }

  override def computeInterResults(model: Array[Array[Double]], arrayWorkSet: ArrayWorkSet[WorkSet],
                                   batchSize: Int, new_seed: Int): Array[Array[Double]] = {
    // first line: w*x, next k line: S_f, next k line: G_f
    val result: Array[Array[Double]] = Array.ofDim[Double](modelK + 1, batchSize)

    val rand = new Random(new_seed)
    for(id_batch <- 0 until batchSize){
      arrayWorkSet.getRandomLabeledPartDataPoint(rand).features match {
        case sp: SparseVector => {
          val indices = sp.indices
          val values = sp.values
          // wx
          for(idx <- 0 until indices.length){
            result(0)(id_batch) += model(0)(indices(idx)) * values(idx)
          }
          // V_f
          for(id_model <- 1 until (modelK + 1)){
            for(idx <- 0 until indices.length) {
              result(id_model)(id_batch) += model(id_model)(indices(idx)) * values(idx) // S_f
              result(0)(id_batch) -= 0.5 * math.pow(model(id_model)(indices(idx)) * values(idx), 2) // G_f
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

    val s2_f: Array[Double] = new Array[Double](batchSize)
    for(id_model <- 1 until (modelK + 1)){
      for(id_batch <- 0 until batchSize){
        s2_f(id_batch) += math.pow(interResults(id_model)(id_batch), 2)
      }
    }

    for(id_batch <- 0 until batchSize){
      val label_scaled = 2 * labels.getRandomLabel(rand) - 1
      batchLoss += MLUtils.log1pExp(-label_scaled * (interResults(0)(id_batch) + 0.5 * s2_f(id_batch)))
    }
    batchLoss / batchSize
  }

  override def updateModel(model: Array[Array[Double]], arrayWorkSet: ArrayWorkSet[WorkSet],
                           interResults: Array[Array[Double]],
                           batchSize: Int, last_seed: Int, iterationId: Int): Unit ={
    val rand = new Random(last_seed)
    // update the model
    for(id_batch <- 0 until batchSize){
      val tmp_data_point = arrayWorkSet.getRandomLabeledPartDataPoint(rand)
      // use one data point to update the model (k sub-models)
      tmp_data_point.features match {
        case sp: SparseVector => {
          val indices = sp.indices
          val values = sp.values
          val label_scaled = 2 * tmp_data_point.label - 1
          var tmp_s2_f = 0.0
          for(id_model <- 1 until (modelK + 1)){
            tmp_s2_f += math.pow(interResults(id_model)(id_batch), 2)
          }
          val tmp_grad = -label_scaled / (1 + math.exp(label_scaled * (interResults(0)(id_batch) + tmp_s2_f * 0.5) ))

          // update w
            for(idx <- 0 until indices.length) {
            model(0)(indices(idx)) -= stepSize / batchSize * tmp_grad * values(idx)
          }
          // update v_f
          for(id_model <- 1 until (modelK + 1)) {
            for(idx <- 0 until indices.length) {
              model(id_model)(indices(idx)) -= stepSize / batchSize * tmp_grad *
                (values(idx) * interResults(id_model)(id_batch) -
                  model(id_model)(indices(idx)) * values(idx) * values(idx))
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