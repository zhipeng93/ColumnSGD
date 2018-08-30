package pku.mllibFP.classfication

import org.apache.spark.SparkEnv
import org.apache.spark.mllib.linalg.{DenseVector, SparseVector}
import pku.mllibFP.util.{ColumnMLDenseVectorException, LabeledPartDataPoint, MLUtils, WorkSet}
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

class AdamLR(@transient inputRDD: RDD[WorkSet],
             numFeatures: Int,
             numPartitions: Int,
             regParam: Double,
             stepSize: Double,
             numIterations: Int,
             miniBatchSize: Int) extends LR(inputRDD, numFeatures, numPartitions,
  regParam, stepSize, numIterations, miniBatchSize) {

  override def generateModel(inputRDD: RDD[WorkSet]): RDD[(WorkSet,
    Array[Array[Double]])] = {
    // generate model
    inputRDD.mapPartitions {
      iter => {
        val model: Array[Array[Double]] = Array.ofDim[Double](3, numFeatures / numPartitions + 1)
        Iterator((iter.next(), model))
      }
    }
  }

  override def updateModel(model: Array[Array[Double]], workSet: WorkSet, interResults: Array[Array[Double]],
                           batchSize: Int, last_seed: Int, iterationId: Int): Unit = {
    val rand = new Random(last_seed)
    val num_data_points = workSet.getNumDataPoints()

    val epsilon = SparkEnv.get.conf.getDouble("spark.ml.sgd.adam.epsilon", 1e-7)
    val beta1 = SparkEnv.get.conf.getDouble("spark.ml.sgd.adam.beta1", 0.9)
    val beta2 = SparkEnv.get.conf.getDouble("spark.ml.sgd.adam.beta2", 0.99)
    val gradient: Array[Double] = new Array[Double](model(0).length)

    for (id_batch <- 0 until batchSize) {
      val id_global = rand.nextInt(num_data_points)
      val tmp_data_point = workSet.getLabeledPartDataPoint(id_global)
      val label_scaled = 2 * tmp_data_point.label - 1
      val coeff = -label_scaled / (1 + math.exp(label_scaled * interResults(0)(id_batch)))
      tmp_data_point.features match {
        case sp: SparseVector => {
          val indices = sp.indices
          val values = sp.values
          for(iid <- 0 until(indices.length)){
            gradient(indices(iid)) += coeff * values(iid)
          }
        }
        case dp: DenseVector => {
          throw new ColumnMLDenseVectorException
        }
      }
    }
    val m_bias = 1 - math.pow(beta1, iterationId + 1)
    val v_bias = 1 - math.pow(beta2, iterationId + 1)
    for(id <- 0 until gradient.length){
      if(gradient(id) != 0) {
        gradient(id) /= batchSize // normalize
        model(1)(id) = beta1 * model(1)(id) + (1 - beta1) * gradient(id) // momentum
        model(2)(id) = beta2 * model(2)(id) + (1 - beta2) * gradient(id) * gradient(id) // v

        model(0)(id) -= stepSize / (math.sqrt(model(2)(id) / v_bias) + epsilon) * model(1)(id) / m_bias
      }
    }
  }

}
