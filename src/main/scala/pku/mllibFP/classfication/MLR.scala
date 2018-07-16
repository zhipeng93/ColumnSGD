package pku.mllibFP.classfication

import pku.mllibFP.util.{IndexedDataPoint, MLUtils}
import org.apache.spark.rdd.RDD

import scala.util.Random

class MLR(@transient inputRDD: RDD[Array[IndexedDataPoint]],
         @transient labels: Array[Double],
          numFeatures: Int,
          numPartitions: Int,
          regParam: Double,
          stepSize: Double,
          numIterations: Int,
          miniBatchSize: Int,
          modelK: Int) extends DimKFPModel(inputRDD, labels, numFeatures, numPartitions,
  regParam, stepSize, numIterations, miniBatchSize, modelK) {

  override def computeCoefficients(dot_products: Array[Array[Double]], seed: Int): Double = {

    val num_data_points = labels.length
    val step_size_per_data_point = stepSize / miniBatchSize
    // softmax for dot products
    var i = 0
    while(i < miniBatchSize) {
      var k = 0
      var tmp_sum = 0.0
      while(k < modelK) {
        coefficients(i)(k) = math.exp(dot_products(i)(k))
        tmp_sum += coefficients(i)(k)
        k += 1
      }
      k = 0
      while(k < modelK) {
        coefficients(i)(k) = -1.0 * step_size_per_data_point * coefficients(i)(k) / tmp_sum
        k += 1
      }
      i += 1
    }

    val rand = new Random(seed)
    var batch_loss: Double = 0
    var id_batch = 0
    var id_global = 0

    while (id_batch < miniBatchSize) {
      id_global = rand.nextInt(num_data_points)
      val label: Int = labels(id_global).toInt
      // batchloss = \sum_{n=1, N} \sum_{k=1, K} -t_{nk} * ln(y_nk)
      batch_loss += -math.log(coefficients(id_batch)(label) / (-step_size_per_data_point))
      coefficients(id_batch)(label) += -step_size_per_data_point * (-1)

      id_batch += 1
    }

    batch_loss / miniBatchSize
  }
}