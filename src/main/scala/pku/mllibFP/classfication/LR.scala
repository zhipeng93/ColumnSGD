package pku.mllibFP.classfication

import pku.mllibFP.util.{IndexedDataPoint, MLUtils}
import org.apache.spark.rdd.RDD

import scala.util.Random

class LR(@transient inputRDD: RDD[Array[IndexedDataPoint]],
         @transient labels: Array[Double],
          numFeatures: Int,
          numPartitions: Int,
          regParam: Double,
          stepSize: Double,
          numIterations: Int,
          miniBatchSize: Int) extends Dim1FPModel(inputRDD, labels, numFeatures, numPartitions,
  regParam, stepSize, numIterations, miniBatchSize) {

  override def computeCoefficients(dot_products: Array[Double], seed: Int): Double = {
    val rand = new Random(seed)
    var batch_loss: Double = 0
    var id_batch = 0
    var id_global = 0
    val num_data_points = labels.length
    while (id_batch < miniBatchSize) {
      id_global = rand.nextInt(num_data_points)
      coefficients(id_batch) =
        (1.0 / (1.0 + math.exp(-dot_products(id_batch)))) - labels(id_global)

      val margin = -1.0 * dot_products(id_batch)
      if (labels(id_global) > 0) {
        // The following is equivalent to log(1 + exp(margin)) but more numerically stable.
        batch_loss += MLUtils.log1pExp(margin)
      } else {
        batch_loss += MLUtils.log1pExp(margin) - margin
      }

      id_batch += 1
    }
    batch_loss / miniBatchSize
  }
}