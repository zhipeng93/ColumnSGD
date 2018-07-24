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
      val label_scaled = 2 * labels(id_global) - 1

      coefficients(id_batch) = -label_scaled /
        (1 + math.exp(label_scaled * dot_products(id_batch)))

      // The following is equivalent to log(1 + exp(margin)) but more numerically stable.
      batch_loss += MLUtils.log1pExp(-label_scaled * dot_products(id_batch))

      id_batch += 1
    }
    batch_loss / miniBatchSize
  }
}