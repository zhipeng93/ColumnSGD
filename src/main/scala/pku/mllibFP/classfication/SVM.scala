package pku.mllibFP.classfication

import pku.mllibFP.util.IndexedDataPoint
import org.apache.spark.rdd.RDD

import scala.util.Random

class SVM(@transient inputRDD: RDD[Array[IndexedDataPoint]],
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
    val step_size_per_data_point = stepSize / miniBatchSize
    while (id_batch < miniBatchSize) {
      id_global = rand.nextInt(num_data_points)
      val label_scaled = 2 * labels(id_global) - 1
      if ((label_scaled * dot_products(id_batch)) < 1) {
        coefficients(id_batch) = (0.0 - label_scaled) * step_size_per_data_point

        batch_loss += 1 - label_scaled * dot_products(id_batch) // max(0, 1-(2y-1)wx)
      }
      else {
        coefficients(id_batch) = 0.0
      }

      id_batch += 1
    }
    batch_loss / miniBatchSize
  }
}