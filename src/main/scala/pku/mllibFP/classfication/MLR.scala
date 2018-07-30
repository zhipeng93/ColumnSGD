package pku.mllibFP.classfication

import org.apache.spark.SparkException
import org.apache.spark.mllib.linalg.{DenseVector, SparseVector, Vector}
import pku.mllibFP.util.IndexedDataPoint
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

  override val coefficients = Array.ofDim[Double](modelK, miniBatchSize)

  /**
    * @param inputRDD
    * @return dataRDD integrated with modelRDD with locality preserved, here model is {k \times localFeatureNum}
    */
  override def generateModel(inputRDD: RDD[Array[IndexedDataPoint]]): RDD[(Array[IndexedDataPoint], Array[Array[Double]])] = {
    val feature_num_per_partition = numFeatures / numPartitions + 1
    inputRDD.mapPartitions(
      iter => {
        val data_points: Array[IndexedDataPoint] = iter.next()

        Iterator((data_points, Array.ofDim[Double](modelK, feature_num_per_partition)))
      }
    )
  }

  override def updateModelViaOneData(model: Array[Array[Double]], features: Vector,
                                     local_coefficient: Array[Array[Double]], id_batch: Int): Unit = {
    features match {
      case sp: SparseVector => {
        val index: Array[Int] = sp.indices
        val values: Array[Double] = sp.values
        // sp.size = dimension of the whole vector,
        // index.size = nnz
        var k = 0
        val step_size_per_data_point = stepSize / miniBatchSize
        while(k < modelK) {
          var i = 0
          val kth_coeff = local_coefficient(k)(id_batch)
          while (i < index.size) {
            model(k)(index(i)) -= values(i) * step_size_per_data_point * kth_coeff
            i += 1
          }
          k += 1
        }

      }
      case dp: DenseVector => {
        throw new SparkException("Currently we do not support denseVecor")
      }
    }
  }

  /**
    *
    * @param dot_products modelK * miniBatchSize
    * @param seed seed to generate the samples
    * @return loss, also the coefficients are stored in ${coefficients}
    */
  override def computeCoefficients(dot_products: Array[Array[Double]], seed: Int): Double = {

    val num_data_points = labels.length
    // softmax for dot products
    val normalization: Array[Double] = new Array[Double](miniBatchSize)
    var k = 0
    while(k < modelK){
      var i = 0
      while(i < miniBatchSize){
        coefficients(k)(i) = math.exp(dot_products(k)(i))
        normalization(i) += coefficients(k)(i)
        i += 1
      }
      k += 1
    }
    k = 0
    while(k < modelK){
      var i = 0
      while(i < miniBatchSize){
        coefficients(k)(i) /= normalization(i)  // this is negative coefficients for convenience
        i += 1
      }
      k += 1
    }

    // compute loss and update coefficients again
    val rand = new Random(seed)
    var batch_loss: Double = 0
    var id_batch = 0
    var id_global = 0
    while(id_batch < miniBatchSize){
      id_global = rand.nextInt(num_data_points)
      val label: Int = labels(id_global).toInt // labels start from 0, follows 0, 1, 2, ...
      // batchloss = \sum_{n=1, N} \sum_{k=1, K} -t_{nk} * ln(y_nk)
      batch_loss += -math.log(coefficients(label)(id_batch)) // reverse negative coefficients

      coefficients(label)(id_batch) -= 1
      id_batch += 1
    }

    batch_loss / miniBatchSize
  }

  override def computeInterResults(model: Array[Array[Double]], data_points: Array[IndexedDataPoint],
                                 new_seed: Int): Array[Array[Double]] = {

    val result: Array[Array[Double]] = Array.ofDim[Double](modelK, miniBatchSize)
    val rand = new Random(new_seed)
    var id_batch = 0
    var id_global = 0
    val num_data_points = data_points.length
    while(id_batch < miniBatchSize){
      id_global = rand.nextInt(num_data_points)
      computeInterResultsOneData(model, data_points(id_global).features, result, id_batch)
      id_batch += 1
    }

    result
  }

  def computeInterResultsOneData(model: Array[Array[Double]], features: Vector,
                            result: Array[Array[Double]], id_batch: Int): Unit = {
    features match {
      case sp: SparseVector => {
        val index: Array[Int] = sp.indices
        val values: Array[Double] = sp.values
        var k = 0
        while(k < modelK){
          var i = 0
          while(i < index.size){
            result(k)(id_batch) += model(k)(index(i)) * values(i)
            i += 1
          }

          k += 1
        }
      }
      case dp: DenseVector => {
        throw new SparkException("Currently We do not support denseVecor")
      }

    }
  }

}