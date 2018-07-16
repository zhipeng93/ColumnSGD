package pku.mllibFP.classfication

import org.apache.spark.SparkException
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.mllib.linalg.{DenseVector, SparseVector, Vector}
import org.apache.spark.rdd.RDD
import pku.mllibFP.util.IndexedDataPoint

import scala.util.Random

abstract class DimKFPModel(@transient inputRDD: RDD[Array[IndexedDataPoint]],
                           @transient labels: Array[Double],
                           numFeatures: Int,
                           numPartitions: Int,
                           regParam: Double,
                           stepSize: Double,
                           numIterations: Int,
                           miniBatchSize: Int,
                           modelK: Int) extends BaseFPModel[Array[Double]](inputRDD, labels, numFeatures, numPartitions,
  regParam, stepSize, numIterations, miniBatchSize) {
  // directly give the parameters to the parent class.

  // size of coefficients
  override val coefficients = Array.ofDim[Double](miniBatchSize, modelK) // modelK * miniBatchSize

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

  override def computeCoefficients(dot_products: Array[Array[Double]], seed: Int): Double

  override def updateModelAndComputeDotProduct(modelRDD: RDD[(Array[IndexedDataPoint], Array[Array[Double]])],
                                               bcCoefficients: Broadcast[Array[Array[Double]]],
                                               lastSeed: Int, newSeed: Int): Array[Array[Double]] = {
    // avoid serialization overhead
    modelRDD.mapPartitions(
      iter => {
        val first_ele = iter.next()
        val data_points: Array[IndexedDataPoint] = first_ele._1
        val model: Array[Array[Double]] = first_ele._2

        // update model first
        var worker_start_time = System.currentTimeMillis()
        var rand = new Random(lastSeed)
        val local_coefficients: Array[Array[Double]] = bcCoefficients.value
        updateL2Regu(model, regParam)

        var id_batch = 0
        var id_global = 0
        val num_data_points = data_points.length
        while (id_batch < miniBatchSize) {
          id_global = rand.nextInt(num_data_points)
          updateModelViaOneData(model, data_points(id_global).features, local_coefficients(id_batch))

          id_batch += 1
        }
        logInfo(s"ghandFP=WorkerTime=updateModel:${(System.currentTimeMillis() - worker_start_time) / 1000.0}")

        // compute dot product
        worker_start_time = System.currentTimeMillis()

        val results: Array[Array[Double]] = new Array[Array[Double]](miniBatchSize)
        rand = new Random(newSeed)
        id_batch = 0
        id_global = 0
        while (id_batch < miniBatchSize) {
          id_global = rand.nextInt(num_data_points)
          results(id_batch) = partDotProductOneData(model, data_points(id_global).features)
          id_batch += 1
        }
        logInfo(s"ghandFP=WorkerTime=BatchDotProduct:${(System.currentTimeMillis() - worker_start_time) / 1000.0}")

        Iterator(results)

      }
    ).reduce(sumArray)

  }

  def sumArray(array1: Array[Array[Double]], array2: Array[Array[Double]]): Array[Array[Double]] = {
    assert(array1.length == array2.length)
    var k: Int = 0
    while (k < array1.length) {
      var i = 0
      while(i < array1(0).length){
        array1(k)(i) += array2(k)(i)
        i += 1
      }
      k += 1
    }
    array1
  }

  def partDotProductOneData(model: Array[Array[Double]], features: Vector): Array[Double] = {
    val dot_products: Array[Double] = new Array[Double](modelK)
    features match {
      case sp: SparseVector => {
        val index: Array[Int] = sp.indices
        val values: Array[Double] = sp.values
        var k = 0
        while(k < modelK){
          var i = 0
          while(i < index.size){
            dot_products(k) += model(k)(index(i)) * values(i)
            i += 1
          }

          k += 1
        }
      }
      case dp: DenseVector => {
        throw new SparkException("Currently We do not support denseVecor")
      }

    }
    dot_products
  }

  def updateModelViaOneData(model: Array[Array[Double]], features: Vector, coeff: Array[Double]): Unit = {
    features match {
      case sp: SparseVector => {
        val index: Array[Int] = sp.indices
        val values: Array[Double] = sp.values
        // sp.size = dimension of the whole vector,
        // index.size = nnz
        var k = 0
        while(k < modelK) {
          var i = 0
          while (i < index.size) {
            model(k)(index(i)) -= values(i) * coeff(k)
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

  override def updateL2Regu(model: Array[Array[Double]], regParam: Double): Unit = {
    if (regParam == 0)
      return

    val len1 = model.length
    val len2 = model(0).length
    var i, j =0
    while (i < len1) {
      j = 0
      while (j < len2){
        model(i)(j) *= (1 - regParam)
        j += 1
      }
      i += 1
    }
  }
}