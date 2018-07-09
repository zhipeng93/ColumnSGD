package pku.mllibFP.classfication

import org.apache.spark.SparkException
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.mllib.linalg.{DenseVector, SparseVector, Vector}
import org.apache.spark.rdd.RDD
import pku.mllibFP.util.IndexedDataPoint

import scala.util.Random

abstract class Dim1FPModel(@transient inputRDD: RDD[Array[IndexedDataPoint]],
                           @transient labels: Array[Double],
                           numFeatures: Int,
                           numPartitions: Int,
                           regParam: Double,
                           stepSize: Double,
                           numIterations: Int,
                           miniBatchSize: Int) extends BaseFPModel[Double](inputRDD, labels, numFeatures, numPartitions,
  regParam, stepSize, numIterations, miniBatchSize) {
  // directly give the parameters to the parent class.

  override val coefficients = new Array[Double](miniBatchSize)

  /**
    * @param inputRDD
    * @return dataRDD integrated with modelRDD with locality preserved
    */
  override def generateModel(inputRDD: RDD[Array[IndexedDataPoint]]): RDD[(Array[IndexedDataPoint], Array[Double])] = {
    val feature_num_per_partition = numFeatures / numPartitions + 1
    inputRDD.mapPartitions(
      iter => {
        val data_points: Array[IndexedDataPoint] = iter.next()
        Iterator((data_points, new Array[Double](feature_num_per_partition)))
      }
    )
  }

  override def computeCoefficients(dot_products: Array[Double], seed: Int): Double

  override def updateModelAndComputeDotProduct(modelRDD: RDD[(Array[IndexedDataPoint], Array[Double])],
                                               bcCoefficients: Broadcast[Array[Double]],
                                               lastSeed: Int, newSeed: Int): Array[Double] = {
    // avoid serialization overhead
    modelRDD.mapPartitions(
      iter => {
        val first_ele = iter.next()
        val data_points: Array[IndexedDataPoint] = first_ele._1
        val model: Array[Double] = first_ele._2

        // update model first
        var worker_start_time = System.currentTimeMillis()
        var rand = new Random(lastSeed)
        val local_coefficients: Array[Double] = bcCoefficients.value
        updateL2Regu(model, regParam)

        var id_batch = 0
        var id_global = 0
        val num_data_points = data_points.length
        while (id_batch < miniBatchSize) {
          id_global = rand.nextInt(num_data_points)
          val custom_stepsize = stepSize / miniBatchSize * local_coefficients(id_batch)
          updateModelViaOneData(model, data_points(id_global).features, custom_stepsize)

          id_batch += 1
        }
        logInfo(s"ghandFP=WorkerTime=updateModel:${(System.currentTimeMillis() - worker_start_time) / 1000.0}")

        // compute dot product
        worker_start_time = System.currentTimeMillis()

        val results: Array[Double] = new Array[Double](miniBatchSize)
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

  def sumArray(array1: Array[Double], array2: Array[Double]): Array[Double] = {
    assert(array1.length == array2.length)
    var k: Int = 0
    while (k < array1.length) {
      array1(k) += array2(k)
      k += 1
    }
    array1
  }

  def partDotProductOneData(model: Array[Double], features: Vector): Double = {
    val partDotProduct: Double = features match {
      case sp: SparseVector => {
        // w * x
        var result: Double = 0.0
        var k = 0
        val index: Array[Int] = sp.indices
        val values: Array[Double] = sp.values

        while (k < index.size) {
          result += model(index(k)) * values(k)
          k += 1
        }
        result
      }
      case dp: DenseVector => {
        throw new SparkException("Currently We do not support denseVecor")
      }

    }
    partDotProduct
  }

  def updateModelViaOneData(model: Array[Double], features: Vector, stepsize_per_sample: Double): Unit = {
    // stepsize_per_sample = stepsize / batchsize * coefficients(i)
    features match {
      case sp: SparseVector => {
        var k = 0
        val index: Array[Int] = sp.indices
        val values: Array[Double] = sp.values
        // sp.size = dimension of the whole vector,
        // index.size = nnz
        while (k < index.size) {
          model(index(k)) -= stepsize_per_sample * values(k)
          k += 1
        }

      }
      case dp: DenseVector => {
        throw new SparkException("Currently we do not support denseVecor")
      }
    }
  }

  def updateL2Regu(model: Array[Double], regParam: Double): Unit = {
    if (regParam == 0)
      return

    var k = 0
    while (k < model.length) {
      model(k) *= (1 - regParam)
      k += 1
    }
  }
}