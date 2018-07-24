package pku.mllibFP.classfication

import org.apache.spark.SparkException
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

  override val coefficients = new Array[Double](miniBatchSize) // include the step size.

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

  /**
    * for one-dimensional model, the coefficient for each data set is a [double]
    * @param dot_products
    * @param seed
    * @return loss, also the coefficients are stored in ${coefficients}
    */
  override def computeCoefficients(dot_products: Array[Double], seed: Int): Double


  override def computeInterResults(model: Array[Double], data_points: Array[IndexedDataPoint],
                                 new_seed: Int): Array[Double] = {

    val rand = new Random(new_seed)
    var id_batch = 0
    var id_global = 0
    val num_data_points = data_points.length
    val result: Array[Double] = new Array[Double](miniBatchSize)
    while(id_batch < miniBatchSize){
      id_global = rand.nextInt(num_data_points)
      computeInterResultsOneData(model, data_points(id_global).features, result, id_batch)
      id_batch += 1
    }

    result
  }

  /**
    * for 1-dim model, the dot product is always w*x
    * @param model
    * @param features
    * @return
    */
  def computeInterResultsOneData(model: Array[Double], features: Vector, result: Array[Double], id_batch: Int): Unit = {
    result(id_batch) = features match {
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
  }

  override def updateModelViaOneData(model: Array[Double], features: Vector,
                                     local_coefficient: Array[Double], id_batch: Int): Unit = {

    features match {
      case sp: SparseVector => {
        var k = 0
        val index: Array[Int] = sp.indices
        val values: Array[Double] = sp.values
        // sp.size = dimension of the whole vector,
        // index.size = nnz
        val step_size_per_data_point = stepSize / miniBatchSize
        while (k < index.size) {
          model(index(k)) -= step_size_per_data_point * local_coefficient(id_batch) * values(k)
          k += 1
        }

      }
      case dp: DenseVector => {
        throw new SparkException("Currently we do not support denseVecor")
      }
    }

  }

  override def updateL2Regu(model: Array[Double], regParam: Double): Unit = {
    if (regParam == 0)
      return

    var k = 0
    while (k < model.length) {
      model(k) *= (1 - regParam)
      k += 1
    }
  }

  override def aggregateResult(array1: Array[Double], array2: Array[Double]): Array[Double] = {
    assert(array1.length == array2.length)
    var k: Int = 0
    while (k < array1.length) {
      array1(k) += array2(k)
      k += 1
    }
    array1
  }
}