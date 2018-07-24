package pku.mllibFP.classfication

import pku.mllibFP.util.{IndexedDataPoint, MLUtils}
import org.apache.spark.rdd.RDD
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.mllib.linalg.{DenseVector, SparseVector, Vector}
import org.apache.spark.SparkException


import scala.util.Random

/**
  * @param inputRDD
  * @param labels
  * @param numFeatures
  * @param numPartitions
  * @param regParam
  * @param stepSize
  * @param numIterations
  * @param miniBatchSize
  * @param modelK : dimension of embeddings. This class can only be used for binary classification.
  */
class FM(@transient inputRDD: RDD[Array[IndexedDataPoint]],
         @transient labels: Array[Double],
         numFeatures: Int,
         numPartitions: Int,
         regParam: Double,
         stepSize: Double,
         numIterations: Int,
         miniBatchSize: Int,
         modelK: Int) extends DimKFPModel(inputRDD, labels, numFeatures, numPartitions,
  regParam, stepSize, numIterations, miniBatchSize, modelK) {

  // size of coefficients
  // first line is the coefficient for w
  // the next k lines S_f on a batch of data points.
  override val coefficients = Array.ofDim[Double](modelK + 1, miniBatchSize) //  miniBatchSize * modelK

  /**
    * @param inputRDD
    * @return dataRDD integrated with modelRDD with locality preserved, here model is {1 + k \times localFeatureNum}
    *         the first array is w, the rest k arrays are V. [Note that V cannot be initialized as all zeros.]
    */
  override def generateModel(inputRDD: RDD[Array[IndexedDataPoint]]): RDD[(Array[IndexedDataPoint], Array[Array[Double]])] = {
    val feature_num_per_partition = numFeatures / numPartitions + 1
    inputRDD.mapPartitions(
      iter => {
        val data_points: Array[IndexedDataPoint] = iter.next()
        val init_model = Array.ofDim[Double](modelK + 1, feature_num_per_partition)
        // initialize model V.
        var i = 1
        while (i < init_model.length) {
          var j = 0
          while (j < init_model(1).length) {
            init_model(i)(j) = 1.0 / modelK.toDouble
            j += 1
          }
          i += 1
        }
        Iterator((data_points, init_model))
      }
    )
  }

  /**
    * @param dot_products . The first array is w*x. The next K arrays is S_f = \sum_{i=1}^{n} v_{i,f} * x_i.
    *                     The last K arrays is G_f = \sum_{i=1}^{n}v_{i,f}^2 * x_i^2
    * @param seed
    * @return loss, also the coefficients are stored in ${coefficients}
    */
  override def computeCoefficients(dot_products: Array[Array[Double]], seed: Int): Double = {

    var batch_loss = 0.0
    // store f(x) of factorization machine in the first array, i.e., dot_product(0)
    val fx: Array[Double] = dot_products(0)
    var i = 1
    // process S_f
    while (i <= modelK) {
      var idx = 0
      while (idx < miniBatchSize) {
        fx(idx) += math.pow(dot_products(i)(idx), 2) / 2
        idx += 1
      }
      i += 1
    }
    // process G_f
    i = modelK + 1
    while (i <= modelK * 2) {
      var idx = 0
      while (idx < miniBatchSize) {
        fx(idx) += dot_products(i)(idx) / 2
        idx += 1
      }
      i += 1
    }

    val num_data_points = labels.length
    // compute coefficients and loss
    val rand = new Random(seed)
    var id_batch = 0
    var label_scaled = 0.0

    while (id_batch < miniBatchSize) {
      label_scaled = 2 * labels(rand.nextInt(num_data_points)) - 1
      batch_loss += MLUtils.log1pExp(-label_scaled * fx(id_batch))
      // coefficients for w.
      coefficients(0)(id_batch) = -label_scaled / (1 + math.exp(label_scaled * fx(id_batch)))

      id_batch += 1
    }

    // copy S_f to coeffcients
    i = 1
    while (i <= modelK) {
      var j = 0
      while (j < miniBatchSize) {
        coefficients(i)(j) = dot_products(i)(j)
        j += 1
      }
      i += 1
    }

    batch_loss / miniBatchSize
  }


  override def computeInterResults(model: Array[Array[Double]], data_points: Array[IndexedDataPoint],
                                 new_seed: Int): Array[Array[Double]] = {

    val result: Array[Array[Double]] = Array.ofDim[Double](modelK * 2 + 1, miniBatchSize)
    val rand = new Random(new_seed)
    var id_batch = 0
    var id_global = 0
    val num_data_points = data_points.length
    while (id_batch < miniBatchSize) {
      id_global = rand.nextInt(num_data_points)
      computeInterResultsOneData(model, data_points(id_global).features, result, id_batch)
      id_batch += 1
    }

    result
  }


  def computeInterResultsOneData(model: Array[Array[Double]], features: Vector,
                            results: Array[Array[Double]], id_batch: Int): Unit = {
    features match {
      case sp: SparseVector => {
        val index: Array[Int] = sp.indices
        val values: Array[Double] = sp.values
        // compute dot product: wx
        var i = 0
        while (i < index.size) {
          results(0)(id_batch) += model(0)(index(i)) * values(i)
          i += 1
        }

        // compute S_f, G_f
        var k = 1
        while (k <= modelK) {
          i = 0
          while (i < index.size) {
            val tmp = model(k)(index(i)) * values(i)
            results(k)(id_batch) += tmp // S_f
            results(k + modelK)(id_batch) += tmp * tmp // G_f

            i += 1
          }
          k += 1
        }

      }
      case dp: DenseVector => {
        throw new SparkException("Currently We do not support denseVecor")
      }

    }
    null
  }

  /**
    * @param model
    * @param features
    * @param local_coefficient
    */
  override def updateModelViaOneData(model: Array[Array[Double]], features: Vector,
                                     local_coefficient: Array[Array[Double]], id_batch: Int): Unit = {
    val step_size_per_data_point = stepSize / miniBatchSize
    features match {
      case sp: SparseVector => {
        val index: Array[Int] = sp.indices
        val values: Array[Double] = sp.values
        // sp.size = dimension of the whole vector,
        // index.size = nnz
        updateL2Regu(model, regParam) // may need to change due to different regularization terms

        // update w
        val w = model(0)
        val w_coeff = local_coefficient(0)(id_batch)
        var i = 0
        while (i < index.length) {
          w(index(i)) -= step_size_per_data_point * w_coeff * values(i)
          i += 1
        }

        // update V
        var k = 1
        while (k <= modelK) {
          val kx = local_coefficient(k)(id_batch)
          i = 0
          while (i < index.length) {
            model(k)(index(i)) -= step_size_per_data_point * w_coeff *
              (values(i) * kx - model(k)(index(i)) * math.pow(values(i), 2))

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

}