package pku.mllibFP.classfication

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import pku.mllibFP.util.IndexedDataPoint
import org.apache.spark.internal.Logging
import org.apache.spark.mllib.linalg.Vector
import org.spark_project.dmg.pmml.Coefficient

import scala.reflect.ClassTag
import scala.util.Random

/**
  * @param inputRDD      : dataRDD, each partition contains only one element, Array[IndexedDataPoint]
  * @param labels        : labels for all datasets
  * @param numFeatures
  * @param numPartitions : number of partitions for the model, e.g., number of tasks per stage
  * @param regParam
  * @param stepSize      : step size for batch
  * @param numIterations
  * @param miniBatchSize
  */
abstract class BaseFPModel[T: ClassTag](@transient inputRDD: RDD[Array[IndexedDataPoint]],
                                        @transient labels: Array[Double],
                                        numFeatures: Int,
                                        numPartitions: Int,
                                        regParam: Double,
                                        stepSize: Double,
                                        numIterations: Int,
                                        miniBatchSize: Int) extends Serializable with Logging {

  def coefficients: Array[T] // T could be double, could be another Array, abstract member.
  /**
    * @param inputRDD
    * @return return modelRDD, an combination of data and model, the model could be Array[Double],
    *         or two-dimensional array.
    */
  def generateModel(inputRDD: RDD[Array[IndexedDataPoint]]): RDD[(Array[IndexedDataPoint], Array[T])]

  /**
    * use the intermediate results (reduced dot products) to compute the coefficients
    * @param dot_products
    * @param seed
    * @return loss, also the coefficients are stored in ${coefficients}
    */
  def computeCoefficients(dot_products: Array[T], seed: Int): Double

  /**
    * combine update model and compute intermediate results (dot products) as one action, to reduce the scheduler delay.
    * @param modelRDD
    * @param bcCoefficients
    * @param lastSeed
    * @param newSeed
    * @return "dotProduct"(not only dot product, but also other reduced results like S_f, G_f in factorization machines)
    */
  def updateModelAndComputeInterResults(modelRDD: RDD[(Array[IndexedDataPoint], Array[T])],
                                      bcCoefficients: Broadcast[Array[T]],
                                      lastSeed: Int, newSeed: Int): Array[T] = {
    modelRDD.mapPartitions(
      iter => {
        val first_ele = iter.next()
        val data_points: Array[IndexedDataPoint] = first_ele._1
        val model: Array[T] = first_ele._2

        // update model first
        var worker_start_time = System.currentTimeMillis()
        updateL2Regu(model, regParam)
        updateModel(model, data_points, bcCoefficients.value, lastSeed)
        logInfo(s"ghandFP=WorkerTime=updateModel:${(System.currentTimeMillis() - worker_start_time) / 1000.0}")

        // compute dot product
        worker_start_time = System.currentTimeMillis()
        val results: Array[T] = computeInterResults(model, data_points, newSeed)
        logInfo(s"ghandFP=WorkerTime=BatchDotProduct:${(System.currentTimeMillis() - worker_start_time) / 1000.0}")

        Iterator(results)

      }
    ).reduce(aggregateResult)
  }

  /**
    * compute the intermediate results to be aggregated to the driver, which could be
    * dot product for linear models, (w, V) for factorization machine, (w1, ..., wk) for MLR.
    * @param model
    * @param data_points
    * @param new_seed
    * @return
    */
  def computeInterResults(model: Array[T], data_points: Array[IndexedDataPoint], new_seed: Int): Array[T]

  /**
    * update model using the coefficients and sampled data from last_seed
    * @param model
    * @param data_points
    * @param coefficients
    * @param last_seed
    */
  def updateModel(model: Array[T], data_points: Array[IndexedDataPoint],
                  coefficients: Array[T], last_seed: Int): Unit = {
    val rand = new Random(last_seed)
    var id_batch = 0
    var id_global = 0
    val num_data_points = data_points.length
    while (id_batch < miniBatchSize) {
      id_global = rand.nextInt(num_data_points)
      updateModelViaOneData(model, data_points(id_global).features, coefficients, id_batch)

      id_batch += 1
    }

  }

  /**
    * update the model using one data point
    * @param model
    * @param features
    * @param local_coefficient
    * @param id_batch
    */
  def updateModelViaOneData(model: Array[T], features: Vector, local_coefficient: Array[T], id_batch: Int): Unit

  def updateL2Regu(model: Array[T], regParam: Double): Unit

  /**
    * aggregate intermediate results, which could be sum arrays in machine learning workloads
    * like LR, SVM, FM, MLR, etc.
    * @param array1
    * @param array2
    * @return
    */
  def aggregateResult(array1: Array[T], array2: Array[T]): Array[T]

  def miniBatchSGD(): Unit = {
    val modelRDD: RDD[(Array[IndexedDataPoint], Array[T])] = generateModel(inputRDD)
    modelRDD.cache()
    modelRDD.setName("modelRDD")
    modelRDD.count()

    var start_time = System.currentTimeMillis()
    var iter_id: Int = 0
    var last_seed = 0
    var cur_seed = 0
    while (iter_id < numIterations) {
      start_time = System.currentTimeMillis()
      // broadcast from coefficients last iteration
      val bcCoefficients = modelRDD.sparkContext.broadcast(coefficients)
      cur_seed = 42 + iter_id
      last_seed = cur_seed - 1
      val dot_products: Array[T] = updateModelAndComputeInterResults(modelRDD, bcCoefficients, last_seed, cur_seed)

      logInfo(s"ghandFP=DriverTime=ComputeDotProductAndUpdateModelTime(reduce dotProducts included):" +
        s" ${(System.currentTimeMillis() - start_time) / 1000.0}")
      start_time = System.currentTimeMillis()

      val loss: Double = computeCoefficients(dot_products, cur_seed)

      logInfo(s"ghandFP=LOSS=BatchLoss:${loss}")
      logInfo(s"ghandFP=DriverTime=ComputeCoefficients(no communication, only in driver): " +
        s"${(System.currentTimeMillis() - start_time) / 1000.0}")

      start_time = System.currentTimeMillis()

      iter_id += 1
      bcCoefficients.destroy()
    }

  }


  def miniBatchLBFGS(modelName: String): Unit = {

  }

}