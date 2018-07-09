package pku.mllibFP.classfication

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import pku.mllibFP.util.{IndexedDataPoint}
import org.apache.spark.internal.Logging

import scala.reflect.ClassTag

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
    * use the reduced dot products to compute the coefficients
    * @param dot_products
    * @param seed
    * @return loss, also the coefficients are stored in ${coefficients}
    */
  def computeCoefficients(dot_products: Array[T], seed: Int): Double

  /**
    * combine update model and compute dot product as one action, to reduce the scheduler delay.
    * @param modelRDD
    * @param bcCoefficients
    * @param lastSeed
    * @param newSeed
    * @return
    */
  def updateModelAndComputeDotProduct(modelRDD: RDD[(Array[IndexedDataPoint], Array[T])],
                                      bcCoefficients: Broadcast[Array[T]],
                                      lastSeed: Int, newSeed: Int): Array[T]


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
      val dot_products: Array[T] = updateModelAndComputeDotProduct(modelRDD, bcCoefficients, last_seed, cur_seed)

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