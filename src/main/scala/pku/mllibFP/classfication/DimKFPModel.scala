package pku.mllibFP.classfication

import org.apache.spark.rdd.RDD
import pku.mllibFP.util.IndexedDataPoint


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

  override def aggregateResult(array1: Array[Array[Double]], array2: Array[Array[Double]]): Array[Array[Double]] = {
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
}