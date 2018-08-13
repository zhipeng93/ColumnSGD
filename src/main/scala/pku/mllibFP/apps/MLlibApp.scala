package pku.mllibFP.apps

import org.apache.spark.mllib.classification.{LogisticRegressionWithSGD, SVMWithSGD}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

object MLlibApp{
  def main(args: Array[String]): Unit ={
    val in_path = args(0)
    val num_partitions = args(1).toInt

    val step_size = args(2).toDouble
    val mini_batch_fraction = args(3).toDouble
    val reg_para = args(4).toDouble
    val num_iteration = args(5).toInt
    val num_features = args(6).toInt

    val model_name: String = args(7).toUpperCase

    val sparkConf = new SparkConf().setAppName("MLlib-" + model_name)
    val sparkContext = new SparkContext(sparkConf)

    // RDD[(Int, Array[IndexedDataPoint])], Array[Double])
    val data_rdd: RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sparkContext, in_path, num_features, minPartitions = num_partitions)
      .repartition(num_partitions)
      .persist(StorageLevel.MEMORY_ONLY)

    val models: Array[String] = model_name.split("-")
    for(m <- models) {
      m match {
        case "SVM" =>
          SVMWithSGD.train(input = data_rdd,
            numIterations = num_iteration,
            stepSize = step_size,
            regParam = reg_para,
            miniBatchFraction = mini_batch_fraction)
        case "LR" =>
          LogisticRegressionWithSGD.train(input = data_rdd,
            numIterations = num_iteration,
            stepSize = step_size,
            miniBatchFraction = mini_batch_fraction)
        // the regularization is zero by default. They did not offer a interface for LR with SGD.
      }
    }

  }
}