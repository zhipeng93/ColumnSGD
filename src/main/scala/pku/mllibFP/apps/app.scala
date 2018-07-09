package pku.mllibFP.apps

import org.apache.spark.{SparkConf, SparkContext}
import pku.mllibFP.util.MLUtils
import pku.mllibFP.classfication.{SVM, LR}

object app{
  def main(args: Array[String]): Unit ={
    val in_path = args(0)
    val num_partitions = args(1).toInt

    val step_size = args(2).toDouble
    val mini_batch_size = args(3).toInt
    val reg_para = args(4).toDouble
    val num_iteration = args(5).toInt
    val num_features = args(6).toInt

    val model_name: String = args(7).toUpperCase

    val sparkConf = new SparkConf().setAppName("FP-" + model_name)
    val sparkContext = new SparkContext(sparkConf)

    // RDD[(Int, Array[IndexedDataPoint])], Array[Double])
    val fp_rdd = MLUtils.loadLibSVMFileFeatureParallel(sparkContext, in_path, num_features, num_partitions)
    // not cached, to be cached in FPModel

    model_name match {
      case "SVM" =>
        new SVM(inputRDD = fp_rdd._1,
          labels = fp_rdd._2,
          numFeatures = num_features,
          numPartitions = num_partitions,
          regParam = reg_para,
          stepSize = step_size,
          numIterations = num_iteration,
          miniBatchSize = mini_batch_size).miniBatchSGD()
      case "LR" =>
        new LR(inputRDD = fp_rdd._1,
          labels = fp_rdd._2,
          numFeatures = num_features,
          numPartitions = num_partitions,
          regParam = reg_para,
          stepSize = step_size,
          numIterations = num_iteration,
          miniBatchSize = mini_batch_size).miniBatchSGD()
    }

  }
}