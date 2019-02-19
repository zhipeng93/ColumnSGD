package pku.mllibFP.apps

import org.apache.spark.{SparkConf, SparkContext, SparkEnv}
import pku.mllibFP.util.MLUtils
import pku.mllibFP.classfication._

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
    val num_class: Int = SparkEnv.get.conf.get("spark.ml.numClasses", "2").toInt

    // RDD[WorkSet]
    val fp_rdd = MLUtils.bulkCSRLoading(sparkContext, in_path, num_features, num_partitions)
    // not cached, to be cached in FPModel

    val models: Array[String] = model_name.split("-")
    for(m <- models) {
      m match {
        case "SVM" =>
          new SVM(inputRDD = fp_rdd,
            numFeatures = num_features,
            numPartitions = num_partitions,
            regParam = reg_para,
            stepSize = step_size,
            numIterations = num_iteration,
            miniBatchSize = mini_batch_size).miniBatchSGD()
        case "LR" =>
          new LR(inputRDD = fp_rdd,
            numFeatures = num_features,
            numPartitions = num_partitions,
            regParam = reg_para,
            stepSize = step_size,
            numIterations = num_iteration,
            miniBatchSize = mini_batch_size).miniBatchSGD()
        case "ADAMSVM" =>
          new AdamSVM(inputRDD = fp_rdd,
            numFeatures = num_features,
            numPartitions = num_partitions,
            regParam = reg_para,
            stepSize = step_size,
            numIterations = num_iteration,
            miniBatchSize = mini_batch_size).miniBatchSGD()
        case "ADAMLR" =>
          new AdamLR(inputRDD = fp_rdd,
            numFeatures = num_features,
            numPartitions = num_partitions,
            regParam = reg_para,
            stepSize = step_size,
            numIterations = num_iteration,
            miniBatchSize = mini_batch_size).miniBatchSGD()
        case "MLR" =>
          val vk = {
            if (args.length > 8)
              args(8).toInt
            else
              2
          }
          new MLR(inputRDD = fp_rdd,
            numFeatures = num_features,
            numPartitions = num_partitions,
            regParam = reg_para,
            stepSize = step_size,
            numIterations = num_iteration,
            miniBatchSize = mini_batch_size,
            modelK = vk).miniBatchSGD()

        case "FM" => {
          val vk = {
            if (args.length > 8)
              args(8).toInt
            else
              10
          }
          new FM(inputRDD = fp_rdd,
            numFeatures = num_features,
            numPartitions = num_partitions,
            regParam = reg_para,
            stepSize = step_size,
            numIterations = num_iteration,
            miniBatchSize = mini_batch_size,
            modelK = vk).miniBatchSGD()
        }
        case "MLP" => {
          new MLP(inputRDD = fp_rdd,
            numFeatures = num_features,
            numPartitions = num_partitions,
            regParam = reg_para,
            stepSize = step_size,
            numIterations = num_iteration,
            miniBatchSize = mini_batch_size,
            numClasses = num_class).miniBatchSGD()
        }
      }
    }

  }
}
