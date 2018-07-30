package pku.mllibFP.apps

import org.apache.spark.ml.classification.LinearSVC
import org.apache.spark.ml.feature.{LabeledPoint => ml_LabeledPoint}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.SQLContext
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{SparkConf, SparkContext, SparkEnv}
import org.apache.spark.ml.classification.LogisticRegression

object MLApp{
  def main(args: Array[String]): Unit ={
    val in_path = args(0)
    val num_partitions = args(1).toInt
    val step_size = args(2).toDouble
    val mini_batch_fraction = args(3).toDouble
    val reg_para = args(4).toDouble
    val num_iteration = args(5).toInt
    val num_features = args(6).toInt

    val model_name: String = args(7).toUpperCase


    val sparkConf = new SparkConf().setAppName("ML-baseline" + model_name)
    val sparkContext = new SparkContext(sparkConf)
    val sqlContext = new SQLContext(sparkContext)

    // convert mllib labeledPoint to ml labeledpoint
    val rdd_train_data = MLUtils.loadLibSVMFile(sparkContext, in_path, numFeatures = num_features)
      .map(x => ml_LabeledPoint(x.label, x.features.asML))
    rdd_train_data.setName("cached data")

    val train_data = sqlContext.createDataFrame(rdd_train_data)
      .repartition(num_partitions)
      .persist(StorageLevel.MEMORY_ONLY)
    train_data.count()

    val models: Array[String] = model_name.split("-")
    for(m <- models) {
      m match {
        case "SVM" => {
          val lsvc = new LinearSVC()
            .setMaxIter(num_iteration)
            .setRegParam(reg_para)
            .setFitIntercept(false)
            .setStandardization(false)
            .setTol(0)
          lsvc.fit(train_data)
        }
        case "LR" => {
          val lr = new LogisticRegression()
            .setMaxIter(num_iteration)
            .setRegParam(reg_para)
            .setElasticNetParam(0)
            .setFitIntercept(false)
            .setStandardization(false)
          lr.fit(train_data)
        }

        case "MLR" => {
          val mlr = new LogisticRegression()
            .setMaxIter(10)
            .setRegParam(reg_para)
            .setElasticNetParam(0)
            .setFamily("multinomial")
            .setFitIntercept(false)
            .setStandardization(false)
          mlr.fit(train_data)
        }
      }

    }
  }
}