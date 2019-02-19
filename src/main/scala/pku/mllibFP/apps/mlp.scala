package pku.mllibFP.apps

import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.sql.SparkSession

object mlp{
  def main(args: Array[String]): Unit =  {
    val in_path = args(0)
    val num_partitions = args(1).toInt
    val step_size = args(2).toDouble
    val mini_batch_size = args(3).toInt
    val reg_para = args(4).toDouble
    val num_iteration = args(5).toInt
    val num_features = args(6).toInt
    val num_class = args(7).toInt
    // Load the data stored in LIBSVM format as a DataFrame.
    val spark = SparkSession.builder()
      .appName("mlp")
      .getOrCreate()

    val data = spark.read.format("libsvm")
      .load(in_path).repartition(num_partitions)

    // specify layers for the neural network:
    // input layer of size 4 (features), two intermediate of size 5 and 4
    // and output of size 3 (classes)
    val layers = Array[Int](num_features, 1000, num_class)

    // create the trainer and set its parameters
    val trainer = new MultilayerPerceptronClassifier()
      .setLayers(layers)
      .setBlockSize(mini_batch_size)
      .setSeed(1234L)
      .setMaxIter(num_iteration)
      .setSolver("gd")
      .setStepSize(step_size)

    // train the model
    val model = trainer.fit(data)
  }
}