package pku.mllibFP.apps

import org.apache.spark.{SparkConf, SparkContext}

object repartition {

  def main(args: Array[String]): Unit = {
    val sparkconf = new SparkConf().setAppName("repartition data")
    val sc = new SparkContext(sparkconf)
    val inpath = args(0)
    val outpath = args(1)

    sc.textFile(inpath).repartition(8).saveAsTextFile(outpath)
  }

}
