package pku.mllibFP.apps

import org.apache.spark.{SparkConf, SparkContext}

object sample{
  def main(args: Array[String]): Unit ={
    val sparkconf = new SparkConf().setAppName("sample data")
    val sc = new SparkContext(sparkconf)
    val inpath = args(0)
    val outpath = args(1)
    val fraction = args(2).toDouble

    val data = sc.textFile(inpath).sample(false, fraction)
    data.saveAsTextFile(outpath)

  }
}