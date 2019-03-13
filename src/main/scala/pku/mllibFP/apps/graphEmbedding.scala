package pku.mllibFP.apps

import org.apache.spark.graphx.{GraphLoader, GraphXUtils, VertexRDD}
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext

object graphEmbedding{
  def main(args: Array[String]) = {
    val input_path = args(0)

    val conf= new SparkConf()
    GraphXUtils.registerKryoClasses(conf)
    val sc = new SparkContext(conf)

    val graph = GraphLoader.edgeListFile(sc, input_path)

    val vertices: VertexRDD[Double] = graph.pageRank(0.001, 0.15).vertices
    vertices.take(10).foreach(println)


  }
}