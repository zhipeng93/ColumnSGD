/**
  * Copyright (C) 2017 TU Berlin DIMA
  *
  * Licensed under the Apache License, Version 2.0 (the "License");
  * you may not use this file except in compliance with the License.
  * You may obtain a copy of the License at
  *
  *         http://www.apache.org/licenses/LICENSE-2.0
  *
  * Unless required by applicable law or agreed to in writing, software
  * distributed under the License is distributed on an "AS IS" BASIS,
  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  * See the License for the specific language governing permissions and
  * limitations under the License.
  */

package pku.mllibFP.apps

import java.nio.charset.Charset

import scala.util.hashing.MurmurHash3
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.rdd.RDD

/**
  * Copied from https://github.com/bodenc/ml-benchmark
  * Transform the Crito click log data to LibSVM data file with One Hot encoding of categorical features
  * based on  https://github.com/citlab/PServer/blob/master/preprocessing/criteo/src/main/scala/de/cit/pserver/CriteoPreprocessingJob.scala
  */

object SparkCriteoExtract {

  val Seed = 0

  // Properties of the Criteo Data Set
  val NUM_LABELS = 1
  val NUM_INTEGER_FEATURES = 13
  val NUM_CATEGORICAL_FEATURES = 26
  val NUM_FEATURES = NUM_LABELS + NUM_INTEGER_FEATURES + NUM_CATEGORICAL_FEATURES

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf()
      .setAppName("Feature Hashing Criteo Data Set")
    val sc = new SparkContext(conf)


    val inputPath= args(0).toString
    val outputPath = args(1).toString
    val percDataPoints = args(2).toDouble
    val numFeatures = args(3).toInt // num of features that categories transforsm to, the final number of

    val data: RDD[(Int, Array[Int], Array[String])] = readInput(sc, inputPath)

    // sub- or supersample the data for experiments
    val input = if(percDataPoints != 1.0) {
      data.sample(true, percDataPoints, 42)
    } else {
      data
    }

    val transformedFeatures = input.map(x => {

      val label = x._1
      val intFeatures = x._2
      val catFeatures = x._3

      val hashedIndices = catFeatures
        .filter(!_.isEmpty)
        .map(murmurHash(_, 1, numFeatures))
        .groupBy(_._1)
        .map(colCount => (colCount._1 + NUM_INTEGER_FEATURES + 1 , colCount._2.map(_._2).sum))
        .filter(_._2 != 0)
        .toSeq.sortBy(_._1)

      val intStrings = for ((col, value) <- 1 to intFeatures.size zip intFeatures) yield s"$col:$value"
      val catStrings = for ((col, value) <- hashedIndices) yield s"$col:$value"

      var retString = ""
      if(label <= 0){
        retString  = "0 "
      }
      else{
        retString  = "1 "
      }

      retString + (intStrings ++ catStrings).mkString(" ")
    })

    transformedFeatures.saveAsTextFile(outputPath)
  }


  /**
    * read the input file and separate label, integer and categorical features
    * @param sc
    * @param input
    * @param delimiter
    * @return
    */
  def readInput(sc: SparkContext, input: String, delimiter: String = "\t") = {
    sc.textFile(input) map { line =>
      val features = line.split(delimiter, -1)

      val label = features.take(NUM_LABELS).head.toInt
      val integerFeatures = features.slice(NUM_LABELS, NUM_LABELS + NUM_INTEGER_FEATURES)
        .map(string => if (string.isEmpty) 0 else string.toInt)
      val categorialFeatures = features.slice(NUM_LABELS + NUM_INTEGER_FEATURES, NUM_FEATURES)

      // add dimension so that similar values in diff. dimensions get a different hash
      for(i <-  categorialFeatures.indices){
        categorialFeatures(i) = i + ":" + categorialFeatures(i)
      }

      (label, integerFeatures, categorialFeatures)
    }
  }

  /**
    * awkward hash function
    * @param feature
    * @param count
    * @param numFeatures
    * @return
    */

  private def murmurHash(feature: String, count: Int, numFeatures: Int): (Int, Int) = {
    val hash = MurmurHash3.bytesHash(feature.getBytes(Charset.forName("UTF-8")), Seed)
    val index = scala.math.abs(hash) % numFeatures


    val value = if (hash >= 0) count else -1 * count
    (index, value)
  }

}
