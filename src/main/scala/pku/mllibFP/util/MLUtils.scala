/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package pku.mllibFP.util

import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.internal.Logging
import org.apache.spark.mllib.linalg._
import org.apache.spark.rdd.RDD

import scala.collection.mutable.ArrayBuilder


/**
  * @param dataId   index each data points, the id ranges from 0 to dataNum - 1
  * @param features features can be a SparseVector or DenseVector. Here the feature should be part of the
  *                 whole features.
  */
case class IndexedDataPoint(dataId: Int, features: Vector) {
  override def toString: String = {
    s"($dataId, $features)"
  }
}

/**
  * Helper methods to load libsvm dataset and split them into feature parallel.
  */
object MLUtils extends Logging {
  /**
    * Loads labeled data in the LIBSVM format into an RDD[IndexedDataPoint].
    * The LIBSVM format is a text-based format used by LIBSVM and LIBLINEAR.
    * Each line represents a labeled sparse feature vector using the following format:
    * {{{label index1:value1 index2:value2 ...}}}
    * where the indices are one-based and in ascending order.
    * This method parses each line into a [[pku.mllibFP.util.IndexedDataPoint]],
    * where the feature indices are converted to zero-based.
    *
    * @param sc             Spark context
    * @param path           file or directory path in any Hadoop-supported file system URI
    * @param num_partitions min number of partitions
    * @return labeled data stored as RDD[(Array[IndexedDataPoint])], labels, numFeatures,
    *         one RDD partition contains only one element, that is the array of all the data points of that dimension.
    */
  def loadLibSVMFileFeatureParallel(
                                     sc: SparkContext,
                                     path: String,
                                     num_features: Int,
                                     num_partitions: Int): (RDD[Array[IndexedDataPoint]], Array[Double]) = {
    // local_data_id, label, indices, values
    val parsed: RDD[(Int, Double, Array[Int], Array[Double])] =
      parseLibSVMFile(sc, path, minPartitions = num_partitions)
    /**
      * ACTION HERE [reduce number of actions], compute
      * (1) global IDs for all partitions,
      * (2) total number of features,
      * (3) also collect labels to the driver.
      */
    // (partitionId, numDataPoints, numFeatures, Array[label])
    val partition_info: Array[(Int, Int, Int, Array[Double])] = parsed.mapPartitionsWithIndex {
      (partitionId, iter) => {
        var local_cnt = 0
        // localId => label
        val arrayBuilder = new ArrayBuilder.ofDouble
        var local_max_num_features = 0
        while(iter.hasNext){
          val data_point = iter.next()
          val data_point_num_feature = data_point._3.lastOption.getOrElse(0)
          local_max_num_features = math.max(local_max_num_features, data_point_num_feature)
          arrayBuilder += data_point._2
          local_cnt += 1
        }
        Iterator((partitionId, local_cnt, local_max_num_features, arrayBuilder.result()))
      }
    }.collect()

    val num_data_points_per_partition: Array[Int] = new Array[Int](partition_info.length)
    var computed_num_features = 0
    var total_num_data_points = 0

    // compute number of features and number of data points.
    var i = 0
    while (i < partition_info.length) {
      num_data_points_per_partition(partition_info(i)._1) = partition_info(i)._2
      computed_num_features = math.max(computed_num_features, partition_info(i)._3)
      total_num_data_points += partition_info(i)._2
      i += 1
    }
    var real_num_features = num_features
    if(num_features < 0)
      real_num_features = computed_num_features + 1 // start from zero.
    logInfo(s"ghandFP=computedNumFeatures=${real_num_features}, totalNumDataPoints:${total_num_data_points}")

    // compute global start Ids for all partitions
    val global_startId_per_partition: Array[Int] = new Array[Int](partition_info.length)
    global_startId_per_partition(0) = 0
    i = 1
    while (i < num_data_points_per_partition.length) {
      global_startId_per_partition(i) = num_data_points_per_partition(i - 1) + global_startId_per_partition(i - 1)
      i += 1
    }

    // compute label information
    val global_labels: Array[Double] = new Array[Double](total_num_data_points)
    i = 0
    while(i < partition_info.length){
      val local_label_info: Array[Double] = partition_info(i)._4
      var k = 0
      while(k < local_label_info.length){
        global_labels(k + global_startId_per_partition(i)) = local_label_info(k)
        k += 1
      }
      i += 1
    }

    val bc_global_startId_per_partition: Broadcast[Array[Int]] = sc.broadcast(global_startId_per_partition)

    // split the data points vertically according to the feature num
    // partition_id, global_data_id, indices, values
    val global_data_point: RDD[(Int, Array[Int], Array[Double])] = parsed.mapPartitionsWithIndex {
      (partitionId, iter) => {
        val local_startID: Int = bc_global_startId_per_partition.value(partitionId)
        iter.map {
          //  (localId: Int, label: Double, indices: Array[Int]. value: Array[Double])
          data_point => {
            (data_point._1 + local_startID, data_point._3, data_point._4)
          }
        }
      }
    }

    // (partitionId, (global_data_id, indices, values))
    val global_splitted_data_point: RDD[(Int, (Int, Array[Int], Array[Double]))] = global_data_point.mapPartitions {
      iter => {
        iter.map(
          data_point => splitDataPoint(data_point, num_partitions, real_num_features).toIterator
        )
      }.flatMap(x => x)
    }

    val num_data_points = global_labels.length
    val indexed_data_point: RDD[(Array[IndexedDataPoint])] = global_splitted_data_point
      .groupByKey(num_partitions).mapPartitions(
      iter => {
        val partition: (Int, Iterable[(Int, Array[Int], Array[Double])]) = iter.next()
        val partition_id = partition._1
        val part_data_points: Iterator[(Int, Array[Int], Array[Double])] = partition._2.toIterator

        val array_indexed_data_points: Array[IndexedDataPoint] = new Array[IndexedDataPoint](num_data_points)
        while (part_data_points.hasNext) {
          val data_point_in_array: (Int, Array[Int], Array[Double]) = part_data_points.next()
          val data_id = data_point_in_array._1
          val indices = data_point_in_array._2
          val values = data_point_in_array._3
          // don't use toArray for SparseVector because it is super inefficient.
          array_indexed_data_points(data_id) = IndexedDataPoint(data_id, new SparseVector(real_num_features, indices, values))
        }

        Iterator((array_indexed_data_points))
      }
    )
    indexed_data_point.setName("input data RDD")
    (indexed_data_point, global_labels)
  }

  /**
    * split one data point into ${num_partitions}, to be distributed over the cluster.
    * partitionByRange
    * @param data_point     (globalDataId, indices, values)
    * @param num_partitions num_partitons vertically
    * @param num_features   total number of features
    * @return (partitionId, (globalDataId, indices, values))
    */
  def splitDataPoint(data_point: (Int, Array[Int], Array[Double]), num_partitions: Int,
                 num_features: Int): Array[(Int, (Int, Array[Int], Array[Double]))] = {
    val indices: Array[Int] = data_point._2
    val values: Array[Double] = data_point._3

    val indices_builders = new Array[ArrayBuilder[Int]](num_partitions)
    val values_builders = new Array[ArrayBuilder[Double]](num_partitions)
    var partition_id = 0
    while (partition_id < num_partitions) {
      indices_builders(partition_id) = new ArrayBuilder.ofInt
      values_builders(partition_id) = new ArrayBuilder.ofDouble
      partition_id += 1
    }

    var featureId = 0
    var new_partition_id = -1
    var new_feature_id = -1
    val worker_feature_num = num_features / num_partitions + 1
    while (featureId < indices.length) {
      // range split: id --> partitionId = id / worker_feature_num, newDataId = id - partitionId * worker_feature_num
//      new_partition_id = indices(featureId) / worker_feature_num
//      new_feature_id = indices(featureId) - new_partition_id * worker_feature_num

      // Hash split: id --> partitionId = id % num_partitions, newDataId = id / num_partitions
      new_partition_id = indices(featureId) % num_partitions
      new_feature_id = indices(featureId) / num_partitions

      indices_builders(new_partition_id) += new_feature_id
      values_builders(new_partition_id) += values(featureId)

      featureId += 1
    }

    // (partitionId, (globalDataId, indices, values))
    val result: Array[(Int, (Int, Array[Int], Array[Double]))] = new Array[(Int, (Int, Array[Int], Array[Double]))](num_partitions)

    partition_id = 0
    while (partition_id < num_partitions) {
      result(partition_id) = (partition_id, (data_point._1, indices_builders(partition_id).result(), values_builders(partition_id).result()))
      partition_id += 1
    }

    result
  }

  /**
    * @param sc
    * @param path
    * @param minPartitions
    * @return
    */
   def parseLibSVMFile(
                                      sc: SparkContext,
                                      path: String,
                                      minPartitions: Int): RDD[(Int, Double, Array[Int], Array[Double])] = {
    sc.textFile(path, minPartitions)
      .mapPartitions {
        var local_dataPoint_id: Int = -1
        iter: Iterator[String] => {
          iter.map(_.trim)
            .filter(line => !(line.isEmpty || line.startsWith("#")))
            .map(
              line => {
                local_dataPoint_id += 1
                parseLibSVMRecord(local_dataPoint_id, line)
              }
            )
        }
      }
  }

  /**
    * @param localDataPointId , the index of the data point in this line.
    * @param line             a data point in libsvm format
    * @return
    */
  def parseLibSVMRecord(localDataPointId: Int, line: String): (Int, Double, Array[Int], Array[Double]) = {
    val items = line.split(' ')
    val label = items.head.toDouble
    val (indices, values) = items.tail.filter(_.nonEmpty).map { item =>
      val indexAndValue = item.split(':')
      val index = indexAndValue(0).toInt - 1 // Convert 1-based indices to 0-based.
    val value = indexAndValue(1).toDouble
      (index, value)
    }.unzip

    // check if indices are one-based and in ascending order
    var previous = -1
    var i = 0
    val indicesLength = indices.length
    while (i < indicesLength) {
      val current = indices(i)
      require(current > previous, s"indices should be one-based and in ascending order;"
        +
        s""" found current=$current, previous=$previous; line="$line"""")
      previous = current
      i += 1
    }
    (localDataPointId, label, indices, values)
  }

  /**
    * When `x` is positive and large, computing `math.log(1 + math.exp(x))` will lead to arithmetic
    * overflow. This will happen when `x > 709.78` which is not a very large number.
    * It can be addressed by rewriting the formula into `x + math.log1p(math.exp(-x))` when `x > 0`.
    *
    * @param x a floating-point value as input.
    * @return the result of `math.log(1 + math.exp(x))`.
    */
  def log1pExp(x: Double): Double = {
    if (x > 0) {
      x + math.log1p(math.exp(-x))
    } else {
      math.log1p(math.exp(x))
    }
  }

}
