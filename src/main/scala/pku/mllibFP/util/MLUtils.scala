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

import org.apache.spark.{HashPartitioner, SparkContext, TaskContext}
import org.apache.spark.internal.Logging
import org.apache.spark.mllib.linalg._
import org.apache.spark.rdd.RDD

import scala.collection.mutable
import scala.collection.mutable.ArrayBuilder

/**
  * Helper methods to load libsvm dataset and split them into feature parallel.
  */
object MLUtils extends Logging {

  /**
    *
    * @param sc
    * @param path
    * @param num_features
    * @param num_partitions
    * @return RDD[ArrayWorkSet[PointWorkset]]
    */
  def bulkCSRLoading(
                           sc: SparkContext,
                           path: String,
                           num_features: Int,
                           num_partitions: Int
                         ): RDD[ArrayWorkSet[WorkSet]] = {
    // first organize each partition into num_partition big Arrays
    // (pid, (workerId, indexEnd, labels, indices, values)
    val csrWorkSet: RDD[(Int, (Int, WorkSet))] = sc.textFile(path, minPartitions = num_partitions)
      .map(_.trim)
      .filter(line => !(line.isEmpty || line.startsWith("#")))
      .mapPartitions(
        iter => {
          // eatract each line into ${num_partitions} REALLY LONG workSets
          // each workSets contains one partitionId and four arrays,
          // i.e., indexEnd[Int], labels[Double],  indices[Int], values[Doule]
          val local_result: Array[(Int, (ArrayBuilder[Int], ArrayBuilder[Double], ArrayBuilder[Int], ArrayBuilder[Double]))] =
          new Array[(Int, (ArrayBuilder[Int], ArrayBuilder[Double], ArrayBuilder[Int], ArrayBuilder[Double]))](num_partitions)

          for (pid <- 0 until (num_partitions)) {
            local_result(pid) =
              (pid, (new mutable.ArrayBuilder.ofInt, new mutable.ArrayBuilder.ofDouble,
                new mutable.ArrayBuilder.ofInt, new mutable.ArrayBuilder.ofDouble))
            local_result(pid)._2._1.sizeHint(1048576) // 2**20
            // indexEnd for each data point.
            // For example, for i-th data point, the range is [index_End(i-1), index_End(i))
            local_result(pid)._2._2.sizeHint(1048576) // labels of each data point
            local_result(pid)._2._3.sizeHint(1048576) // indices
            local_result(pid)._2._4.sizeHint(1048576) // values
          }

          var tmp_string: Array[Char] = null
          val space: Char = ' '
          val colon: Char = ':'
          var label: Double = -1.0
          var index: Int = -1
          var value: Double = -1
          val worker_feature_num: Int = num_features / num_partitions + 1
          var new_partition_id = -1
          var new_feature_id = -1
          val last_index: Array[Int] = new Array[Int](num_partitions)
          // last_index(pid): indexEnd for pid given this data point

          while (iter.hasNext) { // iter.map() is a lazy one.
            val trim_line = iter.next().trim
            if (!(trim_line.isEmpty || trim_line.startsWith("#"))) {
              // deal with one line, and append all the results to localResults.
              tmp_string = trim_line.toCharArray

              var lastPos = 0
              var idx = 0
              while (tmp_string(idx) != space) {
                idx += 1
              }
              label = new String(tmp_string, lastPos, idx - lastPos).toDouble
              for (i <- 0 until num_partitions) {
                local_result(i)._2._2 += label
              }

              while (idx < tmp_string.length) {
                // deal with (key, value)
                lastPos = idx + 1
                while (tmp_string(idx) != colon) {
                  idx += 1
                }
                index = new String(tmp_string, lastPos, idx - lastPos).toInt - 1
                lastPos = idx + 1
                while (idx < tmp_string.length && tmp_string(idx) != space) {
                  idx += 1
                }
                value = new String(tmp_string, lastPos, idx - lastPos).toDouble
                lastPos = idx + 1

                // range split
                //      new_partition_id = (index - 1) / worker_feature_num
                //      new_feature_id = index - new_partition_id * worker_feature_num

                // hash split:
//                new_partition_id = index % num_partitions
//                new_feature_id = index / num_partitions
//                local_result(new_partition_id)._2._3 += new_feature_id
//                local_result(new_partition_id)._2._4 += value
//                last_index(new_partition_id) += 1

                // replicate hash: the features in bucket K is now in $K$ and $K + 1$ % num_buckets
                // NOTE: the index slice of one data point in a partition is not ordered anymore.
                new_partition_id = index % num_partitions
                new_feature_id = index / num_partitions
                local_result(new_partition_id)._2._3 += new_feature_id
                local_result(new_partition_id)._2._4 += value
                last_index(new_partition_id) += 1

                // also put it in the next bucket
                new_partition_id = (new_partition_id + 1) % num_partitions
                new_feature_id = index / num_partitions + (num_features / num_partitions + 1)
                local_result(new_partition_id)._2._3 += new_feature_id
                local_result(new_partition_id)._2._4 += value
                last_index(new_partition_id) += 1


              }
              for (pid <- 0 until num_partitions) {
                local_result(pid)._2._1 += last_index(pid)
              }
            }
          }
          local_result.toIterator.map(
            xx => {
              // partitionId, indexEnd[Int], labels[Double],  indices[Int], values[Doule]
              (xx._1, (TaskContext.getPartitionId(),
                new CSRWorkSet(xx._2._1.result, xx._2._2.result, xx._2._3.result, xx._2._4.result)
              ))
            }
          )
        }
      )

    val ini_worker_num = csrWorkSet.partitions.length

    val result: RDD[ArrayWorkSet[WorkSet]] = csrWorkSet.partitionBy(new HashPartitioner(num_partitions)).mapPartitions(
      iter => {
        // (pid, workset)
        val xx: Array[WorkSet] =
          new Array[WorkSet](ini_worker_num)
        while (iter.hasNext) {
          val tt = iter.next()
          xx(tt._2._1) = tt._2._2
        }
        Iterator(new ArrayWorkSet((xx)))
      }
    )
    result
  }

  /**
    *
    * @param sc
    * @param path
    * @param num_features
    * @param num_partitions
    * @return RDD[ArrayWorkSet[PointWorkset]]
    */
  def bulkPointLoading(
                              sc: SparkContext,
                              path: String,
                              num_features: Int,
                              num_partitions: Int): RDD[ArrayWorkSet[WorkSet]] = {

    // for shuffle, (partitionId, (workerId, PointWorkSet))
    val splitted_data_point_array: RDD[(Int, (Int, PointWorkSet))] = sc.textFile(path, minPartitions = num_partitions)
      .mapPartitionsWithIndex(
        (workerId, iter) => {
          // each partition is compiled into an local_result.
          // (partitionId, (workerId, ArrayBuilder[LabeledDataPoint]))
          val local_result: Array[(Int, (Int, mutable.ArrayBuilder[LabeledPartDataPoint]))] =
          new Array[(Int, (Int, mutable.ArrayBuilder[LabeledPartDataPoint]))](num_partitions)
          for (pid <- 0 until num_partitions) {
            local_result(pid) = (pid, (workerId, new mutable.ArrayBuilder.ofRef[LabeledPartDataPoint]()))
            local_result(pid)._2._2.sizeHint(1048576)
          }
          while (iter.hasNext) { // iter.map() is a lazy one.
            val trim_line = iter.next().trim
            if (!(trim_line.isEmpty || trim_line.startsWith("#"))) {
              val result: Array[LabeledPartDataPoint] = splitLine(trim_line, num_partitions, num_features)

              for (pid <- 0 until (num_partitions)) {
                local_result(pid)._2._2 += result(pid)
              }
            }
          }

          val xx = local_result.map(
            ele => (ele._1, (ele._2._1, new PointWorkSet(ele._2._2.result())))
          ).toIterator

          xx
        }
      )

    // partitionId, Iterable(workerId, Array[LabeledPartDataPoint])
    val ini_worker_num = splitted_data_point_array.partitions.length
    splitted_data_point_array.partitionBy(new HashPartitioner(num_partitions)).mapPartitions(
      iter => {
        val xx: Array[WorkSet] = new Array[WorkSet](ini_worker_num)
        while (iter.hasNext) {
          val tt = iter.next()
          xx(tt._2._1) = tt._2._2
        }

        Iterator(new ArrayWorkSet[WorkSet](xx))

      }
    )

  }

  /**
    * stream shuffle, groupByKey on each data point utilizing the order of streaming in each partition.
    *
    * @param sc
    * @param path
    * @param num_features
    * @param num_partitions
    * @return RDD[ArrayWorkSet[PointWorkset]]
    */
  def pipeLinePointLoading(
                                  sc: SparkContext,
                                  path: String,
                                  num_features: Int,
                                  num_partitions: Int): RDD[ArrayWorkSet[WorkSet]] = {

    // for shuffle, (partitionId, (workerId, label, indices, values))
    val splitted_data_point: RDD[(Int, (Int, LabeledPartDataPoint))] = sc.textFile(path, minPartitions = num_partitions)
      .map(_.trim)
      .filter(line => !(line.isEmpty || line.startsWith("#")))
      .map(
        line => {
          val xx: Array[LabeledPartDataPoint] = splitLine(line, num_partitions, num_features)
          var pid = -1
          val yy: Iterator[(Int, (Int, LabeledPartDataPoint))] = xx.toIterator.map(lpd => {
            pid += 1
            (pid, (TaskContext.getPartitionId(), lpd))
          }
          )
          yy
        }
      ).flatMap(x => x)

    val ini_num_partitons = splitted_data_point.partitions.length

    splitted_data_point.partitionBy(new HashPartitioner(num_partitions)).mapPartitions(
      iter => {
        val dataPoints: Array[mutable.ArrayBuilder[LabeledPartDataPoint]] =
          new Array[mutable.ArrayBuilder[LabeledPartDataPoint]](ini_num_partitons)
        for (pid <- 0 until ini_num_partitons) {
          dataPoints(pid) = new mutable.ArrayBuilder.ofRef[LabeledPartDataPoint]()
          dataPoints(pid).sizeHint(1048576)
        }

        while (iter.hasNext) {
          val ele = iter.next()
          val wid = ele._2._1
          val lpd = ele._2._2
          dataPoints(wid) += lpd
        }

        val result: ArrayWorkSet[WorkSet] =
          new ArrayWorkSet[WorkSet](dataPoints.map(x => new PointWorkSet(x.result())))

        Iterator(result)
      }
    )
  }

  /**
    * Loads labeled data in the LIBSVM format into an RDD[ArrayWorkSet].
    * The LIBSVM format is a text-based format used by LIBSVM and LIBLINEAR.
    * Each line represents a labeled sparse feature vector using the following format:
    * {{{label index1:value1 index2:value2 ...}}}
    * where the indices are one-based and in ascending order.
    * This method parses each line into a [[pku.mllibFP.util.LabeledPartDataPoint]],
    * where the feature indices are converted to zero-based.
    *
    * @param sc             Spark context
    * @param path           file or directory path in any Hadoop-supported file system URI
    * @param num_partitions min number of partitions
    * @return RDD[ArrayWorkSet[PointWorkset]]
    */
  def loadLibSVMFileFeatureParallel(
                                     sc: SparkContext,
                                     path: String,
                                     num_features: Int,
                                     num_partitions: Int): RDD[ArrayWorkSet[WorkSet]] = {
    // (partitionId, label, indices, values), partitionId is used for re-ordering the data points
    val parsed: RDD[(Int, Double, Array[Int], Array[Double])] =
      parseLibSVMFile(sc, path, minPartitions = num_partitions)

    // (partitionId, (workerId, label, indices, values))
    val splitted_data_point: RDD[(Int, (Int, LabeledPartDataPoint))] = parsed.mapPartitions {
      iter => {
        iter.map(
          data_point => {
            val xx: Array[LabeledPartDataPoint] = splitDataPoint((data_point._2, data_point._3, data_point._4),
              num_partitions, num_features)
            var pid = -1
            val yy: Iterator[(Int, (Int, LabeledPartDataPoint))] = xx.toIterator.map(lpd => {
              pid += 1
              (pid, (TaskContext.getPartitionId(), lpd))
            })
            yy
          }
        )
      }.flatMap(x => x)
    }
    val ini_num_partitons = splitted_data_point.getNumPartitions
    splitted_data_point.partitionBy(new HashPartitioner(num_partitions)).mapPartitions(
      iter => {
        val dataPoints: Array[mutable.ArrayBuilder[LabeledPartDataPoint]] =
          new Array[mutable.ArrayBuilder[LabeledPartDataPoint]](ini_num_partitons)
        for (pid <- 0 until ini_num_partitons) {
          dataPoints(pid) = new mutable.ArrayBuilder.ofRef[LabeledPartDataPoint]()
          dataPoints(pid).sizeHint(1048576)
        }

        while (iter.hasNext) {
          val ele = iter.next()
          val wid = ele._2._1
          val lpd = ele._2._2
          dataPoints(wid) += lpd
        }

        val result: ArrayWorkSet[WorkSet] =
          new ArrayWorkSet[WorkSet](dataPoints.map(x => new PointWorkSet(x.result())))

        Iterator(result)
      }
    )

  }

  def splitLine(line: String, num_partitions: Int, num_features: Int): Array[LabeledPartDataPoint] = {
    val result: Array[LabeledPartDataPoint] = new Array[LabeledPartDataPoint](num_partitions)
    val separator = " "

    val indices_builders = new Array[ArrayBuilder[Int]](num_partitions)
    val values_builders = new Array[ArrayBuilder[Double]](num_partitions)
    for (pid <- 0 until (num_partitions)) {
      indices_builders(pid) = new ArrayBuilder.ofInt
      values_builders(pid) = new ArrayBuilder.ofDouble
    }
    val xx: Array[Char] = line.toCharArray
    val space: Char = ' '
    val colon: Char = ':'
    var lastPos = 0
    var label: Double = -1.0
    var index: Int = -1
    var value: Double = -1
    val worker_feature_num: Int = num_features / num_partitions + 1
    var new_partition_id = -1
    var new_feature_id = -1

    lastPos = 0
    var idx = 0
    while (xx(idx) != space) {
      idx += 1
    }
    label = new String(xx, lastPos, idx - lastPos).toDouble
    while (idx < xx.length) {
      // deal with (key, value)
      lastPos = idx + 1
      while (xx(idx) != colon) {
        idx += 1
      }
      index = new String(xx, lastPos, idx - lastPos).toInt
      lastPos = idx + 1
      while (idx < xx.length && xx(idx) != space) {
        idx += 1
      }
      value = new String(xx, lastPos, idx - lastPos).toDouble
      lastPos = idx + 1

      // range split
      //      new_partition_id = (index - 1) / worker_feature_num
      //      new_feature_id = index - new_partition_id * worker_feature_num

      // hash split:
      new_partition_id = index % num_partitions
      new_feature_id = index / num_partitions

      indices_builders(new_partition_id) += new_feature_id
      values_builders(new_partition_id) += value
    }

    for (pid <- 0 until (num_partitions)) {
      result(pid) = LabeledPartDataPoint(label, new SparseVector(num_features, indices_builders(pid).result(), values_builders(pid).result()))
    }

    result
  }

  /**
    * split one data point into ${num_partitions}, to be distributed over the cluster.
    * partitionByRange
    *
    * @param data_point     (idx, label, indices, values) // idx could be the workerId for this data point
    *                       where it comes from, or globalDataId
    * @param num_partitions num_partitons vertically
    * @param num_features   total number of features
    * @return Array[LabeledPartDataPoint]
    */
  def splitDataPoint(data_point: (Double, Array[Int], Array[Double]), num_partitions: Int,
                     num_features: Int): Array[LabeledPartDataPoint] = {
    val indices: Array[Int] = data_point._2
    val values: Array[Double] = data_point._3

    val indices_builders = new Array[ArrayBuilder[Int]](num_partitions)
    val values_builders = new Array[ArrayBuilder[Double]](num_partitions)
    for (pid <- 0 until num_partitions) {
      indices_builders(pid) = new ArrayBuilder.ofInt
      values_builders(pid) = new ArrayBuilder.ofDouble
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

    val result: Array[LabeledPartDataPoint] = new Array[LabeledPartDataPoint](num_partitions)
    for (pid <- 0 until num_partitions) {
      result(pid) = LabeledPartDataPoint(data_point._1,
        new SparseVector(num_features, indices_builders(pid).result, values_builders(pid).result))
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
        iter: Iterator[String] => {
          iter.map(_.trim)
            .filter(line => !(line.isEmpty || line.startsWith("#")))
            .map(
              line => {
                parseLibSVMRecord(TaskContext.getPartitionId(), line)
              }
            )
        }
      }
  }

  /**
    * parse one line into a data point, with a idx as its identifier.
    *
    * @param idx  , the index of the data point in this line; or the workerId of this line.
    *             For different usage --- localDataId, or workerId
    * @param line a data point in libsvm format
    * @return
    */
  def parseLibSVMRecord(idx: Int, line: String): (Int, Double, Array[Int], Array[Double]) = {
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
    (idx, label, indices, values)
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


