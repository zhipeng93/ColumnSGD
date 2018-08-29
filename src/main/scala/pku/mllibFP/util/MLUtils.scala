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

import org.apache.avro.SchemaBuilder
import org.apache.spark.{HashPartitioner, SparkContext, TaskContext}
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.internal.Logging
import org.apache.spark.mllib.linalg._
import org.apache.spark.rdd.RDD

import scala.collection.mutable
import scala.collection.mutable.ArrayBuilder


/**
  * @param label   label for each data points, could be a [Double]
  * @param features features can be a SparseVector or DenseVector. Here the feature should be part of the
  *                 whole features.
  */
case class LabeledPartDataPoint(label: Double, features: Vector) {
  override def toString: String = {
    s"($label, $features)"
  }
}

/**
  * Helper methods to load libsvm dataset and split them into feature parallel.
  */
object MLUtils extends Logging {


  def bulkBigArrayLoading(
                         sc: SparkContext,
                         path: String,
                         num_features: Int,
                         num_partitions: Int
                         ): RDD[Array[LabeledPartDataPoint]] = {
    // first organize each partition into num_partition big Arrays
    // (pid, (workerId, indexEnd, labels, indices, values)
    val tmp: RDD[(Int, (Int, Array[Int], Array[Double], Array[Int], Array[Double]))] = sc.textFile(path, minPartitions = num_partitions)
      .map(_.trim)
      .filter(line => !(line.isEmpty || line.startsWith("#")))
      .mapPartitions(
        iter => {
          // eatract each line into ${num_partitions} REALLY LONG workSets
          // each workSets contains one partitionId and four arrays,
          // i.e., indexEnd[Int], labels[Double],  indices[Int], values[Doule]
          val local_result: Array[(Int, (ArrayBuilder[Int], ArrayBuilder[Double],ArrayBuilder[Int], ArrayBuilder[Double]))] =
          new Array[(Int, (ArrayBuilder[Int], ArrayBuilder[Double], ArrayBuilder[Int], ArrayBuilder[Double]))](num_partitions)

          for(pid <- 0 until(num_partitions)) {
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

          while(iter.hasNext){  // iter.map() is a lazy one.
            val trim_line = iter.next().trim
            if(!(trim_line.isEmpty || trim_line.startsWith("#"))){
              // deal with one line, and append all the results to localResults.
              tmp_string = trim_line.toCharArray

              var lastPos = 0
              var idx = 0
              while(tmp_string(idx) != space){
                idx += 1
              }
              label = new String(tmp_string, lastPos, idx - lastPos).toDouble
              for(i <- 0 until num_partitions){
                local_result(i)._2._2 += label
              }

              while(idx < tmp_string.length){
                // deal with (key, value)
                lastPos = idx + 1
                while(tmp_string(idx) != colon){
                  idx += 1
                }
                index = new String(tmp_string, lastPos, idx - lastPos).toInt - 1
                lastPos = idx + 1
                while(idx < tmp_string.length && tmp_string(idx) != space){
                  idx += 1
                }
                value = new String(tmp_string, lastPos, idx - lastPos).toDouble
                lastPos = idx + 1

                // range split
                //      new_partition_id = (index - 1) / worker_feature_num
                //      new_feature_id = index - new_partition_id * worker_feature_num

                // hash split:
                new_partition_id = index % num_partitions
                new_feature_id = index / num_partitions
                local_result(new_partition_id)._2._3 += new_feature_id
                local_result(new_partition_id)._2._4 += value
                last_index(new_partition_id) += 1

              }
              for(pid <- 0 until num_partitions){
                local_result(pid)._2._1 += last_index(pid)
              }
            }
          }
          local_result.toIterator.map(
            xx => {
              // partitionId, indexEnd[Int], labels[Double],  indices[Int], values[Doule]
              (xx._1, (TaskContext.getPartitionId(), xx._2._1.result(), xx._2._2.result(), xx._2._3.result(), xx._2._4.result()))
            }
          )
        }
      )

    val ini_worker_num = tmp.partitions.length

    val result: RDD[Array[LabeledPartDataPoint]] = tmp.partitionBy(new HashPartitioner(num_partitions)).mapPartitions(
      iter => {
        val xx: Array[(Int, Array[Int], Array[Double], Array[Int], Array[Double])] =
          new Array[(Int, Array[Int], Array[Double], Array[Int], Array[Double])](ini_worker_num)
        while(iter.hasNext){
          val tt = iter.next()
          xx(tt._2._1) = tt._2
        }
        // restore the arrays into Array[LabeledPartDataPoint]
        var num_data_points = 0
        for(wid <- 0 until ini_worker_num){
          num_data_points += xx(wid)._3.length
        }
        logInfo(s"ghand=number of data points is: ${num_data_points}")
        val result: Array[LabeledPartDataPoint] = new Array[LabeledPartDataPoint](num_data_points)
        var gdid = 0 // global data id
        var startIndexId = -1
        var endIndexId = -1
        for(wid <- 0 until ini_worker_num){
          // deal with one part from the initial worker
          val last_index: Array[Int] = xx(wid)._2
          val labels: Array[Double] = xx(wid)._3
          val indices: Array[Int] = xx(wid)._4
          val values: Array[Double] = xx(wid)._5
          startIndexId = 0
          endIndexId = last_index(0)
          result(gdid) = LabeledPartDataPoint(labels(0),
            new SparseVector(num_features,
              indices.slice(startIndexId, endIndexId),
              values.slice(startIndexId, endIndexId)))
          gdid += 1
          for(ldid <- 1 until last_index.length){ // local data id
            // start: 1
            startIndexId = last_index(ldid - 1)
            endIndexId = last_index(ldid)
            result(gdid) = LabeledPartDataPoint(labels(ldid),
              new SparseVector(num_features,
                indices.slice(startIndexId, endIndexId),
                values.slice(startIndexId, endIndexId)))
            gdid += 1
          }
        }
        Iterator(result)
      }
    )
    result
  }

  def bulkRepartitionLoading(
                           sc: SparkContext,
                           path: String,
                           num_features: Int,
                           num_partitions: Int): RDD[Array[LabeledPartDataPoint]] = {

    // for shuffle, (partitionId, (workerId, label, indices, values))
    val tmp: RDD[(Int, (Int, Array[LabeledPartDataPoint]))] = sc.textFile(path, minPartitions = num_partitions)
      //      .repartition(num_partitions) // this will increase the overhead.
      .mapPartitionsWithIndex(
      (workerId, iter) => {
        var start: Long = System.currentTimeMillis()
        // each partition is compiled into an local_result.
        // (partitionId, (workerId, ArrayBuilder[LabeledDataPoint]))
        val local_result: Array[(Int, (Int, mutable.ArrayBuilder[LabeledPartDataPoint]))] =
        new Array[(Int, (Int, mutable.ArrayBuilder[LabeledPartDataPoint]))](num_partitions)
        for(pid <- 0 until num_partitions){
          local_result(pid) = (pid, (workerId, new mutable.ArrayBuilder.ofRef[LabeledPartDataPoint]()))
          local_result(pid)._2._2.sizeHint(10000000)
        }
        while(iter.hasNext){  // iter.map() is a lazy one.
          val trim_line = iter.next().trim
          if(!(trim_line.isEmpty || trim_line.startsWith("#"))){
            val result: Array[LabeledPartDataPoint] = splitLine(trim_line, num_partitions, num_features)

            for(pid <- 0 until(num_partitions)){
              local_result(pid)._2._2 += result(pid)
            }
          }
        }
        logInfo(s"ghand=parseDataTime(s):${(System.currentTimeMillis() - start ) / 1000.0}, workerId:${workerId}")
        start = System.currentTimeMillis()
        val xx = local_result.map(
          ele => (ele._1, (ele._2._1, ele._2._2.result()))
        ).toIterator
        logInfo(s"ghand=deepCopy(s):${(System.currentTimeMillis() - start ) / 1000.0}, workerId:${workerId}")

        xx
      }
    )

    // partitionId, Iterable(workerId, Array[LabeledPartDataPoint])
    val ini_worker_num = tmp.partitions.length
    val tmp2: RDD[(Int, (Int, Array[LabeledPartDataPoint]))] = tmp.partitionBy(new HashPartitioner(num_partitions))
    // this shuffle takes too long time, start --> mapPartitionWithIndex done, 23s,  however, start --> groupByKey, 70s.

    val xx: RDD[(Array[LabeledPartDataPoint])] = tmp2.mapPartitions(
      iter => {
        var start = System.currentTimeMillis()
        val xx: Array[Array[LabeledPartDataPoint]] = new Array[Array[LabeledPartDataPoint]](ini_worker_num)
        while(iter.hasNext){
          val tt = iter.next()
          xx(tt._2._1) = tt._2._2
        }

        for(wid <- 1 until xx.length){
          xx(0) ++= xx(wid)
        }
        logInfo(s"ghand=combineArrays:${(System.currentTimeMillis() - start ) / 1000.0}")
        Iterator(xx(0))
      }
    )
    xx
  }

  /**
    * stream shuffle, groupByKey on each data point utilizing the order of streaming in each partition.
    * @param sc
    * @param path
    * @param num_features
    * @param num_partitions
    * @return
    */
  def pipeLineRepartitionLoading(
                       sc: SparkContext,
                       path: String,
                       num_features: Int,
                       num_partitions: Int): RDD[Array[LabeledPartDataPoint]] = {

    // for shuffle, (partitionId, (workerId, label, indices, values))
    val tmp: RDD[(Int, (Int, LabeledPartDataPoint))] = sc.textFile(path, minPartitions = num_partitions)
      .map(_.trim)
      .filter(line => !(line.isEmpty || line.startsWith("#")))
      .map(
        line => {
          val xx: Array[LabeledPartDataPoint] = splitLine(line, num_partitions, num_features)
          var pid = -1
          val yy: Iterator[(Int, (Int, LabeledPartDataPoint))] = xx.toIterator.map(lpd =>
            {
              pid += 1
              (pid, (TaskContext.getPartitionId(), lpd))
            }
          )
          yy
        }
      ).flatMap(x => x)

    val ini_num_partitons = tmp.partitions.length

    val tt: RDD[(Int, (Int, LabeledPartDataPoint))] =
//      new ShuffledRDD[Int, (Int, LabeledPartDataPoint), (Int, LabeledPartDataPoint)](tmp, new HashPartitioner(num_partitions))
      tmp.partitionBy(new HashPartitioner(num_partitions))

    tt.mapPartitions(
      iter => {
        val dataPoints: Array[mutable.ArrayBuilder[LabeledPartDataPoint]] =
          new Array[mutable.ArrayBuilder[LabeledPartDataPoint]](ini_num_partitons)
        for(pid <-0 until ini_num_partitons) {
          dataPoints(pid) = new mutable.ArrayBuilder.ofRef[LabeledPartDataPoint]()
          dataPoints(pid).sizeHint(10000000)
        }

        while(iter.hasNext) {
          val ele = iter.next()
          val wid = ele._2._1
          val lpd = ele._2._2
          dataPoints(wid) += lpd
        }

        for(pid <- 1 until(ini_num_partitons))
          dataPoints(0) ++= dataPoints(pid).result()

        Iterator(dataPoints(0).result())
      }
    )
  }


  def splitLine(line: String, num_partitions: Int, num_features: Int): Array[LabeledPartDataPoint] = {
    val result: Array[LabeledPartDataPoint] = new Array[LabeledPartDataPoint](num_partitions)
    val separator = " "

    val indices_builders = new Array[ArrayBuilder[Int]](num_partitions)
    val values_builders = new Array[ArrayBuilder[Double]](num_partitions)
    for(pid <- 0 until(num_partitions)){
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
    while(xx(idx) != space){
      idx += 1
    }
    label = new String(xx, lastPos, idx - lastPos).toDouble
    while(idx < xx.length){
      // deal with (key, value)
      lastPos = idx + 1
      while(xx(idx) != colon){
        idx += 1
      }
      index = new String(xx, lastPos, idx - lastPos).toInt
      lastPos = idx + 1
      while(idx < xx.length && xx(idx) != space){
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

    for(pid <- 0 until(num_partitions)){
      result(pid) = LabeledPartDataPoint(label, new SparseVector(num_features, indices_builders(pid).result(), values_builders(pid).result()))
    }

    result
  }

  /**
    * Loads labeled data in the LIBSVM format into an RDD[IndexedDataPoint].
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
    * @return labeled data stored as RDD[(Array[pku.mllibFP.util.Label])], labels, numFeatures,
    *         one RDD partition contains only one element, that is the array of all the data points of that dimension.
    */
  def loadLibSVMFileFeatureParallel(
                                     sc: SparkContext,
                                     path: String,
                                     num_features: Int,
                                     num_partitions: Int): RDD[Array[LabeledPartDataPoint]] = {
    // local_data_id, label, indices, values
    val parsed: RDD[(Int, Double, Array[Int], Array[Double])] =
      parseLibSVMFile(sc, path, minPartitions = num_partitions)
    /**
      * ACTION HERE [reduce number of actions], compute
      * (1) global IDs for all partitions,
      * (2) total number of features,
      */
    // (partitionId, numDataPoints, numFeatures)
    val partition_info: Array[(Int, Int, Int)] = parsed.mapPartitionsWithIndex {
      (partitionId, iter) => {
        var local_cnt = 0
        // localId => label
        var local_max_num_features = 0
        while(iter.hasNext){
          val data_point = iter.next()
          val data_point_num_feature = data_point._3.lastOption.getOrElse(0)
          local_max_num_features = math.max(local_max_num_features, data_point_num_feature)
          local_cnt += 1
        }
        Iterator((partitionId, local_cnt, local_max_num_features))
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

    val bc_global_startId_per_partition: Broadcast[Array[Int]] = sc.broadcast(global_startId_per_partition)

    // split the data points vertically according to the feature num
    // global_data_id, label, indices, values
    val global_data_point: RDD[(Int, Double, Array[Int], Array[Double])] = parsed.mapPartitionsWithIndex {
      (partitionId, iter) => {
        val local_startID: Int = bc_global_startId_per_partition.value(partitionId)
        iter.map {
          //  (localId: Int, label: Double, indices: Array[Int]. value: Array[Double])
          data_point => {
            (data_point._1 + local_startID, data_point._2, data_point._3, data_point._4)
          }
        }
      }
    }

    // (partitionId, (global_data_id, label, indices, values))
    val global_splitted_data_point: RDD[(Int, (Int, Double, Array[Int], Array[Double]))] = global_data_point.mapPartitions {
      iter => {
        iter.map(
          data_point => splitDataPoint(data_point, num_partitions, real_num_features).toIterator
        )
      }.flatMap(x => x)
    }

    val indexed_data_point: RDD[(Array[LabeledPartDataPoint])] = global_splitted_data_point
      .groupByKey(num_partitions).mapPartitions(
      iter => {
        val partition: (Int, Iterable[(Int, Double, Array[Int], Array[Double])]) = iter.next()
        val partition_id = partition._1
        val part_data_points: Iterator[(Int, Double, Array[Int], Array[Double])] = partition._2.toIterator

        val array_indexed_data_points: Array[LabeledPartDataPoint] = new Array[LabeledPartDataPoint](total_num_data_points)
        while (part_data_points.hasNext) {
          val data_point_in_array: (Int, Double, Array[Int], Array[Double]) = part_data_points.next()
          val data_id = data_point_in_array._1
          val label = data_point_in_array._2
          val indices = data_point_in_array._3
          val values = data_point_in_array._4
          // don't use toArray for SparseVector because it is super inefficient.
          array_indexed_data_points(data_id) = LabeledPartDataPoint(label, new SparseVector(real_num_features, indices, values))
        }

        Iterator((array_indexed_data_points))
      }
    )
    indexed_data_point.setName("input data RDD")
    indexed_data_point
  }

  /**
    * split one data point into ${num_partitions}, to be distributed over the cluster.
    * partitionByRange
    * @param data_point     (idx, label, indices, values) // idx could be the workerId for this data point
    *                       where it comes from, or globalDataId
    * @param num_partitions num_partitons vertically
    * @param num_features   total number of features
    * @return Array[(partitionId, (idx, label, indices, values))]
    */
  def splitDataPoint(data_point: (Int, Double, Array[Int], Array[Double]), num_partitions: Int,
                 num_features: Int): Array[(Int, (Int, Double, Array[Int], Array[Double]))] = {
    val indices: Array[Int] = data_point._3
    val values: Array[Double] = data_point._4

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

    // (partitionId, (idx, indices, values))
    val result: Array[(Int, (Int, Double, Array[Int], Array[Double]))] =
      new Array[(Int, (Int, Double, Array[Int], Array[Double]))](num_partitions)

    partition_id = 0
    while (partition_id < num_partitions) {
      result(partition_id) = (partition_id,
        (data_point._1, data_point._2, indices_builders(partition_id).result(), values_builders(partition_id).result())
      )
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
    * parse one line into a data point, with a idx as its identifier.
    * @param idx , the index of the data point in this line; or the workerId of this line.
    *            For different usage --- localDataId, or workerId
    * @param line             a data point in libsvm format
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
