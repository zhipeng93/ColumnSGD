package pku.mllibFP.util

import org.apache.spark.internal.Logging
import org.apache.spark.mllib.linalg.{SparseVector, Vector}

import scala.util.Random

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
  * worksets on a worker, that is, it contains a set of data points, but each data point contains part of the features.
  * Each worker can contain several workerSets, but they need to be ordered --- the worksets on each worker are related.
  */
abstract class WorkSet extends Serializable with Logging{
  def getLabeledPartDataPoint(index: Int): LabeledPartDataPoint
  def getNumDataPoints(): Int
  def getLabels(): Array[Double]
}

class CSRWorkSet(val index_pointers: Array[Int], val labels: Array[Double], val indices: Array[Int],
                 val values: Array[Double]) extends WorkSet {
  override def getLabeledPartDataPoint(index: Int): LabeledPartDataPoint = {
    assert(index < getNumDataPoints())
    var start = 0
    if (index != 0) {
      start = index_pointers(index - 1)
    }
    val end = index_pointers(index)
    if(end < start)
      println(s"ghand=ArrayOutOfIndex: start: ${start}, end: ${end}")
    LabeledPartDataPoint(labels(index),
      new SparseVector(end - start, indices.slice(start, end), values.slice(start, end)))
    // don't use toArray method, here size is not the correct one.
  }

  override def getNumDataPoints(): Int = labels.length

  override def getLabels(): Array[Double] = {
    labels
  }

//  /**
//    * group the worksets into one in order.
//    * there is a problem that the number of a CSRWorkSet must < INT.MAX
//    * */
//  def group(others: Array[CSRWorkSet]): WorkSet = {
//    // combine this and other WorkSets
//    var dataNum = 0
//    var featureNum: Int = 0
//    for(wid <- 0 until others.length){
//      dataNum += others(wid).labels.length
//      featureNum += others(wid).indices.length
//    }
//    logInfo(s"ghand=featureNumber:${featureNum}")
//    val index_pointers_t = new Array[Int](dataNum)
//    val labels_t = new Array[Double](dataNum)
//    val indices_t = new Array[Int](featureNum)
//    val values_t = new Array[Double](featureNum)
//    // copy all of the worksets to the tmp values
//
//    // update index pointer
//    var next_label_id = 0 // the global id of the first elements in the next workset
//    var next_index_id = 0 // the global id of the first element in the next workset
//    for(wid <- 0 until(others.length)){
//      val tmp_workset = others(wid)
//      System.arraycopy(tmp_workset.labels, 0, labels_t, next_label_id, tmp_workset.labels.length)
//      System.arraycopy(tmp_workset.indices, 0, indices_t, next_index_id, tmp_workset.indices.length)
//      System.arraycopy(tmp_workset.values, 0, values_t, next_index_id, tmp_workset.values.length)
//
//      for(ip <- 0 until(tmp_workset.index_pointers.length)){
//        index_pointers_t(next_label_id + ip) = next_index_id + tmp_workset.index_pointers(ip)
//      }
//
//      next_label_id += tmp_workset.labels.length
//      next_index_id += tmp_workset.indices.length
//    }
//
//    new CSRWorkSet(index_pointers_t, labels_t, indices_t, values_t)
//
//  }
}


class PointWorkSet(xarray: Array[LabeledPartDataPoint]) extends WorkSet{
  override def getLabeledPartDataPoint(index: Int): LabeledPartDataPoint = {
    xarray(index)
  }

  override def getNumDataPoints(): Int = xarray.length

  override def getLabels(): Array[Double] = {
    val result: Array[Double] = new Array[Double](getNumDataPoints())
    for(id <-0 until(getNumDataPoints())){
      result(id) = getLabeledPartDataPoint(id).label
    }

    result
  }
}



/**
  * As a optimization for CSRWorkSet/PointWorkSet for two reasons:
  *  1. avoid memory copy
  *  2. to hold features more than INTEGER.MAX on a single partition
  */
class ArrayWorkSet[T <: WorkSet](val arrayWorkset: Array[T]){

  def length() : Int = arrayWorkset.length
  /**
    * sample a data point using a random generator
    * @param random
    * @return
    */
  def getRandomLabeledPartDataPoint(random: Random): LabeledPartDataPoint = {
    var workSetId: Int = random.nextInt(arrayWorkset.length)
    while(arrayWorkset(workSetId).getNumDataPoints() == 0) // avoid empty partition
      workSetId = random.nextInt(arrayWorkset.length)
    val localNumDataPoints: Int = arrayWorkset(workSetId).getNumDataPoints()
    arrayWorkset(workSetId).getLabeledPartDataPoint(random.nextInt(localNumDataPoints))
  }

  /**
    * get labels of some worksets referenced by WorkSetIds
    * @param workSetIds
    * @return
    */
  def getLabels(workSetIds: Array[Int]): Array[(Int, Array[Double])] = {
    val result: Array[(Int, Array[Double])] = new Array[(Int, Array[Double])](workSetIds.length)
    for(id <- 0 until(workSetIds.length)){
      result(id) = (workSetIds(id), arrayWorkset(workSetIds(id)).getLabels())
    }

    result
  }
}


class ArrayLabels[T](val arrayLabels: Array[Array[T]]){
  def length: Int = arrayLabels.length
  def numLabels: Int = {
    var result = 0
    for(id <- 0 until arrayLabels.length){
      result += arrayLabels(id).length
    }
    result
  }

  /**
    * this function must get exactly the same result as ArrayWorkSet.getRandomLabeledPartDataPoint given
    * the same seed.
    * @param random
    * @return
    */
  def getRandomLabel(random: Random): T = {
    var labelSetId: Int = random.nextInt(arrayLabels.length)
    while(arrayLabels(labelSetId).length == 0) // avoid empty set
      labelSetId = random.nextInt(arrayLabels.length)
    val tmpLabelSet = arrayLabels(labelSetId)
    tmpLabelSet(random.nextInt(tmpLabelSet.length))
  }

}