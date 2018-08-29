package pku.mllibFP.util

import org.apache.spark.mllib.linalg.{SparseVector, Vector}

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
  */
abstract class WorkSet extends Serializable {
  def getLabeledPartDataPoint(index: Int): LabeledPartDataPoint
  def getNumDataPoints(): Int
  def getLabels(start: Int, end: Int): Array[Double]
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

  override def getLabels(start: Int, end: Int): Array[Double] = {
    val result = new Array[Double](end - start)
    System.arraycopy(labels, start, result, 0, end - start)
    result
  }

  /**
    * group the worksets into one in order.
    * !!! Not tested.
    */
  def group(others: Array[CSRWorkSet]): WorkSet = {
    // combine this and other WorkSets
    var dataNum = 0
    var featureNum = 0
    for(wid <- 0 until others.length){
      dataNum += others(wid).labels.length
      featureNum += others(wid).indices.length
    }
    val index_pointers_t = new Array[Int](dataNum)
    val labels_t = new Array[Double](dataNum)
    val indices_t = new Array[Int](featureNum)
    val values_t = new Array[Double](featureNum)
    // copy all of the worksets to the tmp values

    // update index pointer
    var next_label_id = 0 // the global id of the first elements in the next workset
    var next_index_id = 0 // the global id of the first element in the next workset
    for(wid <- 0 until(others.length)){
      val tmp_workset = others(wid)
      System.arraycopy(tmp_workset.labels, 0, labels_t, next_label_id, tmp_workset.labels.length)
      System.arraycopy(tmp_workset.indices, 0, indices_t, next_index_id, tmp_workset.indices.length)
      System.arraycopy(tmp_workset.values, 0, values_t, next_index_id, tmp_workset.values.length)

      for(ip <- 0 until(tmp_workset.index_pointers.length)){
        index_pointers_t(next_label_id + ip) = next_index_id + tmp_workset.index_pointers(ip)
      }

      next_label_id += tmp_workset.labels.length
      next_index_id += tmp_workset.indices.length
    }

    new CSRWorkSet(index_pointers_t, labels_t, indices_t, values_t)

  }
}


class ArrayWorkSet(xarray: Array[LabeledPartDataPoint]) extends WorkSet{
  override def getLabeledPartDataPoint(index: Int): LabeledPartDataPoint = {
    xarray(index)
  }

  override def getNumDataPoints(): Int = xarray.length

  override def getLabels(start: Int, end: Int): Array[Double] = {
    val result = new Array[Double](end - start)
    for(id <- 0 until(end - start)){
      result(id) = getLabeledPartDataPoint(start + id).label
    }
    result
  }
}