//package pku.mllibFP.classfication
//
//import org.apache.spark.rdd.RDD
//import pku.mllibFP.util.LabeledPartDataPoint
//
//
//abstract class DimKFPModel(@transient inputRDD: RDD[Array[LabeledPartDataPoint]],
//                           numFeatures: Int,
//                           numPartitions: Int,
//                           regParam: Double,
//                           stepSize: Double,
//                           numIterations: Int,
//                           miniBatchSize: Int,
//                           modelK: Int) extends BaseFPModel[Array[Double]](inputRDD, numFeatures, numPartitions,
//  regParam, stepSize, numIterations, miniBatchSize, modelK) {
//  // directly give the parameters to the parent class.
//
//  override labels = new Array[Double](5)
//
//}
