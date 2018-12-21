package pku.mllibFP.loss

abstract class Loss extends Layer{
  def compute(labels: Array[Int], dots: Array[Double]): Double
}