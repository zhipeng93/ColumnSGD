object test{
  def main(args: Array[String]): Unit ={
    val a: Array[Array[Int]] = Array.ofDim(2,2)
    val b: Array[Array[Int]] = Array.ofDim(2,2)
    a(0)(0) = 1
    a(0)(1) = 2
    a(1)(0) = 3
    a(1)(1) = 4

    deepCopy(a,b)

    for(i <- 0 until(b.length)){
      for(j <- 0 until(b(0).length)){
        println(b(i)(j))
      }
    }

  }

  def deepCopy(srcArray: Array[Array[Int]], dstArray: Array[Array[Int]]): Unit ={
    for(i <- 0 until(srcArray.length)) {
      System.arraycopy(srcArray(i), 0, dstArray(i), 0, srcArray(i).length)
    }
  }
}