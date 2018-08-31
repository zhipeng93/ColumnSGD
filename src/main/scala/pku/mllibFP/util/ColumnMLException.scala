package pku.mllibFP.util


class ColumnMLException (cause: String) extends Exception(cause){

}

class ColumnMLDenseVectorException extends ColumnMLException("Currently we don't support DenseVectors.")

class ColumnMLTaskException extends ColumnMLException("Spark Task failure")

class ColumnMLExecutorException extends ColumnMLException("Spark Executor lost failure")