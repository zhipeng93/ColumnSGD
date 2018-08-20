package pku.mllibFP.util


class ColumnMLException (cause: String) extends RuntimeException(cause){

}

class ColumnMLDenseVectorException extends ColumnMLException("Currently we don't support DenseVectors.")