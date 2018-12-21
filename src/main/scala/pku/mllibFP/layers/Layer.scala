package pku.mllibFP.layers

import pku.mllibFP.dataModel.TwoDTensor

class Layer{
  def forward(inputX: TwoDTensor, inputY: TwoDTensor): TwoDTensor
  def backward()
}