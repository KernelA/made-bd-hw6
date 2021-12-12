package org.apache.spark.ml.made

import breeze.linalg.{DenseMatrix, DenseVector, pinv}

class LinearRegression() {
  private var _coeffs: DenseVector[Double] = DenseVector.ones[Double](1)

  def coeffs = _coeffs(0 until _coeffs.length - 1)

  def shift = _coeffs(-1)

  def fit(
           trainFeatures: DenseMatrix[Double],
           target: DenseVector[Double]
         ) = {
    val newFeatures =
      DenseMatrix.horzcat(trainFeatures, DenseMatrix.ones[Double](trainFeatures.rows, 1))
    this._coeffs = (pinv(newFeatures) * target.asDenseMatrix.t).toDenseVector
  }

  def predict(
               testFeatures: DenseMatrix[Double]
             ): DenseVector[Double] = {
    val newFeatures =
      DenseMatrix.horzcat(testFeatures, DenseMatrix.ones[Double](testFeatures.rows, 1))

    return (newFeatures * this._coeffs.asDenseMatrix.t).toDenseVector
  }
}
