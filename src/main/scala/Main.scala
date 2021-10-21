import breeze.linalg._
import breeze.linalg.shuffle
import java.io.File
import com.typesafe.scalalogging.Logger
import scala.math
import javax.sound.sampled.Line
import breeze.stats.mode

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

class CrossVal(cv: Int) {

  def fit(trainFeatures: DenseMatrix[Double], target: DenseVector[Double]): LinearRegression = {

    var logger = Logger("Main")

    var indices = Range(0, trainFeatures.rows).toArray
    indices = shuffle(indices)

    val shuffledRows =
      indices.map(index => trainFeatures(index, ::).t.toArray).flatMap(_.toList).toArray

    val shuffledTarget = new DenseVector[Double](indices.map(index => target(index)))

    val newFeatures = DenseMatrix.create(
      trainFeatures.rows,
      trainFeatures.cols,
      shuffledRows,
      0,
      trainFeatures.cols,
      true
    )

    val blockSize = trainFeatures.rows / cv

    logger.info("Start cross validation")

    for (cvBlock <- 0 to cv - 1) {
      logger.info(f"Cross validation ${cvBlock + 1} of ${cv}")
      val startTestIndex = cvBlock * blockSize
      val endTestIndex = startTestIndex + blockSize

      val testFeatures = newFeatures(startTestIndex until endTestIndex, ::)
      val trainFeaturesCV = newFeatures.delete(startTestIndex until endTestIndex, Axis._0)
      val testTarget = shuffledTarget(startTestIndex until endTestIndex)
      val trainTarget = new DenseVector[Double](
        Range(0, shuffledTarget.length)
          .filter(index => index < startTestIndex || index >= endTestIndex)
          .map(index => shuffledTarget(index))
          .toArray
      )

      var regression = new LinearRegression()

      regression.fit(trainFeaturesCV, trainTarget)

      val predictedTarget = regression.predict(trainFeaturesCV)

      val trainError = Metrics.MSE(trainTarget, predictedTarget)
      logger.info(f"MSE error on train ${trainError}")

      val predictedTest = regression.predict(testFeatures)

      val testError = Metrics.MSE(testTarget, predictedTest)

      logger.info(f"MSE error on test ${testError}")

    }

    logger.info("Train on full data")
    var finalModel = new LinearRegression()

    finalModel.fit(trainFeatures, target)

    return finalModel;

  }
}

object FeaturePreprocessing {
  def splitFeatureTarget(
      featuresTarget: DenseMatrix[Double]
  ): (DenseMatrix[Double], DenseVector[Double]) = {
    return (featuresTarget(::, 0 until featuresTarget.cols - 1), featuresTarget(::, -1))
  }
}

object Metrics {
  def MSE(trueValues: DenseVector[Double], predictedValues: DenseVector[Double]): Double = {
    return math.sqrt((trueValues - predictedValues).map(x => x * x).sum)
  }
}

object Main extends App {
  def findOpt(args: Array[String], optName: String): Int = {
    val optIndex = args.indexOf(optName)
    if (optIndex == -1) {
      println(f"Cannot find required arg ${optName}")
      java.lang.System.exit(1)
    }

    return optIndex
  }

  def readMatrix(filepath: String): DenseMatrix[Double] = {
    return csvread(new File(filepath), skipLines = 1)
  }

  if (args.length != 8) {
    println(
      "Please specify arguments as --train file --test file --out path_to_out --true path_to_true_values"
    )
    java.lang.System.exit(1)
  }

  var logger = Logger("Main")

  val trainOptIndex = findOpt(args, "--train")
  val testOptIndex = findOpt(args, "--test")
  val outOptIndex = findOpt(args, "--out")
  val trueOptIndex = findOpt(args, "--true")

  val trainFeaturesTarget = readMatrix(args(trainOptIndex + 1))

  val (trainFeatures, trainTarget) = FeaturePreprocessing.splitFeatureTarget(trainFeaturesTarget)

  var crossValid = new CrossVal(4)

  val model = crossValid.fit(trainFeatures, trainTarget)

  val trueCoeffs = readMatrix(args(trueOptIndex + 1)).toDenseVector

  model.coeffs.mapPairs((index, value) =>
    logger.info(
      f"true_alpha_${index + 1}%-4d = ${trueCoeffs(index)}%-18f alpha_${index + 1}%-4d = ${value}%-18f"
    )
  )
  logger.info(f"true_b = ${trueCoeffs(-1)}%-18f b = ${model.shift}%-18f")

  val testFeatures = readMatrix(args(testOptIndex + 1))
  val predictedTest = model.predict(testFeatures)

  logger.info("Save prediction on test")
  csvwrite(new File(args(outOptIndex + 1)), predictedTest.asDenseMatrix.t)

}
