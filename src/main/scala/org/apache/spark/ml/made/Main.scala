package org.apache.spark.ml.made

import breeze.linalg._
import com.typesafe.scalalogging.Logger
import org.apache.spark.sql.SparkSession



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
//
//  def readMatrix(filepath: String): DenseMatrix[Double] = {
//    return csvread(new File(filepath), skipLines = 1)
//  }

//  if (args.length != 8) {
//    println(
//      "Please specify arguments as --train file --test file --out path_to_out --true path_to_true_values"
//    )
//    java.lang.System.exit(1)
//  }

  var logger = Logger("linreg")

  val spark = SparkSession.builder
    .appName("Word count")
    .config("spark.driver.cores", 2)
    .config("spark.driver.memory", 6)
    .master("local[2]")
    .getOrCreate()

  val data = spark.sparkContext.parallelize(
    Seq("I like Spark", "Spark is awesome", "My first Spark job is working now and is counting these words"))

  val wordCounts = data
    .flatMap(row => row.split(" "))
    .map(word => (word, 1))
    .reduceByKey(_ + _)

  wordCounts.foreach(println)

  spark.stop()
}
