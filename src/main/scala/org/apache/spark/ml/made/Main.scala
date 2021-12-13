package org.apache.spark.ml.made

import breeze.linalg._
import breeze.stats.distributions.RandBasis
import com.typesafe.scalalogging.Logger
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.types.{StructField, StructType}
import org.apache.spark.sql.{DataFrame, SparkSession, functions}

object Constants {
  val NUM_CORES: Int = 2
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

  var logger = Logger("main")

  val spark = SparkSession.builder
    .appName("Linear regression")
    .master(s"local[${Constants.NUM_CORES}]")
    .getOrCreate()

  import spark.sqlContext.implicits._

  val generator = RandBasis.withSeed(124)

  val features = DenseMatrix.rand[Double](100000, 3, rand = generator.uniform)
  val trueCoeff = DenseVector(1.5, 0.3, -0.7).asDenseMatrix.t
  val target = (features * trueCoeff).toDenseVector

  val featuresWitTarget = Range(0, features.rows)
    .map(index => Tuple2(Vectors.fromBreeze(features(index, ::).t), target(index)))
    .toSeq
    .toDF("features", "label")

  var model = new LinearRegression()
    .setNumIter(5000)
    .setLearningRate(1)
    .setEps(1e-6)
    .setPrintEvery(100)
    .setOutputCol("target")
  var trainedModel = model.fit(featuresWitTarget)

  var predicted = trainedModel.transform(featuresWitTarget)

  predicted =
    predicted.withColumn("abs_diff", functions.abs(predicted("target") - predicted("label")))
  val maxAbsDiff = predicted.select(functions.max(predicted("abs_diff"))).first()

  logger.info(f"Max abs diff: ${maxAbsDiff}")

  trueCoeff.toDenseVector.mapPairs((index, value) =>
    logger.info(
      f"true_value_${index + 1}%-4d = ${value}%-18f est_value_${index + 1}%-4d = ${trainedModel.modelCoeff(index)}%-18f"
    )
  )
  logger.info(
    f"true_b = ${0.0}%-18f est_b = ${trainedModel.modelCoeff(trainedModel.modelCoeff.size - 1)}%-18f"
  )

  spark.stop()
}
