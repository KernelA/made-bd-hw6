package org.apache.spark.ml.made

import org.apache.spark.ml.linalg
import com.typesafe.scalalogging.Logger
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.{Pipeline}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.ml.feature.{HashingTF, IDF, RegexTokenizer}
import org.apache.spark.sql.functions
import org.apache.spark.sql.functions.{monotonically_increasing_id}

object Constants {
  val NUM_CORES: Int = 2
}

object Main extends App {
  var logger = Logger("main")

  val spark = SparkSession.builder
    .appName("RandomLSH")
    .master(s"local[${Constants.NUM_CORES}]")
    .getOrCreate()

  val numFeatures = 1000

  val preprocessingPipe = new Pipeline()
    .setStages(
      Array(
        new RegexTokenizer()
          .setInputCol("Review")
          .setOutputCol("tokenized")
          .setPattern("\\W+"),
        new HashingTF()
          .setInputCol("tokenized")
          .setOutputCol("tf")
          .setBinary(true)
          .setNumFeatures(numFeatures),
        new HashingTF()
          .setInputCol("tokenized")
          .setOutputCol("tf2")
          .setNumFeatures(numFeatures),
        new IDF()
          .setInputCol("tf2")
          .setOutputCol("tfidf")
      )
    )

  val dataset = spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .option("sep", ",")
    .csv("./data/tripadvisor_hotel_reviews.csv").sample(0.3)

  val trainSize = 0.8

  val Array(train, test) = dataset.randomSplit(Array(trainSize, 1 - trainSize), 0)

  val pipe = preprocessingPipe.fit(train)
  val trainFeatures = pipe.transform(train).cache()
  val testFeatures = pipe.transform(test)

  dataset.drop("Review")
  trainFeatures.drop("tokenized", "Review", "tf2", "tf")
  testFeatures.drop("tokenized", "Review", "tf2", "tf")

  val testFeaturesWithIndex = testFeatures.withColumn("id", monotonically_increasing_id()).cache()

  val metrics = new RegressionEvaluator()
    .setLabelCol("Rating")
    .setPredictionCol("predict")
    .setMetricName("rmse")

  val results = Seq(10, 20, 30).map(numHashes =>
    Seq(0.5, 0.7, 0.8).map(distance => {
      val cosineLSHModel = new CosineRandomHyperplanesLSH()
        .setInputCol("tfidf")
        .setOutputCol("hash")
        .setNumHashTables(numHashes)
        .setSeed(0)
        .fit(trainFeatures)

      val neighbors =
        cosineLSHModel.approxSimilarityJoin(trainFeatures, testFeaturesWithIndex, distance)

      val predictions = neighbors
        .withColumn("similarity", functions.col("distCol"))
        .groupBy("datasetB.id")
        .agg(
          (functions.sum(functions.col("similarity") * functions.col("datasetA.Rating")) / functions
            .sum(functions.col("similarity"))).as("predict"),
          functions.count("datasetA.Rating").as("numNeighbors")
        )

      val forMetric = testFeaturesWithIndex.join(predictions, Seq("id"))

      val meanNumNeighbors = forMetric.select(functions.avg("numNeighbors")).collect.head(0)

      logger.info("Compute mean neighbors")

      val metric = metrics.evaluate(forMetric)

      val res = (numHashes, metric, meanNumNeighbors)
      logger.info(
        s"Num hashes: ${numHashes} Distance: ${distance} RMSE: ${metric} Mean num neighbours: ${meanNumNeighbors}"
      )
      res
    })
  )

  spark.stop()
}
