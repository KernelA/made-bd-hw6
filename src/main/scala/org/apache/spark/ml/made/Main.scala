package org.apache.spark.ml.made

import breeze.linalg._
import breeze.stats.distributions.RandBasis
import com.google.common.io.Files
import com.typesafe.scalalogging.Logger
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.feature.{BucketedRandomProjectionLSH, MinHashLSH}
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
    .appName("RandomLSH")
    .master(s"local[${Constants.NUM_CORES}]")
    .getOrCreate()

  import spark.sqlContext.implicits._

  

  spark.stop()
}
