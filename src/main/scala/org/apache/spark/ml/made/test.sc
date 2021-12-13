import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.stats.distributions.RandBasis
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.made.Constants
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.{DataFrame, SparkSession}

val spark = SparkSession.builder
  .appName("Linera regression")
  .master(s"local[${Constants.NUM_CORES}]")
  .getOrCreate()

import spark.sqlContext.implicits._

val generator = RandBasis.withSeed(124)

val features = DenseMatrix.rand[Double](10000, 3, rand=generator.uniform)

val trueCoeff = DenseVector(1.5, 0.3, -0.7).asDenseMatrix.t

val target = (features * trueCoeff).toDenseVector

(features(1, ::) dot trueCoeff.toDenseVector.t) == target(1)

val featuresWitTarget = Range(0, features.rows).map(index => Tuple2(Vectors.fromBreeze(features(index, ::).t), target(index))).toSeq.toDF("features", "label")

