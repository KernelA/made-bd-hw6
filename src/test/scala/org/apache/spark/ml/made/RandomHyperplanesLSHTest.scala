package org.apache.spark.ml.made

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.stats.distributions.RandBasis
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.{DataFrame, Encoder}
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should
import com.google.common.io.Files
import com.github.mrpowers.spark.fast.tests.DatasetComparer
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder

class RandomHyperplanesLSHTest
    extends AnyFlatSpec
    with should.Matchers
    with WithSpark
    with DatasetComparer {

  lazy val features: DataFrame = RandomHyperplanesLSHTest._featuresDt

  lazy val model: DenseVector[Double] = RandomHyperplanesLSHTest._model

  lazy val randNormals = RandomHyperplanesLSHTest._randNormals

  "Params" should "contains" in {

    val model = new RandomHyperplanesLSHModel(randNormals)
      .setNumBits(120)

    model.getNumBits() should be(120)
  }

  "Dataframe" should "contains" in {
    features.schema.fieldNames.exists(col => col == "features") should be(true)
  }

  "Model" should "compute hash values" in {
    val model: RandomHyperplanesLSHModel =
      new RandomHyperplanesLSHModel(randNormals)
        .setNumBits(randNormals.rows)

    val featureVector = features.first().getAs[Vector](0)
    val hashVector = model.hashFunction(featureVector)
    hashVector.length should be(1)
    hashVector(0).size should be(model.getNumBits())
  }

  "Model" should "compute hamming distance on hash values" in {
    val model: RandomHyperplanesLSHModel =
      new RandomHyperplanesLSHModel(randNormals).setNumBits(randNormals.rows)

    val size = 10
    val ones = Seq(Vectors.dense(Range(0, size).map(x => 1.0).toArray))
    val zeros = Seq(Vectors.zeros(size))
    val hammingDistance = model.hashDistance(ones, zeros)
    val totalSim = model.hashDistance(ones, ones)
    val totalSimZeros = model.hashDistance(zeros, zeros)

    hammingDistance should be(10)
    totalSim should be(0)
    totalSimZeros should be(0)
  }

  "Model" should "compute cosine distance" in {
    val model: RandomHyperplanesLSHModel =
      new RandomHyperplanesLSHModel(randNormals).setNumBits(randNormals.rows)

    val size = 10
    val ones = Vectors.dense(Range(0, size).map(x => 1.0).toArray)
    val zeros = Vectors.zeros(size)

    val cosineDist = model.keyDistance(ones, zeros)
    val eq = model.keyDistance(ones, ones)

    val delta = 1e-8

    eq should be(0.0 +- delta)
    cosineDist should be(1.0 +- delta)
  }

//  "Model" should "predict target" in {
//    val model: LinearRegressionModel =
//      new LinearRegressionModel(modelCoeff = Vectors.fromBreeze(DenseVector(1.0, 2.0, 1.0, -1.0)))
//        .setOutputCol("predicted")
//
//    val vectors: Array[Vector] =
//      model.transform(featuresWithTarget).collect().map(_.getAs[Vector](0))
//
//    vectors.length should be(100)
//  }

//  "Estimator" should "estimate parameters" in {
//    val estimator = new LinearRegression().setOutputCol("predicted").setNumIter(10)
//
//    val model = estimator.fit(featuresWithTarget)
//
//    model.isInstanceOf[LinearRegressionModel] should be(true)
//  }
//
//  "Estimator" should "work after re-read" in {
//
//    val pipeline = new Pipeline().setStages(
//      Array(
//        new LinearRegression().setOutputCol("predicted").setNumIter(10)
//      )
//    )
//
//    val tmpFolder = Files.createTempDir()
//
//    pipeline.write.overwrite().save(tmpFolder.getAbsolutePath)
//
//    val reRead = Pipeline.load(tmpFolder.getAbsolutePath)
//
//    val model = reRead.fit(featuresWithTarget).stages(0).asInstanceOf[LinearRegressionModel]
//
//    model.modelCoeff.size should be(4)
//  }
//
//  "Model" should "work after re-read" in {
//
//    val pipeline = new Pipeline().setStages(
//      Array(
//        new LinearRegression()
//          .setOutputCol("predicted")
//          .setNumIter(10)
//      )
//    )
//
//    val model = pipeline.fit(featuresWithTarget)
//
//    val tmpFolder = Files.createTempDir()
//
//    model.write.overwrite().save(tmpFolder.getAbsolutePath)
//
//    val reRead: PipelineModel = PipelineModel.load(tmpFolder.getAbsolutePath)
//
//    val transformed =
//      model.stages(0).asInstanceOf[LinearRegressionModel].transform(featuresWithTarget)
//    val transformedReRead =
//      reRead.stages(0).asInstanceOf[LinearRegressionModel].transform(featuresWithTarget)
//
//    assertSmallDatasetEquality(transformed, transformedReRead)
//  }

}

object RandomHyperplanesLSHTest extends WithSpark {
  val _generator = RandBasis.withSeed(124)

  lazy val _features = DenseMatrix.rand(100, 3, rand = _generator.uniform)

  lazy val _randNormals = (2.0 * DenseMatrix.rand[Double](10, 3, rand = _generator.uniform) - 1.0)

  lazy val _model = DenseVector[Double](1, 3, 4)

  lazy val _featuresDt: DataFrame = {
    import sqlc.implicits._

    Range(0, _features.rows)
      .map(index => Tuple1(Vectors.fromBreeze(_features(index, ::).t)))
      .toDF("features")
  }
}
