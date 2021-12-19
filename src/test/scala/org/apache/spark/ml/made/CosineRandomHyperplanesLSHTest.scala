package org.apache.spark.ml.made

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.stats.distributions.RandBasis
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.{DataFrame, Encoder}
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should
import com.google.common.io.Files
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import com.github.mrpowers.spark.fast.tests.DatasetComparer


class CosineRandomHyperplanesLSHTest
    extends AnyFlatSpec
    with should.Matchers
    with WithSpark with DatasetComparer  {

  lazy val features: DataFrame = CosineRandomHyperplanesLSHTest._featuresDt

  lazy val model: DenseVector[Double] = CosineRandomHyperplanesLSHTest._model

  lazy val randNormals = CosineRandomHyperplanesLSHTest._randNormals


  "Dataframe" should "contains" in {
    features.schema.fieldNames.exists(col => col == "features") should be(true)
  }

  "Model" should "compute hash values" in {
    val model: CosineRandomHyperplanesLSHModel =
      new CosineRandomHyperplanesLSHModel(randNormals)

    val featureVector = features.first().getAs[Vector](0)
    val hashVector = model.hashFunction(featureVector)
    hashVector.length should be(model.randNormals.rows)
    hashVector(0).size should be(1)
  }

  "Model" should "compute hamming distance on hash values" in {
    val model: CosineRandomHyperplanesLSHModel =
      new CosineRandomHyperplanesLSHModel(randNormals)

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
    val model: CosineRandomHyperplanesLSHModel =
      new CosineRandomHyperplanesLSHModel(randNormals)

    val size = 10
    val ones = Vectors.dense(Range(0, size).map(x => 1.0).toArray)
    val zeros = Vectors.zeros(size)

    val cosineDist = model.keyDistance(ones, zeros)
    val eq = model.keyDistance(ones, ones)

    val delta = 1e-8

    eq should be(0.0 +- delta)
    cosineDist should be(1.0 +- delta)
  }

  "Model" should "compute hash" in {
    val model: CosineRandomHyperplanesLSHModel =
      new CosineRandomHyperplanesLSHModel(randNormals).setInputCol("features")
        .setOutputCol("hash")

    val vectors: Array[Vector] =
      model.transform(features).collect().map(_.getAs[Vector](0))

    vectors(0).size should be(randNormals.cols)
  }

  "Estimator" should "should parameters" in {
    val estimator = new CosineRandomHyperplanesLSH().setNumHashTables(10).setInputCol("features").setOutputCol("hash")
    val model = estimator.fit(features)
    model.isInstanceOf[CosineRandomHyperplanesLSHModel] should be(true)
  }

  "Model" should "work after re-read" in {

    val numHashTables = 10
    val pipeline = new Pipeline().setStages(
      Array(
        new CosineRandomHyperplanesLSH().setInputCol("features").setOutputCol("hash").setNumHashTables(numHashTables)
      )
    )

    val tmpFolder = Files.createTempDir()

    pipeline.write.overwrite().save(tmpFolder.getAbsolutePath)

    val reRead = Pipeline.load(tmpFolder.getAbsolutePath)

    val model = reRead.fit(features).stages(0).asInstanceOf[CosineRandomHyperplanesLSHModel]

    model.randNormals.rows should be(numHashTables)
  }

  "Estimator" should "work after re-read" in {

    val numHashTables = 10

    val pipeline = new Pipeline().setStages(
      Array(
        new CosineRandomHyperplanesLSH()
          .setInputCol("features").setOutputCol("hash").setNumHashTables(numHashTables)
      )
    )

    val model = pipeline.fit(features)

    val tmpFolder = Files.createTempDir()

    model.write.overwrite().save(tmpFolder.getAbsolutePath)

    val reRead: PipelineModel = PipelineModel.load(tmpFolder.getAbsolutePath)

    val transformed =
      model.stages(0).asInstanceOf[CosineRandomHyperplanesLSHModel].transform(features)
    val transformedReRead =
      reRead.stages(0).asInstanceOf[CosineRandomHyperplanesLSHModel].transform(features)

    assertSmallDatasetEquality(transformed, transformedReRead)
  }

  "Estimator" should "find nearest" in {
    val estimator = new CosineRandomHyperplanesLSH().setNumHashTables(10).setInputCol("features").setOutputCol("hash")
    val model = estimator.fit(features)
    val key = features.select(features("features")).first().getAs[Vector](0)

    val nnNeigh = model.approxNearestNeighbors(features, key, 3)
  }
}

object CosineRandomHyperplanesLSHTest extends WithSpark {
  val _generator = RandBasis.withSeed(124)

  lazy val _features = DenseMatrix.rand(50, 3, rand = _generator.uniform)

  lazy val _randNormals = (2.0 * DenseMatrix.rand[Double](10, 3, rand = _generator.uniform) - 1.0)

  lazy val _model = DenseVector[Double](1, 3, 4)

  lazy val _featuresDt: DataFrame = {
    import sqlc.implicits._

    Range(0, _features.rows)
      .map(index => Tuple1(Vectors.fromBreeze(_features(index, ::).t)))
      .toDF("features")
  }
}
