package org.apache.spark.ml.made

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.stats.distributions.RandBasis
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.DataFrame
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should
import com.google.common.io.Files
import com.github.mrpowers.spark.fast.tests.DatasetComparer

class LinearRegressionTest
    extends AnyFlatSpec
    with should.Matchers
    with WithSpark
    with DatasetComparer {
  lazy val featuresWithTarget: DataFrame = LinearRegressionTest._featuresDt

  lazy val model: DenseVector[Double] = LinearRegressionTest._model

  "Params" should "contains" in {
    var model = new LinearRegression()
      .setNumIter(10)
      .setLearningRate(0.2)
      .setLabelCol("label")
      .setNumIter(100)

    model.getNumIter() should be(100)
    model.getLearningRate() should be(0.2)
    model.getLabelCol should be("label")
  }

  "Dataframe" should "contains" in {
    featuresWithTarget.schema.fieldNames.exists(col => col == "features") should be(true)
    featuresWithTarget.schema.fieldNames.exists(col => col == "label") should be(true)
  }

  "Model" should "predict target" in {
    val model: LinearRegressionModel =
      new LinearRegressionModel(modelCoeff = Vectors.fromBreeze(DenseVector(1.0, 2.0, 1.0, -1.0)))
        .setOutputCol("predicted")

    val vectors: Array[Vector] =
      model.transform(featuresWithTarget).collect().map(_.getAs[Vector](0))

    vectors.length should be(100)
  }

  "Estimator" should "estimate parameters" in {
    val estimator = new LinearRegression().setOutputCol("predicted").setNumIter(10)

    val model = estimator.fit(featuresWithTarget)

    model.isInstanceOf[LinearRegressionModel] should be(true)
  }

  "Estimator" should "work after re-read" in {

    val pipeline = new Pipeline().setStages(
      Array(
        new LinearRegression().setOutputCol("predicted").setNumIter(10)
      )
    )

    val tmpFolder = Files.createTempDir()

    pipeline.write.overwrite().save(tmpFolder.getAbsolutePath)

    val reRead = Pipeline.load(tmpFolder.getAbsolutePath)

    val model = reRead.fit(featuresWithTarget).stages(0).asInstanceOf[LinearRegressionModel]

    model.modelCoeff.size should be(4)
  }

  "Model" should "work after re-read" in {

    val pipeline = new Pipeline().setStages(
      Array(
        new LinearRegression()
          .setOutputCol("predicted")
          .setNumIter(10)
      )
    )

    val model = pipeline.fit(featuresWithTarget)

    val tmpFolder = Files.createTempDir()

    model.write.overwrite().save(tmpFolder.getAbsolutePath)

    val reRead: PipelineModel = PipelineModel.load(tmpFolder.getAbsolutePath)

    val transformed =
      model.stages(0).asInstanceOf[LinearRegressionModel].transform(featuresWithTarget)
    val transformedReRead =
      reRead.stages(0).asInstanceOf[LinearRegressionModel].transform(featuresWithTarget)

    assertSmallDatasetEquality(transformed, transformedReRead)
  }

}

object LinearRegressionTest extends WithSpark {
  val _generator = RandBasis.withSeed(124)

  lazy val _features = DenseMatrix.rand(100, 3, rand = _generator.uniform)

  lazy val _model = DenseVector[Double](1, 3, 4)

  lazy val _target = (_features * _model.asDenseMatrix.t).toDenseVector

  lazy val _featuresDt: DataFrame = {
    import sqlc.implicits._

    Range(0, _features.rows)
      .map(index => Tuple2(Vectors.fromBreeze(_features(index, ::).t), _target(index)))
      .toSeq
      .toDF("features", "label")
  }
}
