package org.apache.spark.ml.made

import breeze.linalg.{DenseVector, sum}
import com.typesafe.scalalogging.Logger
import org.apache.spark.ml.attribute.AttributeGroup
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.{Estimator, Model}
import breeze.stats.distributions.Rand
import org.apache.spark.ml.linalg.{Vector, VectorUDT, Vectors}
import org.apache.spark.ml.param.{DoubleParam, IntParam, Param, ParamMap}
import org.apache.spark.ml.param.shared.{HasFeaturesCol, HasLabelCol, HasOutputCol}
import org.apache.spark.ml.util.{
  DefaultParamsReadable,
  DefaultParamsReader,
  DefaultParamsWritable,
  DefaultParamsWriter,
  Identifiable,
  MLReadable,
  MLReader,
  MLWritable,
  MLWriter,
  SchemaUtils
}
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.{DataFrame, Dataset, Encoder}
import org.apache.spark.sql.types.StructType

object LinearRegressionConstants {
  val RELATIVE_PATH = "/coeff"
}

trait LinearRegressionParams extends HasFeaturesCol with HasLabelCol with HasOutputCol {

  val learningRate = new DoubleParam(this, "learningRate", "Learning rate of the algorithm")

  val numIter = new IntParam(this, "numIter", "Number of iterations")

  val eps =
    new DoubleParam(this, "gradientNorm", "The squared norm of a gradient for stopping condition.")

  val printEvery = new IntParam(this, "printEvery", "Print progress at each n-th iteration")

  def getLearningRate(): Double = $(learningRate)

  def getNumIter(): Int = $(numIter)

  def getEps(): Double = $(eps)

  def getPrintEvery(): Int = $(printEvery)

  def setEps(value: Double): this.type = set(eps, value)

  def setPrintEvery(value: Int): this.type = set(printEvery, value)

  def setLearningRate(value: Double): this.type = set(learningRate, value)

  def setNumIter(value: Int): this.type = set(numIter, value)

  def setFeaturesCol(value: String): this.type = set(featuresCol, value)

  def setLabelCol(value: String): this.type = set(labelCol, value)

  def setOutputCol(value: String): this.type = set(outputCol, value)

  setDefault(learningRate -> 1e-3)

  setDefault(numIter -> 100)

  setDefault(eps -> 1e-4)

  setDefault(printEvery -> 20)

  protected def validateAndTransformSchema(schema: StructType): StructType = {
    SchemaUtils.checkNumericType(schema, getLabelCol)

    if (schema.fieldNames.contains($(outputCol))) {
      SchemaUtils.checkColumnType(schema, getOutputCol, new VectorUDT())
      schema
    } else {
      SchemaUtils.appendColumn(schema, schema(getFeaturesCol).copy(name = getOutputCol))
    }
  }
}

// modelCoeff is coefficient + shift as last value
class LinearRegressionModel private[made] (override val uid: String, val modelCoeff: Vector)
    extends Model[LinearRegressionModel]
    with LinearRegressionParams
    with MLWritable {

  private[made] def this(modelCoeff: Vector) =
    this(Identifiable.randomUID("linearRegressionModel"), modelCoeff)

  override def copy(extra: ParamMap): LinearRegressionModel =
    copyValues(new LinearRegressionModel(modelCoeff), extra)

  override def transform(dataset: Dataset[_]): DataFrame = {
    val breezeCoeff = modelCoeff.asBreeze.toDenseVector
    val shift = breezeCoeff(-1)
    val coeff = Vectors.fromBreeze(breezeCoeff(0 to breezeCoeff.size - 2))

    val transformUdf =
      dataset.sqlContext.udf.register(
        uid + "_transform",
        (x: Vector) => {
          x.dot(coeff) + shift
        }
      )

    dataset.withColumn($(outputCol), transformUdf(dataset($(featuresCol))))
  }

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

  override def write: MLWriter = new DefaultParamsWriter(this) {
    override protected def saveImpl(path: String): Unit = {
      super.saveImpl(path)
      val coeff = Tuple1(modelCoeff)
      sqlContext
        .createDataFrame(Seq(coeff))
        .write
        .parquet(path + LinearRegressionConstants.RELATIVE_PATH)
    }
  }
}

class LinearRegression(override val uid: String)
    extends Estimator[LinearRegressionModel]
    with LinearRegressionParams
    with DefaultParamsWritable {

  protected var logger = Logger("linreg")

  def this() = this(Identifiable.randomUID("linearRegression"))

  protected def computeMseGradient(
      featuresWithTarget: Vector,
      weights: DenseVector[Double],
      shift: Double
  ): (DenseVector[Double], Double) = {
    val breezeVector = featuresWithTarget.asBreeze
    val features = breezeVector(0 until breezeVector.length - 1)
    val target = breezeVector(-1)
    val residual = (features dot weights) + shift - target
    Tuple2(residual * features.toDenseVector, residual)
  }

  override def fit(dataset: Dataset[_]): LinearRegressionModel = {
    // Used to convert untyped dataframes to datasets with vectors
    implicit val encoder: Encoder[Vector] = ExpressionEncoder()

    val resCol = "featuresWithTarget"
    val assembler = new VectorAssembler()
      .setInputCols(Array($(featuresCol), $(labelCol)))
      .setOutputCol(resCol)

    val transformed = assembler.transform(dataset)
    val vectors: Dataset[Vector] = transformed.select(transformed(resCol)).as[Vector]

    // Target is the last value
    val modelDim: Int = AttributeGroup
      .fromStructField((dataset.schema($(featuresCol))))
      .numAttributes
      .getOrElse(vectors.first().size - 1)

    // Init [-1; 1]
    val weights = 2.0 * DenseVector.rand[Double](modelDim) - 1.0
    var shift: Double = 2.0 * Rand.uniform.draw() - 1.0

    val numIters = getNumIter()
    val learningRate = getLearningRate()
    val gradientNorm = getEps()
    val squaredGradientNormThreshold = gradientNorm * gradientNorm
    var iter = 0
    val printEvery = getPrintEvery()

    while (iter < numIters) {
      val distrGradientRdd = vectors.rdd

      val broadcastedWeights = distrGradientRdd.sparkContext.broadcast(weights)

      val (rddGradientParams, rddGradientShift) = distrGradientRdd
        .map(row => computeMseGradient(row, broadcastedWeights.value, shift))
        .reduce((gradientWithResidual, nextGradientWithResidual) =>
          Tuple2(
            gradientWithResidual._1 + nextGradientWithResidual._1,
            gradientWithResidual._2 + nextGradientWithResidual._2
          )
        )

      val totalSamples = distrGradientRdd.count()
      val step = learningRate / totalSamples
      weights -= step * rddGradientParams
      shift -= step * rddGradientShift

      val squaredGradientNorm = (sum(
        rddGradientParams.map(x => x * x)
      ) + rddGradientShift * rddGradientShift) / totalSamples

      if (iter % printEvery == 0) {
        logger.info(
          s"Iter ${iter} / ${numIters} Squared norm of the gradient: ${squaredGradientNorm}"
        )
      }

      if (squaredGradientNorm < squaredGradientNormThreshold) {
        logger.debug("Early stopping.")
        iter = numIters
      }
      iter += 1
    }

    var solution = DenseVector.tabulate[Double](modelDim + 1)(index =>
      if (index < modelDim) weights(index) else 0.0
    )
    solution(-1) = shift
    copyValues(new LinearRegressionModel(Vectors.fromBreeze(solution))).setParent(this)
  }

  override def copy(extra: ParamMap): Estimator[LinearRegressionModel] = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

}

object LinearRegression extends DefaultParamsReadable[LinearRegression]

object LinearRegressionModel extends MLReadable[LinearRegressionModel] {
  override def read: MLReader[LinearRegressionModel] = new MLReader[LinearRegressionModel] {

    override def load(path: String): LinearRegressionModel = {
      val metadata = DefaultParamsReader.loadMetadata(path, sc)

      val vectors = sqlContext.read.parquet(path + LinearRegressionConstants.RELATIVE_PATH)

      // Used to convert untyped dataframes to datasets with vectors
      implicit val encoder: Encoder[Vector] = ExpressionEncoder()

      val coeff = vectors.select(vectors("_1").as[Vector]).first()

      val model = new LinearRegressionModel(coeff)
      metadata.getAndSetParams(model)
      model
    }
  }
}
