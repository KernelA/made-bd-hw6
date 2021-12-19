package org.apache.spark.ml.made

import scala.util.Random
import breeze.linalg.{DenseMatrix, DenseVector, SparseVector}
import com.typesafe.scalalogging.Logger
import org.apache.spark.ml.feature.{LSH, LSHModel, LSHParams}
import org.apache.spark.ml.linalg.{Vector, VectorUDT, Vectors}
import org.apache.spark.ml.param.{IntParam, LongParam, ParamMap, ParamValidators}
import org.apache.spark.ml.param.shared.{
  HasFeaturesCol,
  HasInputCol,
  HasLabelCol,
  HasOutputCol,
  HasSeed
}
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
import org.apache.spark.sql.types.{DataTypes, StructType}

object RandomHyperplanesLSHPConstants {
  val RELATIVE_PATH = "/coeff"
  val EPS = 1e-9
}

trait CosineRandomHyperplanesLSHParams extends LSHParams {
  val seed: LongParam = new LongParam(this, "seed", "A random seed", ParamValidators.gtEq(0L))

  def setSeed(value: Long): this.type = set(seed, value)

  def getSeed: Long = $(seed)

  setDefault(seed -> 123L)

  protected def customValidateAndTransformSchema(schema: StructType): StructType = {
    if (schema.fieldNames.contains($(outputCol))) {
      SchemaUtils.checkColumnType(schema, getOutputCol, new VectorUDT())
      schema
    } else {
      SchemaUtils.appendColumn(schema, $(outputCol), DataTypes.createArrayType(new VectorUDT))
    }
  }
}

class CosineRandomHyperplanesLSHModel private[made] (
    override val uid: String,
    val randNormals: DenseMatrix[Double]
) extends LSHModel[CosineRandomHyperplanesLSHModel]
    with MLWritable
    with CosineRandomHyperplanesLSHParams {

  private[made] def this(randNormals: DenseMatrix[Double]) =
    this(Identifiable.randomUID("cosineRandomHyperplanesLSHModel"), randNormals)

  override def copy(extra: ParamMap): CosineRandomHyperplanesLSHModel =
    copyValues(new CosineRandomHyperplanesLSHModel(randNormals), extra)

  override def hashFunction(elems: Vector): Array[Vector] = {
    val binHash = (randNormals * elems.asBreeze.toDenseVector.asDenseMatrix.t).map(value =>
      if (value > 0) 1.0 else 0.0
    )
    Range(0, randNormals.rows).map(index => Vectors.fromBreeze(binHash(index, ::).t)).toArray
  }

  override def hashDistance(x: Seq[Vector], y: Seq[Vector]): Double = {
    // hamming distance
    x.iterator
      .zip(y.iterator)
      .map(vectorPair =>
        vectorPair._1.toArray.zip(vectorPair._2.toArray).count(pair => pair._1 != pair._2)
      )
      .sum
  }

  override def keyDistance(x: Vector, y: Vector): Double = {
    val norm1 = math.max(Vectors.norm(x, 2), RandomHyperplanesLSHPConstants.EPS)
    val norm2 = math.max(Vectors.norm(y, 2), RandomHyperplanesLSHPConstants.EPS)

    1.0 - x.dot(y) / (norm1 * norm2)
  }

  override def transformSchema(schema: StructType): StructType = customValidateAndTransformSchema(
    schema
  )

  override def write: MLWriter = new DefaultParamsWriter(this) {
    override protected def saveImpl(path: String): Unit = {
      super.saveImpl(path)

      val alignedNormals = Range(0, randNormals.rows)
        .map(index => Tuple1(Vectors.fromBreeze(randNormals(index, ::).t)))

      sqlContext
        .createDataFrame(alignedNormals)
        .write
        .parquet(path + RandomHyperplanesLSHPConstants.RELATIVE_PATH)
    }
  }

}

class CosineRandomHyperplanesLSH(override val uid: String)
    extends LSH[CosineRandomHyperplanesLSHModel]
    with CosineRandomHyperplanesLSHParams
    with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("cosineRandomHyperPlaneLSH"))

  override def setNumHashTables(value: Int): this.type = super.setNumHashTables(value)

  override def copy(extra: ParamMap): CosineRandomHyperplanesLSH = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = customValidateAndTransformSchema(
    schema
  )

  override def createRawLSHModel(inputDim: Int): CosineRandomHyperplanesLSHModel = {
    val rand = new Random(getSeed)

    val randHyperPlanes = Array
      .fill($(numHashTables)) {
        Array.fill(inputDim)({ if (rand.nextDouble() > 0.5) 1.0 else -1.0 })
      }
      .flatten

    val randNormals = DenseMatrix.create(
      $(numHashTables),
      inputDim,
      randHyperPlanes,
      0,
      inputDim,
      isTranspose = true
    )
    new CosineRandomHyperplanesLSHModel(randNormals)
  }
}

object CosineRandomHyperplanesLSH extends DefaultParamsReadable[CosineRandomHyperplanesLSH]

object CosineRandomHyperplanesLSHModel extends MLReadable[CosineRandomHyperplanesLSHModel] {
  override def read: MLReader[CosineRandomHyperplanesLSHModel] =
    new MLReader[CosineRandomHyperplanesLSHModel] {

      override def load(path: String): CosineRandomHyperplanesLSHModel = {
        val metadata = DefaultParamsReader.loadMetadata(path, sc)

        val normalsDt = sqlContext.read.parquet(path + RandomHyperplanesLSHPConstants.RELATIVE_PATH)

        // Used to convert untyped dataframes to datasets with vectors
        implicit val encoder: Encoder[Vector] = ExpressionEncoder()

        val rows = normalsDt.count().toInt

        val randNormals =
          normalsDt.select(normalsDt("_1").as[Vector]).collect().map(vector => vector.toArray)

        val cols = randNormals(0).size

        val matrix =
          DenseMatrix.create[Double](rows, cols, randNormals.flatten, 0, cols, isTranspose = true)
        val model = new CosineRandomHyperplanesLSHModel(matrix)
        metadata.getAndSetParams(model)
        model
      }
    }
}

//class LinearRegression(override val uid: String)
//    extends Estimator[LinearRegressionModel]
//    with LinearRegressionParams
//    with DefaultParamsWritable {
//
//  protected var logger = Logger("linreg")
//
//  def this() = this(Identifiable.randomUID("linearRegression"))
//
//  protected def computeMseGradient(
//      featuresWithTarget: Vector,
//      weights: DenseVector[Double],
//      shift: Double
//  ): (DenseVector[Double], Double) = {
//    val breezeVector = featuresWithTarget.asBreeze
//    val features = breezeVector(0 until breezeVector.length - 1)
//    val target = breezeVector(-1)
//    val residual = (features dot weights) + shift - target
//    Tuple2(residual * features.toDenseVector, residual)
//  }
//
//  override def fit(dataset: Dataset[_]): LinearRegressionModel = {
//    // Used to convert untyped dataframes to datasets with vectors
//    implicit val encoder: Encoder[Vector] = ExpressionEncoder()
//
//    val resCol = "featuresWithTarget"
//    val assembler = new VectorAssembler()
//      .setInputCols(Array($(featuresCol), $(labelCol)))
//      .setOutputCol(resCol)
//
//    val transformed = assembler.transform(dataset)
//    val vectors: Dataset[Vector] = transformed.select(transformed(resCol)).as[Vector]
//
//    // Target is the last value
//    val modelDim: Int = AttributeGroup
//      .fromStructField((dataset.schema($(featuresCol))))
//      .numAttributes
//      .getOrElse(vectors.first().size - 1)
//
//    // Init [-1; 1]
//    val weights = 2.0 * DenseVector.rand[Double](modelDim) - 1.0
//    var shift: Double = 2.0 * Rand.uniform.draw() - 1.0
//
//    val numIters = getNumIter()
//    val learningRate = getLearningRate()
//    val gradientNorm = getEps()
//    val squaredGradientNormThreshold = gradientNorm * gradientNorm
//    var iter = 0
//    val printEvery = getPrintEvery()
//
//    while (iter < numIters) {
//      val distrGradientRdd = vectors.rdd
//
//      val broadcastedWeights = distrGradientRdd.sparkContext.broadcast(weights)
//
//      val (rddGradientParams, rddGradientShift) = distrGradientRdd
//        .map(row => computeMseGradient(row, broadcastedWeights.value, shift))
//        .reduce((gradientWithResidual, nextGradientWithResidual) =>
//          Tuple2(
//            gradientWithResidual._1 + nextGradientWithResidual._1,
//            gradientWithResidual._2 + nextGradientWithResidual._2
//          )
//        )
//
//      val totalSamples = distrGradientRdd.count()
//      val step = learningRate / totalSamples
//      weights -= step * rddGradientParams
//      shift -= step * rddGradientShift
//
//      val squaredGradientNorm = (sum(
//        rddGradientParams.map(x => x * x)
//      ) + rddGradientShift * rddGradientShift) / totalSamples
//
//      if (iter % printEvery == 0) {
//        logger.info(
//          s"Iter ${iter} / ${numIters} Squared norm of the gradient: ${squaredGradientNorm}"
//        )
//      }
//
//      if (squaredGradientNorm < squaredGradientNormThreshold) {
//        logger.debug("Early stopping.")
//        iter = numIters
//      }
//      iter += 1
//    }
//
//    var solution = DenseVector.tabulate[Double](modelDim + 1)(index =>
//      if (index < modelDim) weights(index) else 0.0
//    )
//    solution(-1) = shift
//    copyValues(new LinearRegressionModel(Vectors.fromBreeze(solution))).setParent(this)
//  }
//
//  override def copy(extra: ParamMap): Estimator[LinearRegressionModel] = defaultCopy(extra)
//
//  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)
//
//}
//
//object LinearRegression extends DefaultParamsReadable[LinearRegression]
//
//object LinearRegressionModel extends MLReadable[LinearRegressionModel] {
//  override def read: MLReader[LinearRegressionModel] = new MLReader[LinearRegressionModel] {
//
//    override def load(path: String): LinearRegressionModel = {
//      val metadata = DefaultParamsReader.loadMetadata(path, sc)
//
//      val vectors = sqlContext.read.parquet(path + LinearRegressionConstants.RELATIVE_PATH)
//
//      // Used to convert untyped dataframes to datasets with vectors
//      implicit val encoder: Encoder[Vector] = ExpressionEncoder()
//
//      val coeff = vectors.select(vectors("_1").as[Vector]).first()
//
//      val model = new LinearRegressionModel(coeff)
//      metadata.getAndSetParams(model)
//      model
//    }
//  }
//}
