package org.apache.spark.ml.made

import breeze.linalg.{DenseMatrix, DenseVector, sum}
import com.typesafe.scalalogging.Logger
import org.apache.spark.ml.attribute.AttributeGroup
import org.apache.spark.ml.feature.{LSH, LSHModel, VectorAssembler}
import org.apache.spark.ml.{Estimator, Model}
import breeze.stats.distributions.{Rand, RandBasis}
import org.apache.spark.ml.linalg.{Vector, VectorUDT, Vectors}
import org.apache.spark.ml.param.{DoubleParam, IntParam, Param, ParamMap}
import org.apache.spark.ml.param.shared.{HasFeaturesCol, HasLabelCol, HasOutputCol, HasSeed}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsReader, DefaultParamsWritable, DefaultParamsWriter, Identifiable, MLReadable, MLReader, MLWritable, MLWriter, SchemaUtils}
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.{DataFrame, Dataset, Encoder}
import org.apache.spark.sql.types.StructType

object RandomHyperplanesLSHPConstants {
  val RELATIVE_PATH = "/coeff"
  val EPS = 1e-9
}

trait RandomHyperplanesLSHParams extends HasFeaturesCol {

  val numBits = new IntParam(this, "learningRate", "The number of bits in the hash")

  def getNumBits(): Int = $(numBits)

  def setNumBits(value: Int): this.type = set(numBits, value)

  def setFeaturesCol(value: String): this.type = set(featuresCol, value)

}

class RandomHyperplanesLSHModel private[made] (
    override val uid: String,
    val randNormals: DenseMatrix[Double]
) extends LSHModel[RandomHyperplanesLSHModel]
    with RandomHyperplanesLSHParams
    with MLWritable {

  private[made] def this(randNormals: DenseMatrix[Double]) =
    this(Identifiable.randomUID("randomHyperplanesLSHModel"), randNormals)

  override def copy(extra: ParamMap): RandomHyperplanesLSHModel =
    copyValues(new RandomHyperplanesLSHModel(randNormals), extra)

  //  override def hashFunction(elems: Vector): Array[Vector] = {
//    elems.
//  }

  override def hashFunction(elems: Vector): Array[Vector] = {
    val binHash = (randNormals * elems.asBreeze.toDenseVector.asDenseMatrix.t)
      .map(value => if (value > 0) 1.0 else 0.0)
      .toDenseVector
    Array(Vectors.fromBreeze(binHash))
  }

  override def hashDistance(x: Seq[Vector], y: Seq[Vector]): Double = {
    // hamming distance
    x.iterator
      .zip(y.iterator)
      .map(vectorPair =>
        vectorPair._1.toArray.zip(vectorPair._2.toArray).count(pair => pair._1 != pair._2)
      )
      .max
  }

  override def keyDistance(x: Vector, y: Vector): Double = {
    val norm1 = math.max(Vectors.norm(x, 2), RandomHyperplanesLSHPConstants.EPS)
    val norm2 = math.max(Vectors.norm(y, 2), RandomHyperplanesLSHPConstants.EPS)

    1.0 - x.dot(y) / (norm1 * norm2)
  }

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

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

class RandomHyperplanesLSH(override  val uid: String) extends  LSH with RandomHyperplanesLSHParams with DefaultParamsWritable {
  def this() = this(Identifiable.randomUID("randomHyperPlaneLSH"))

    override def copy(extra: ParamMap): LSH[RandomHyperplanesLSH] = defaultCopy(extra)

    override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

    override def createRawLSHModel(inputDim: Int): RandomHyperplanesLSHModel {
      require(inputDim <= MinHashLSH.HASH_PRIME,
      s"The input vector dimension $inputDim exceeds the threshold ${MinHashLSH.HASH_PRIME}.")
      val rand = new Random($(seed))
      val randCoefs: Array[(Int, Int)] = Array.fill($(numHashTables)) {
      (1 + rand.nextInt(MinHashLSH.HASH_PRIME - 1), rand.nextInt(MinHashLSH.HASH_PRIME - 1))
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
