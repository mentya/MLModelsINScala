package com.naven.test.ml.scala.models
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.sql.SQLContext
import org.apache.spark.ml.feature.{ VectorAssembler, StringIndexer }
import org.apache.spark.sql.functions._
import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.classification.{RandomForestClassifier,RandomForestClassificationModel}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import com.naveen.test.utility.ModelUtility
object RFModel {
  val inputPath = "InputData/brestCancerData.csv"
  def main(args: Array[String]) {
    System.setProperty("hadoop.home.dir", "C:/winutils");
    val conf = new SparkConf().setAppName("LRBrestCancerPrediction").setMaster("local[4]")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)
    val ModelUtility = new ModelUtility()
    import sqlContext._
    import ModelUtility._
    // val inputFile=sqlContext.read.csv(inputPath)
    val inputDF = sqlContext.read.format("com.databricks.spark.csv").option("header", "True").load(inputPath)
    val addLabelToDF = inputDF.withColumn("clas", when(inputDF("diagnosis") === "M", 1.0).otherwise(0.0))
    addLabelToDF.registerTempTable("people")
    val parseLabelData = sqlContext.sql("select id,cast(clas as double),cast(radius_mean as double),cast(texture_mean as double),cast(perimeter_mean as double),cast(area_mean as double),cast(smoothness_mean as double),cast(compactness_mean as double),cast(concavity_mean as double),cast(concave_points_mean as double),cast(symmetry_mean as double),cast(fractal_dimension_mean  as double) from people")
    val featureCols = Array("radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean", "compactness_mean", "concavity_mean", "concave_points_mean", "symmetry_mean", "fractal_dimension_mean")
    val assembler = new VectorAssembler().setInputCols(featureCols).setOutputCol("features")
    val df2 = assembler.transform(parseLabelData)
    val labelIndexer = new StringIndexer().setInputCol("clas").setOutputCol("label")
    val df3 = labelIndexer.fit(df2).transform(df2)
    val splitSeed = 10
    val Array(trainingData, testData) = df3.randomSplit(Array(0.7, 0.3), splitSeed)
    val rf = new RandomForestClassifier().setImpurity("gini").setMaxDepth(3).setNumTrees(30).setFeatureSubsetStrategy("auto").setSeed(45)
    val model = rf.fit(trainingData)

    val predictions = model.transform(testData)
    val perform = getPerformance(predictions)
    println("accuracy-" + perform.apply("accuracy"))
    println("recal-" + perform.apply("recal"))
    println("precision-" + perform.apply("precision"))
    println("f1score-" + perform.apply("f1score"))
  }
}