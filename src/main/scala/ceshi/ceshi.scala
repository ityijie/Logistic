package ceshi

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by Administrator on 2017/3/23.
  */


case class TestIri(features: org.apache.spark.mllib.linalg.Vector, label: Int)

object ceshi {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org.apache.spark").setLevel(Level.INFO)


    //设置运行环境
    val sparkConf = new SparkConf().setAppName("MovieLensLog").setMaster("local[2]")
    val sc = new SparkContext(sparkConf)
    val sqlContext = new SQLContext(sc)


    //1. 加载数据 17个数据
    val data = sc.textFile("d://sparktest/ubi/trainDataP.csv").map(_.split(","))
      .map(p => TestIri(Vectors.dense(
        p(0).toDouble, p(1).toDouble, p(2).toDouble, p(3).toDouble,
        p(4).toDouble, p(5).toDouble, p(6).toDouble, p(7).toDouble, p(8).toDouble),
        p(9).toInt
      )
      )

    val dataDF = sqlContext.createDataFrame(data)

    val Array(trainingData, testData) = dataDF.randomSplit(Array(0.9, 0.1))

    trainingData.registerTempTable("trainingData");

    sqlContext.sql("select * from trainingData").show(10)

    testData.registerTempTable("testData");

    sqlContext.sql("select * from testData  ").show(10)


    val lr = new LogisticRegression()
      .setLabelCol("indexedLabel")
      .setFeaturesCol("indexedFeatures")
      .setMaxIter(20)
      .setRegParam(0.1)
      .setElasticNetParam(0.8)

    val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(trainingData)
    val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").fit(trainingData)
    val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)

    //塞数据 ,构建pipeline，设置stage，然后调用fit()来训练模型。
    val lrPipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, lr, labelConverter))
    val lrPipelineModel = lrPipeline.fit(trainingData)

    val lrPredictions = lrPipelineModel.transform(testData)

    lrPredictions.show(20)




  }


}
