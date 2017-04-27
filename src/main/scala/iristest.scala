import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by Administrator on 2017/3/22.
  */


case class Iri(features: org.apache.spark.mllib.linalg.Vector, label: Int)
object Iritest {

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org.apache.spark").setLevel(Level.INFO)

    //设置运行环境
    val sparkConf = new SparkConf().setAppName("MovieLensLog").setMaster("local[2]")
    val sc = new SparkContext(sparkConf)
    val sqlContext = new SQLContext(sc)

    //1. 加载数据 17个数据
    val data = sc.textFile("d://sparktest/iris/iris.csv").map(_.split(","))
      .map(p => Iri(Vectors.dense(
        p(0).toDouble, p(1).toDouble, p(2).toDouble, p(3).toDouble),
        p(4).toInt
         )
      )

    val dataDF = sqlContext.createDataFrame(data)

    //dataDF.show();

    //2.过滤数据--test
    dataDF.registerTempTable("iris");


    val df = sqlContext.sql("select * from iris where label != 1")

    df.map(t => t(1) + ":" + t(0)).collect().foreach(println)


    //3. 构建ML的pipeline
    //3.1 修改列名

    val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(dataDF)
    val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").fit(dataDF)

    //3.2 切分数据集
    val Array(trainingData, testData) = df.randomSplit(Array(0.7, 0.3))


    val lr = new LogisticRegression()
      .setLabelCol("indexedLabel")
      .setFeaturesCol("indexedFeatures")
      .setMaxIter(20)
      .setRegParam(0.3)
      .setElasticNetParam(0.8)

    //解释参数
    println("LogisticRegression parameters:\n" + lr.explainParams() + "\n")


    //这里我们设置一个labelConverter，目的是把预测的类别重新转化成字符型的。
    val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)


    //塞数据 ,构建pipeline，设置stage，然后调用fit()来训练模型。
    val lrPipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, lr, labelConverter))

    val lrPipelineModel = lrPipeline.fit(trainingData)

    //pipeline本质上是一个Estimator，当pipeline调用fit()的时候就产生了一个PipelineModel，
    // 本质上是一个Transformer。然后这个PipelineModel就可以调用transform()来进行预测，生成一个新的DataFrame，
    // 即利用训练得到的模型对测试集进行验证。
    val lrPredictions = lrPipelineModel.transform(testData)
    lrPredictions.show(10)

    //打印预测的列
    //​ 最后我们可以输出预测的结果，其中select选择要输出的列，collect获取所有行的数据，用foreach把每行打印出来。
    // 其中打印出来的值依次分别代表该行数据的真实分类和特征值、预测属于不同分类的概率、预测的分类。
//    lrPredictions.select("predictedLabel", "label", "features", "probability").collect().foreach {
//      println
//    }

    //4. 模型评估
    // 创建一个MulticlassClassificationEvaluator实例，用setter方法把预测分类的列名和真实分类的列名进行设置；然后计算预测准确率和错误率。
    val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction")
    val lrAccuracy = evaluator.evaluate(lrPredictions)
    println("Test Error = " + (1.0 - lrAccuracy))


  }

}
