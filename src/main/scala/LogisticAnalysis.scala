import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.sql.{Row, SQLContext}
import org.apache.spark.sql.types._

/**
  * Created by Administrator on 2017/3/17.
  */
case
class LogisticAnalysis extends Serializable {

  val driveUrl = "jdbc:mysql://172.20.1.15:3306/statistics?" +
    "user=root&password=123456&zeroDateTimeBehavior=convertToNull&amp;characterEncoding=utf-8"

  val testDataSql = "select u.vehicle_id as obj_id,(YEAR('2017-01-01 00:00:00')-YEAR(u.birthday)) as age," +
    " case when u.sex=2 then 0 else u.sex end as sex, ds.jjiasu as jjiasu,ds.jjiansu as jjiansu,ds.jzhwan as jzhwan,ds.jshache as jshache," +
    " ds.avg_mileage as avg_mileage,ds.avg_time as avg_time,ds.night_time as night_time,ds.avg_speed as avg_speed,ds.avg_fuel as avg_fuel,ds.cxgl as cxgl" +
    " from user u join (select obj_id,avg(acc_num) as jjiasu,avg(dec_num) as jjiansu,avg(turn_num) as jzhwan,avg(brakes_num) as jshache," +
    " avg(round(mileage/1000,6)) as avg_mileage,avg(round(track_time/60,5)) as avg_time,avg(night_time) as night_time," +
    " avg(avg_speed) as avg_speed,avg(avg_fuel) as avg_fuel,avg(danger_probability) as cxgl" +
    " from driving_safety_index_daily group by obj_id) ds on u.vehicle_id=ds.obj_id" +
    " where (YEAR('2017-01-01 00:00:00')-YEAR(u.birthday))>=18"

  val load_trainData_sql = "select age,sex,jjiasu,jjiansu,jzhwan,jshache,avg_mileage," +
    " avg_time,night_time,avg_speed,avg_fuel,cxqk from analysis.logi_trainData"

  val insert2hive_sql = "insert overwrite table analysis.logis_res select obj_id,age,sex,jjiasu,jjiansu,jzhwan,jshache,avg_mileage," +
    " avg_time,night_time,avg_speed,avg_fuel,cxgl,cxgl_pre from results"

  val sourceData_schema = StructType(Array(
    StructField("jjiasu", DoubleType, false),
    StructField("jjiansu", DoubleType, false),
    StructField("jzhwan", DoubleType, false),
    StructField("jshache", DoubleType, false),
    StructField("avg_mileage", DoubleType, false),
    StructField("avg_time", DoubleType, false),
    StructField("night_time", DoubleType, false),
    StructField("avg_speed", DoubleType, false),
    StructField("avg_fuel", DoubleType, false),
    StructField("cxqk", IntegerType, false)
  ))


  val testData_schema = StructType(Array(
    StructField("jjiasu", DoubleType, false),
    StructField("jjiansu", DoubleType, false),
    StructField("jzhwan", DoubleType, false),
    StructField("jshache", DoubleType, false),
    StructField("avg_mileage", DoubleType, false),
    StructField("avg_time", DoubleType, false),
    StructField("night_time", DoubleType, false),
    StructField("avg_speed", DoubleType, false),
    StructField("avg_fuel", DoubleType, false)
  ))


  def analysis() = {

    Logger.getLogger("org.apache.spark").setLevel(Level.INFO)

    //设置运行环境
    val sparkConf = new SparkConf().setAppName("MovieLensLog").setMaster("local[2]")
    val sc = new SparkContext(sparkConf)
    val sqlContext = new SQLContext(sc)

    //1. 加载数据 17个数据


    val data = sc.textFile("d://sparktest/ubi/trainData.csv").map(_.split(",")).map {
      p => Row(p(0).toDouble, p(1).toDouble, p(2).toDouble, p(3).toDouble, p(4).toDouble, p(5).toDouble, p(6).toDouble, p(7).toDouble, p(8).toDouble, p(9).toInt)
    }


    val trainData = sqlContext.createDataFrame(data, sourceData_schema)
    trainData.registerTempTable("ceshi")
    sqlContext.sql("select * from ceshi limit 20 ").show();

    // 屏蔽不必要的日志显示在终端上
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)


    //将训练数据转换成labeled point
    val trainDataMap = trainData.map(r => {

      new LabeledPoint(r.get(9).toString.toInt, Vectors.dense(
        r.get(0).toString.toDouble,
        r.get(1).toString.toDouble,
        r.get(2).toString.toDouble,
        r.get(3).toString.toDouble,
        r.get(4).toString.toDouble,
        r.get(5).toString.toDouble,
        r.get(6).toString.toDouble,
        r.get(7).toString.toDouble,
        r.get(8).toString.toDouble
      ))

    }
    )


    val logisModel = new LogisticRegressionWithLBFGS()
      .setNumClasses(2)   //二分类
      .run(trainDataMap)

    //结果输出概率P
    /*logisModel.clearThreshold()   // 结果输出为概率 或者 0/1*/
    logisModel.clearThreshold()


    //加载测试数据
    val testData = sc.textFile("d://sparktest/ubi/testDataP.txt").map(_.split("\t")).map {
      p => Row(p(3).toDouble, p(4).toDouble, p(5).toDouble, p(6).toDouble, p(7).toDouble, p(8).toDouble, p(9).toDouble, p(10).toDouble, p(11).toDouble)
    }

    val testDataMap = sqlContext.createDataFrame(testData, testData_schema)
    testDataMap.registerTempTable("ceshidata")

    sqlContext.sql("select * from ceshidata limit 20 ").show();


    //对测试数据每一行进行预测
    val testData2Resul = testDataMap.map(r => {
      val lineVector = Vectors.dense(
        cast2Double(r.get(0)),
        cast2Double(r.get(1)),
        cast2Double(r.get(2)),
        cast2Double(r.get(3)),
        cast2Double(r.get(4)),
        cast2Double(r.get(5)),
        cast2Double(r.get(6)),
        cast2Double(r.get(7)),
        cast2Double(r.get(8))
      )

      val prediction = logisModel.predict(lineVector)


      Row(lineVector,prediction)

   /*   new LogisPrediction(r.getString(0), r.getInt(1), r.getInt(2), cast2Double(r.get(3)), cast2Double(r.get(4)), cast2Double(r.get(5)), cast2Double(r.get(6)),
        cast2Double(r.get(7)), cast2Double(r.get(8)), cast2Double(r.get(9)), cast2Double(r.get(10)), cast2Double(r.get(11)), cast2Double(r.get(12)), prediction)
*/
    })


    testData2Resul.take(10).foreach(x => {
      println(x)
    })





    /*testData2Resul.take(10).foreach(x=>{
      println(x)
    })*/


  }


  def cast2Double(x: Any): Double = {
    x match {
      case _: Int => x.toString.toDouble
      case _: String => x.toString.toDouble
      case _: BigDecimal => x.toString.toDouble
      case _: Double => x.toString.toDouble
      case _ => throw new RuntimeException("no match case!!!")
    }


  }

}


object LogisticAnalysis {


  def main(args: Array[String]): Unit = {

    val analysis = new LogisticAnalysis
    analysis.analysis()


  }

}
