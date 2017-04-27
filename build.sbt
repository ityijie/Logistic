name := "spark2"

version := "1.0"

scalaVersion := "2.10.4"

// https://mvnrepository.com/artifact/org.apache.spark/spark-core_2.10
libraryDependencies += "org.apache.spark" % "spark-core_2.10" % "1.6.1"

// https://mvnrepository.com/artifact/org.apache.spark/spark-mllib_2.10
libraryDependencies += "org.apache.spark" % "spark-mllib_2.10" % "1.6.1"

// https://mvnrepository.com/artifact/log4j/log4j
libraryDependencies += "log4j" % "log4j" % "1.2.17"

libraryDependencies += "com.databricks" % "spark-csv_2.10" % "1.4.0"


resolvers += "repo2" at "http://repo2.maven.org/maven2/"