// Databricks notebook source
val sqlContext = new org.apache.spark.sql.SQLContext(sc)

// COMMAND ----------

import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql._

// COMMAND ----------

////////////////////////////////////////////////links_Dataframe///////////////////////////////////////////////

// COMMAND ----------

val links_schema = StructType(Array(
StructField("movieId",IntegerType, true),
StructField("title", StringType, true),
StructField("genres", StringType, true),
))

// COMMAND ----------

case class links(movieId: Integer, title: String, genres: String) extends Serializable

// COMMAND ----------

val file_links ="/FileStore/tables/links.csv"

// COMMAND ----------

val df_link = spark.read.format("csv").option("inferSchema", "false").schema(links_schema).load(file_links).as[links]

// COMMAND ----------

display(df_link)

// COMMAND ----------

/////////////////////////////////////////////////Movies_Dataframe////////////////////////////////////////

// COMMAND ----------

val movies_schema = StructType(Array(
StructField("movieId",IntegerType, true),
StructField("title", StringType, true),
StructField("genres", StringType, true),
))

// COMMAND ----------

case class movies(movieId: Integer, title: String, genres: String) extends Serializable

// COMMAND ----------

val file_movies ="/FileStore/tables/movies.csv"

// COMMAND ----------

val df_movies = spark.read.format("csv").option("inferSchema", "false").schema(movies_schema).load(file_movies).as[movies]

// COMMAND ----------

display(df_movies)

// COMMAND ----------

////////////////////////////////////////////////Rating//////////////////////////////////////////////////

// COMMAND ----------

val rating_schema = StructType(Array(
StructField("userId",IntegerType, true),
StructField("movieId",IntegerType, true),
StructField("rating", FloatType, true),
StructField("timestamp", StringType, true),
))

// COMMAND ----------

case class rating(userId: Integer, movieId: Integer, rating: Float,timestamp:String ) extends Serializable

// COMMAND ----------

val file_ratings ="/FileStore/tables/ratings-1.csv"

// COMMAND ----------

val df_ratings = spark.read.format("csv").option("inferSchema", "false").schema(rating_schema).load(file_ratings).as[rating]

// COMMAND ----------

display(df_ratings)

// COMMAND ----------

////////////////////////////////////////////////Tags//////////////////////////////////////////////////

// COMMAND ----------

val tags_schema = StructType(Array(
StructField("userId",IntegerType, true),
StructField("movieId",IntegerType, true),
StructField("tag", StringType, true),
StructField("timestamp", IntegerType, true),
))

// COMMAND ----------

case class tags(userId: Integer, movieId: Integer, tag: String,timestamp:String ) extends Serializable

// COMMAND ----------

val file_tags ="/FileStore/tables/tags.csv"

// COMMAND ----------

val df_tags = spark.read.format("csv").option("inferSchema", "false").schema(tags_schema).load(file_tags).as[tags]

// COMMAND ----------

////////////////////////////////Lecture du contenu de "rating.csv" dans une sqlContext/////////////////////////

// COMMAND ----------

/////////////////// ALS Model//////////////////////////////////////////

// COMMAND ----------

import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.recommendation.ALS

// COMMAND ----------

def parseRating(str: String): rating = {
  val fields = str.split("::")
  assert(fields.size == 4)
  rating(fields(0).toInt, fields(1).toInt, fields(2).toFloat, fields(3).toString)
}


// COMMAND ----------

val Array(training, test) = df_ratings.randomSplit(Array(0.8, 0.2))

// COMMAND ----------

// Build the recommendation model using ALS on the training data
val als = new ALS()
  .setMaxIter(5)
  .setRegParam(0.01)
  .setUserCol("userId")
  .setItemCol("movieId")
  .setRatingCol("rating")
val model = als.fit(training)
model.setColdStartStrategy("drop")
val predictions = model.transform(test)
val evaluator = new RegressionEvaluator()
  .setMetricName("rmse")
  .setLabelCol("rating")
  .setPredictionCol("prediction")
val rmse = evaluator.evaluate(predictions)
println(s"Root-mean-square error = $rmse")

// COMMAND ----------

// Generate top 10 movie recommendations for each user
val userRecs = model.recommendForAllUsers(10)
// Generate top 10 user recommendations for each movie
val movieRecs = model.recommendForAllItems(10)

// COMMAND ----------

display(userRecs)

// COMMAND ----------

display(movieRecs)

// COMMAND ----------


