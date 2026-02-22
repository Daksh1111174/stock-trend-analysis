# spark_processing.py

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, stddev, when
from pyspark.sql.window import Window

def create_spark_session():
    spark = SparkSession.builder \
        .appName("StockTrendAnalysis") \
        .getOrCreate()
    return spark

def process_stock_data(spark, pandas_df):
    """
    Convert Pandas DF → Spark DF
    Add technical indicators
    """

    spark_df = spark.createDataFrame(pandas_df)

    # Window specs
    window_20 = Window.orderBy("Date").rowsBetween(-20, 0)
    window_50 = Window.orderBy("Date").rowsBetween(-50, 0)

    # Moving Averages
    spark_df = spark_df.withColumn("MA20", avg("Close").over(window_20))
    spark_df = spark_df.withColumn("MA50", avg("Close").over(window_50))

    # Daily Return
    spark_df = spark_df.withColumn(
        "Daily_Return",
        (col("Close") - col("Open")) / col("Open")
    )

    # Volatility
    spark_df = spark_df.withColumn(
        "Volatility",
        stddev("Daily_Return").over(window_20)
    )

    # Trend Signal
    spark_df = spark_df.withColumn(
        "Trend",
        when(col("MA20") > col("MA50"), "Bullish")
        .otherwise("Bearish")
    )

    return spark_df.toPandas()
