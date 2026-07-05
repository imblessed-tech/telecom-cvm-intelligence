import logging
from datetime import datetime
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import DoubleType, IntegerType

# Set up logging
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger("spark_feature_pipeline")

# File paths
BASE_DIR = Path(__file__).resolve().parent.parent
IBM_PATH = str(BASE_DIR / "data" / "raw" / "telco_churn.csv")
EVENTS_PATH = str(BASE_DIR / "data" / "raw" / "customer_events.csv")
OUTPUT_PATH = str(BASE_DIR / "data" / "processed" / "spark_features")

def create_spark_session() -> SparkSession:
    """Create and configure SparkSession for local execution."""
    spark = SparkSession.builder \
        .master("local[*]") \
        .appName("CVM_Feature_Pipeline") \
        .config("spark.sql.shuffle.partitions", "8") \
        .config("spark.driver.memory", "2g") \
        .getOrCreate()
        
    spark.sparkContext.setLogLevel("WARN")
    return spark

def load_data(spark: SparkSession, ibm_path: str, events_path: str) -> tuple:
    """Load IBM churn and synthetic customer events datasets."""
    logger.warning(f"Loading raw datasets from {ibm_path} and {events_path}...")
    ibm_df = spark.read.csv(ibm_path, header=True, inferSchema=True)
    events_df = spark.read.csv(events_path, header=True, inferSchema=True)
    return ibm_df, events_df

def clean_ibm_data(df) -> SparkSession:
    """Clean IBM churn dataset."""
    # Cast TotalCharges to float and fill nulls with 0
    cleaned_df = df.withColumn("TotalCharges", F.col("TotalCharges").cast("float")) \
                   .fillna({"TotalCharges": 0.0})
                   
    # Convert Churn to binary churn_label
    cleaned_df = cleaned_df.withColumn(
        "churn_label", 
        F.when(F.col("Churn") == "Yes", 1).otherwise(0)
    )
    return cleaned_df

def compute_event_aggregations(events_df) -> SparkSession:
    """Aggregate customer event transactions using Spark Window functions."""
    # Ensure event_date is cast to date type
    events_df = events_df.withColumn("event_date", F.to_date(F.col("event_date")))
    
    # Define reference date as the maximum date in the dataset
    reference_date_row = events_df.select(F.max("event_date")).first()
    if reference_date_row and reference_date_row[0]:
        reference_date = reference_date_row[0]
    else:
        reference_date = datetime.today().date()
        
    logger.warning(f"Using reference date for aggregations: {reference_date}")
    
    # 1. Recharge aggregations (filter to RECHARGE events)
    recharge_events = events_df.filter(F.col("event_type") == "RECHARGE")
    recharge_df = recharge_events.groupBy("customer_id").agg(
        F.count("*").alias("recharge_count_90d"),
        F.sum("recharge_amount").alias("total_recharge_90d"),
        F.avg("recharge_amount").alias("avg_recharge_amount"),
        F.max("event_date").alias("last_recharge_date")
    )
    
    # Compute days since last recharge
    recharge_df = recharge_df.withColumn(
        "days_since_last_recharge",
        F.datediff(F.lit(reference_date), F.col("last_recharge_date"))
    ).drop("last_recharge_date")
    
    # 2. Data usage aggregations (filter to DATA_SESSION)
    data_sessions = events_df.filter(F.col("event_type") == "DATA_SESSION")
    data_df = data_sessions.groupBy("customer_id").agg(
        F.sum("data_mb_used").alias("data_usage_mb_90d"),
        F.avg("data_mb_used").alias("avg_session_mb")
    )
    
    # 3. Overall activity aggregations
    activity_df = events_df.groupBy("customer_id").agg(
        F.countDistinct("event_date").alias("active_days_90d")
    )
    
    # Join all aggregations together on customer_id
    aggregations_df = activity_df.join(recharge_df, "customer_id", "left") \
                                 .join(data_df, "customer_id", "left")
    return aggregations_df

def compute_recharge_trend(events_df) -> SparkSession:
    """Compute slope of recharge amount over time per customer using linear regression slope formula."""
    recharge_events = events_df.filter(F.col("event_type") == "RECHARGE")
    
    # Step 1 - Order recharges chronologically per customer and number them
    w_ordered = Window.partitionBy("customer_id").orderBy("event_date")
    recharge_events = recharge_events.withColumn("day_number", F.row_number().over(w_ordered))
    
    # Step 2 - Compute linear regression slope trend: (n*Σxy - Σx*Σy) / (n*Σx² - (Σx)²)
    trend_df = recharge_events.groupBy("customer_id").agg(
        (
            (F.count("*") * F.sum(F.col("day_number") * F.col("recharge_amount")) -
             F.sum("day_number") * F.sum("recharge_amount")) /
            (F.count("*") * F.sum(F.col("day_number") * F.col("day_number")) -
             F.sum("day_number") * F.sum("day_number"))
        ).alias("recharge_trend")
    )
    return trend_df

def join_and_save(ibm_df, aggregations_df, output_path: str) -> None:
    """Left join cleaned customer demographic data with event aggregations and save to CSV."""
    logger.warning("Joining datasets...")
    result = ibm_df.join(aggregations_df, ibm_df.customerID == aggregations_df.customer_id, "left")
    
    # Fill null values for event-based columns
    fill_cols = {
        "recharge_count_90d": 0,
        "total_recharge_90d": 0.0,
        "avg_recharge_amount": 0.0,
        "days_since_last_recharge": 90,  # Max window length if no recharges
        "data_usage_mb_90d": 0.0,
        "avg_session_mb": 0.0,
        "active_days_90d": 0,
        "recharge_trend": 0.0
    }
    result = result.fillna(fill_cols)
    
    logger.warning(f"Saving features to Spark CSV output path: {output_path}...")
    result.write.csv(output_path, header=True, mode="overwrite")
    
    row_count = result.count()
    logger.warning(f"PySpark Pipeline completed successfully. Output count: {row_count} rows.")
    logger.warning("Schema of generated Spark DataFrame:")
    result.printSchema()

if __name__ == "__main__":
    logger.warning("Starting PySpark CVM Feature Engineering Pipeline...")
    spark = create_spark_session()
    
    try:
        ibm_df, events_df = load_data(spark, IBM_PATH, EVENTS_PATH)
        ibm_clean = clean_ibm_data(ibm_df)
        aggregations = compute_event_aggregations(events_df)
        trend = compute_recharge_trend(events_df)
        
        agg_with_trend = aggregations.join(trend, "customer_id", "left")
        join_and_save(ibm_clean, agg_with_trend, OUTPUT_PATH)
        
    except Exception as e:
        logger.error(f"Error in PySpark execution: {e}", exc_info=True)
    finally:
        spark.stop()
        logger.warning("SparkSession stopped.")
    print("PySpark pipeline complete.")
