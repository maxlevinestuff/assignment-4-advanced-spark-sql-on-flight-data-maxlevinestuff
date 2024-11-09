from pyspark.sql import SparkSession
from pyspark.sql.functions import abs, col, expr
from pyspark.sql.window import Window
from pyspark.sql import functions as F

# Create a Spark session
spark = SparkSession.builder.appName("Advanced Flight Data Analysis").getOrCreate()

# Load datasets
flights_df = spark.read.csv("flights.csv", header=True, inferSchema=True)
airports_df = spark.read.csv("airports.csv", header=True, inferSchema=True)
carriers_df = spark.read.csv("carriers.csv", header=True, inferSchema=True)

# Define output paths
output_dir = "output/"
task1_output = output_dir + "task1_largest_discrepancy.csv"
task2_output = output_dir + "task2_consistent_airlines.csv"
task3_output = output_dir + "task3_canceled_routes.csv"
task4_output = output_dir + "task4_carrier_performance_time_of_day.csv"

# ------------------------
# Task 1: Flights with the Largest Discrepancy Between Scheduled and Actual Travel Time
# ------------------------
def task1_largest_discrepancy(flights_df, carriers_df):
    # TODO: Implement the SQL query for Task 1
    # Hint: Calculate scheduled vs actual travel time, then find the largest discrepancies using window functions.

    # Calculate scheduled travel time and actual travel time in minutes
    flights_df = flights_df.withColumn("ScheduledTravelTime", 
                                       (col("ScheduledArrival").cast("long") - col("ScheduledDeparture").cast("long")) / 60)
    flights_df = flights_df.withColumn("ActualTravelTime", 
                                       (col("ActualArrival").cast("long") - col("ActualDeparture").cast("long")) / 60)
    
    # Calculate the discrepancy as the absolute difference between scheduled and actual travel times
    flights_df = flights_df.withColumn("Discrepancy", 
                                       abs(col("ScheduledTravelTime") - col("ActualTravelTime")))
    
    # Define a window partitioned by CarrierCode to find the largest discrepancy within each carrier
    window_spec = Window.partitionBy("CarrierCode").orderBy(col("Discrepancy").desc())
    
    # Rank flights by the largest discrepancy for each carrier
    ranked_df = flights_df.withColumn("rank", expr("row_number()").over(window_spec)).filter(col("rank") == 1)
    
    # Join with carriers_df to get carrier names and resolve ambiguity
    largest_discrepancy = ranked_df.alias("flights") \
        .join(carriers_df.alias("carriers"), col("flights.CarrierCode") == col("carriers.CarrierCode"), "left") \
        .select("FlightNum", col("carriers.CarrierName"), "Origin", "Destination", 
                "ScheduledTravelTime", "ActualTravelTime", "Discrepancy", col("flights.CarrierCode"))

    # Write the result to a CSV file
    # Uncomment the line below after implementing the logic
    largest_discrepancy.write.csv(task1_output, header=True)
    print(f"Task 1 output written to {task1_output}")

# ------------------------
# Task 2: Most Consistently On-Time Airlines Using Standard Deviation
# ------------------------
def task2_consistent_airlines(flights_df, carriers_df):
    # TODO: Implement the SQL query for Task 2
    # Hint: Calculate standard deviation of departure delays, filter airlines with more than 100 flights.

    # Calculate the departure delay in minutes
    flights_df = flights_df.withColumn("DepartureDelay", 
                                       (F.col("ActualDeparture").cast("long") - F.col("ScheduledDeparture").cast("long")) / 60)
    
    # Group by CarrierCode to calculate the standard deviation of the departure delay and count of flights
    delay_stats = flights_df.groupBy("CarrierCode") \
        .agg(
            F.stddev("DepartureDelay").alias("StdDevDepartureDelay"),
            F.count("FlightNum").alias("NumFlights")
        ) \
        .filter(F.col("NumFlights") > 100)  # Filter to include only carriers with more than 100 flights
    
    # Join with carriers_df to get carrier names
    consistent_airlines = delay_stats.join(carriers_df, "CarrierCode", "left") \
        .select("CarrierName", "NumFlights", "StdDevDepartureDelay") \
        .orderBy("StdDevDepartureDelay")  # Order by smallest standard deviation for consistency

    # Write the result to a CSV file
    # Uncomment the line below after implementing the logic
    consistent_airlines.write.csv(task2_output, header=True)
    print(f"Task 2 output written to {task2_output}")

# ------------------------
# Task 3: Origin-Destination Pairs with the Highest Percentage of Canceled Flights
# ------------------------
def task3_canceled_routes(flights_df, airports_df):
    # TODO: Implement the SQL query for Task 3
    # Hint: Calculate cancellation rates for each route, then join with airports to get airport names.

    # Mark flights as canceled if ActualDeparture is null
    flights_df = flights_df.withColumn("IsCanceled", F.when(F.col("ActualDeparture").isNull(), 1).otherwise(0))

    # Calculate the cancellation rate for each origin-destination pair
    cancellation_stats = flights_df.groupBy("Origin", "Destination") \
        .agg(
            F.sum("IsCanceled").alias("NumCanceled"),
            F.count("FlightNum").alias("TotalFlights")
        ) \
        .withColumn("CancellationRate", (F.col("NumCanceled") / F.col("TotalFlights")) * 100)
    
    # Join with airports_df twice to get names and cities for origin and destination airports
    canceled_routes = cancellation_stats \
        .join(airports_df.alias("orig"), F.col("Origin") == F.col("orig.AirportCode"), "left") \
        .join(airports_df.alias("dest"), F.col("Destination") == F.col("dest.AirportCode"), "left") \
        .select(
            F.col("orig.AirportName").alias("OriginAirport"),
            F.col("orig.City").alias("OriginCity"),
            F.col("dest.AirportName").alias("DestinationAirport"),
            F.col("dest.City").alias("DestinationCity"),
            "CancellationRate"
        ) \
        .orderBy(F.desc("CancellationRate"))  # Order by highest cancellation rate

    # Write the result to a CSV file
    # Uncomment the line below after implementing the logic
    canceled_routes.write.csv(task3_output, header=True)
    print(f"Task 3 output written to {task3_output}")

# ------------------------
# Task 4: Carrier Performance Based on Time of Day
# ------------------------
def task4_carrier_performance_time_of_day(flights_df, carriers_df):
    # TODO: Implement the SQL query for Task 4
    # Hint: Create time of day groups and calculate average delay for each carrier within each group.

    # Step 1: Convert ScheduledDeparture to an hour column
    flights_df = flights_df.withColumn("ScheduledHour", F.hour("ScheduledDeparture"))

    # Step 2: Define time-of-day groups based on ScheduledHour
    flights_df = flights_df.withColumn(
        "TimeOfDay",
        F.when((F.col("ScheduledHour") >= 6) & (F.col("ScheduledHour") < 12), "Morning")
         .when((F.col("ScheduledHour") >= 12) & (F.col("ScheduledHour") < 18), "Afternoon")
         .when((F.col("ScheduledHour") >= 18) & (F.col("ScheduledHour") < 24), "Evening")
         .otherwise("Night")
    )

    # Step 3: Calculate the departure delay as the difference between ActualDeparture and ScheduledDeparture
    flights_df = flights_df.withColumn("DepartureDelay", F.col("ActualDeparture").cast("long") - F.col("ScheduledDeparture").cast("long"))

    # Step 4: Calculate average departure delay for each carrier and time of day
    avg_delay_df = flights_df.groupBy("CarrierCode", "TimeOfDay") \
        .agg(F.avg("DepartureDelay").alias("AvgDepartureDelay"))

    # Step 5: Join with carriers_df to get carrier names
    carrier_performance_time_of_day = avg_delay_df.join(carriers_df, "CarrierCode", "left") \
        .select("CarrierName", "TimeOfDay", "AvgDepartureDelay") \
        .orderBy("TimeOfDay", "AvgDepartureDelay")

    # Write the result to a CSV file
    # Uncomment the line below after implementing the logic
    carrier_performance_time_of_day.write.csv(task4_output, header=True)
    print(f"Task 4 output written to {task4_output}")

# ------------------------
# Call the functions for each task
# ------------------------
task1_largest_discrepancy(flights_df, carriers_df)
task2_consistent_airlines(flights_df, carriers_df)
task3_canceled_routes(flights_df, airports_df)
task4_carrier_performance_time_of_day(flights_df, carriers_df)

# Stop the Spark session
spark.stop()
