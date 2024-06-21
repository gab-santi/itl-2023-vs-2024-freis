# Databricks notebook source
# MAGIC %md
# MAGIC # Data Engineering Project: ITL Online Scores Comparison
# MAGIC
# MAGIC ## Overview
# MAGIC This project aims to compare my scores from two different years of the ITL Online competition: 2023 and 2024. Each year's scores are stored in separate JSON files, with details for each song.
# MAGIC
# MAGIC ## Data Structure
# MAGIC
# MAGIC ### JSON Files (stored in DBFS)
# MAGIC - `/FileStore/ITL_Online_2023___Export.json`: Contains scores for each song in 2023.
# MAGIC - `/FileStore/ITL_Online_2024___Export.json`: Contains scores for each song in 2024.
# MAGIC
# MAGIC ## Analysis Steps
# MAGIC
# MAGIC 1. **Data Extraction**
# MAGIC    - Extract chart data from both JSON files.
# MAGIC    - Extract score data from both JSON files.
# MAGIC
# MAGIC 2. **Data Cleaning**
# MAGIC    - Ensure consistency in scoring metrics and point totals across both files. This can be cross checked with my own stats.
# MAGIC
# MAGIC 3. **Data Comparison**
# MAGIC    - Compare scores for each song from 2023 to 2024.
# MAGIC    - Identify metrics for comparing year-on-year performance.
# MAGIC    - Identify improvements or declines in performance.
# MAGIC
# MAGIC 4. **Visualization**
# MAGIC    - Create visual representations of the metric comparisons for easier analysis.
# MAGIC
# MAGIC 5. **Conclusion**
# MAGIC    - Summarize findings and insights.
# MAGIC
# MAGIC ## Tools and Libraries
# MAGIC - JSON parsing library (e.g., `json` in Python)
# MAGIC - Data analysis library (e.g., `pandas` in Python)
# MAGIC - Visualization library (e.g., `matplotlib` or `seaborn` in Python)

# COMMAND ----------

# Import packages
import json
import pandas as pd
import seaborn as sns

# COMMAND ----------

# Data Extraction & Cleaning
# Create functions for getting charts and scores tables
def load_json_and_get_charts_exploded(filename):
    df = spark.read.option("multiline", True).json(filename)

    charts_exploded = df.selectExpr(
        "explode(charts) as chart"
    )

    return charts_exploded

def get_charts_df_from_json(filename):
    charts_exploded = load_json_and_get_charts_exploded(filename)

    charts = charts_exploded.selectExpr(
        "chart.id AS chart_id"
        ,"chart.points"
        ,"chart.playstyle"
        ,"chart.title"
        ,"chart.subtitle"
        ,"chart.artist"
        ,"chart.pack"
        ,"chart.stepartist AS step_artist"
        ,"chart.techDescription AS tech_description"
        ,"chart.difficulty"
        ,"chart.meter AS level"
        ,"chart.minBpm AS min_bpm"
        ,"chart.maxBpm AS max_bpm"
        ,"chart.isNoCmod AS is_no_cmod"
    )

    charts.printSchema()
    return charts

def get_scores_df_from_json(filename):
    charts_exploded = load_json_and_get_charts_exploded(filename)

    """
    best_scores = charts_exploded.selectExpr(
        "chart.id AS chart_id"
        ,"chart.topScore.dateAdded AS date_added"
        ,"chart.topScore.clearType AS clear_type"
        ,"chart.topScore.fantasticPlus AS fantastic_plus"
        ,"chart.topScore.ex / 100 AS ex"
        ,"chart.topScore.points"
        ,"TRUE as is_best_score"
    )
    """

    scores_exploded = charts_exploded.selectExpr(
        "chart.id AS chart_id",
        "explode(chart.scores) as score"
    )

    scores = scores_exploded.selectExpr(
        "chart_id"
        ,"score.dateAdded AS date_added"
        ,"score.clearType AS clear_type"
        ,"score.fantasticPlus AS fantastic_plus"
        ,"score.ex / 100 AS ex"
        ,"score.points"
        # ,"FALSE as is_best_score"
    )

    scores = scores.withColumn("score_date", F.to_date("date_added", "yyyy-MM-dd'T'HH:mm:ss.SSS'Z'"))

    # scores = best_scores.union(scores)
    scores.printSchema()

    return scores

def get_best_scores_df_from_json(filename):
    charts_exploded = load_json_and_get_charts_exploded(filename)

    best_scores = charts_exploded.selectExpr(
        "chart.id AS chart_id"
        ,"chart.topScore.dateAdded AS date_added"
        ,"chart.topScore.clearType AS clear_type"
        ,"chart.topScore.fantasticPlus AS fantastic_plus"
        ,"chart.topScore.ex / 100 AS ex"
        ,"chart.topScore.points"
        # ,"TRUE as is_best_score"
    )

    best_scores = best_scores.where(best_scores.ex.isNotNull())

    best_scores = best_scores.withColumn("score_date", F.to_date("date_added", "yyyy-MM-dd'T'HH:mm:ss.SSS'Z'"))

    best_scores.printSchema()

    return best_scores

# COMMAND ----------

# Data Extraction
# 1. Load 2023 charts and scores data
# filter to remove post-ITL scores
charts_2023 = get_charts_df_from_json("dbfs:/FileStore/ITL_Online_2023___Export.json")

scores_2023 = get_scores_df_from_json("dbfs:/FileStore/ITL_Online_2023___Export.json")
scores_2023 = scores_2023.filter(scores_2023.score_date <= "2023-06-19")

best_scores_2023 = get_best_scores_df_from_json("dbfs:/FileStore/ITL_Online_2023___Export.json")
best_scores_2023 = best_scores_2023.filter(best_scores_2023.score_date <= "2023-06-19")


# COMMAND ----------

# 2. Load 2024 charts and scores data
charts_2024 = get_charts_df_from_json("dbfs:/FileStore/ITL_Online_2024___Export.json")

scores_2024 = get_scores_df_from_json("dbfs:/FileStore/ITL_Online_2024___Export.json")

best_scores_2024 = get_best_scores_df_from_json("dbfs:/FileStore/ITL_Online_2024___Export.json")

# COMMAND ----------

# Data checks
# ensure max EX is correct
# for 2023, it should be 99.95
scores_2023.selectExpr("max(ex)").show()

# COMMAND ----------

# for 2024, it should be 100
scores_2024.selectExpr("max(ex)").show()

# COMMAND ----------

# Check TP
# for 2023, it should be 1286852
best_scores_2023.selectExpr("sum(points)").show() 

# COMMAND ----------

# for 2024, it should be 1360370 (subtracted EX pyramid)
best_scores_2024.selectExpr("sum(points)").show()

# COMMAND ----------

# MAGIC %md
# MAGIC # Data Comparison
# MAGIC Identified metrics:
# MAGIC 1. Highest EX score attained accross all songs (with song info)
# MAGIC 2. Top 3 EX scores attained per difficulty (7-15)
# MAGIC 3. Top 10 plays (by points)
# MAGIC 4. Song Points increase from 2023 to 2024
# MAGIC 5. Number of sessions played accross the event
# MAGIC 6. Total number of FA+ steps submitted
# MAGIC
# MAGIC Additional Viz:
# MAGIC 1. Visualization of Song Points progression over time

# COMMAND ----------

# 1. Highest EX score attained accross all songs (with song info)
def get_highest_ex_score_with_song_info(scores_df, charts_df):
    max_ex_score = scores_df.selectExpr("max(ex) as max_ex").collect()[0]["max_ex"]
    df = scores_df.join(charts_df, 'chart_id', 'left').where(scores_df.ex == max_ex_score).select(charts_df.level, charts_df.difficulty, charts_df.title, charts_df.points, scores_df.points, scores_df.ex, scores_df.date_added)

    return df

# COMMAND ----------

highest_ex_score_2023 = get_highest_ex_score_with_song_info(scores_2023, charts_2023)

highest_ex_score_2024 = get_highest_ex_score_with_song_info(scores_2024, charts_2024)

# COMMAND ----------

highest_ex_score_2023.show()

# COMMAND ----------

highest_ex_score_2024.show()

# COMMAND ----------

# 2. Top 3 EX scores attained per level
from pyspark.sql.window import Window
from pyspark.sql.functions import col, desc, rank

def get_top_3_ex_per_level(best_scores_df, charts_df):
    best_scores_with_chart_info_df = best_scores_df.join(charts_df, "chart_id", "left").select(charts_df.level, charts_df.difficulty, charts_df.title, charts_df.points.alias("max_points"), best_scores_df.points, best_scores_df.ex, best_scores_df.date_added)

    window_spec = Window.partitionBy("level").orderBy(desc("ex"))

    scores_with_rank = best_scores_with_chart_info_df.withColumn("rank", rank().over(window_spec))

    top_3_df = scores_with_rank.filter(scores_with_rank.rank <= 3)

    return top_3_df

# COMMAND ----------

top_3_per_level_2023 = get_top_3_ex_per_level(best_scores_2023, charts_2023)

top_3_per_level_2024 = get_top_3_ex_per_level(best_scores_2024, charts_2024)

# COMMAND ----------

top_3_per_level_2023.show(top_3_per_level_2023.count())

# COMMAND ----------

top_3_per_level_2024.show(top_3_per_level_2024.count())

# COMMAND ----------

# 3. Top 10 plays
def get_top_10_plays(best_scores_df, charts_df):
    best_scores_with_chart_info_df = best_scores_df.join(charts_df, "chart_id", "left").select(charts_df.level, charts_df.difficulty, charts_df.title, charts_df.points.alias("max_points"), best_scores_df.points, best_scores_df.ex, best_scores_df.date_added)

    df = best_scores_with_chart_info_df.sort("points", ascending=False).limit(10)

    return df

# COMMAND ----------

top_10_plays_2023 = get_top_10_plays(best_scores_2023, charts_2023)
top_10_plays_2024 = get_top_10_plays(best_scores_2024, charts_2024)

# COMMAND ----------

top_10_plays_2023.show()

# COMMAND ----------

top_10_plays_2024.show()

# COMMAND ----------

# 4. SP difference
def get_sp(best_scores_df):
    return best_scores_df.sort("points", ascending=False).limit(75).selectExpr("sum(points) as sum_points").collect()[0]["sum_points"]

# COMMAND ----------

sp_2023 = get_sp(best_scores_2023)
sp_2024 = get_sp(best_scores_2024)

# COMMAND ----------

print("2023 Song Points: ", sp_2023) # Should be 478255
print("2024 Song Points: ", sp_2024) # Should be 518144
print("Difference: ", sp_2024-sp_2023)

# COMMAND ----------

from pyspark.sql import functions as F

# 5. Number of sessions played (session = 1 day)
def get_session_count(scores_df):
    df = scores_df.groupBy("score_date").count()

    return df.count()

# COMMAND ----------

session_count_2023 = get_session_count(scores_2023)
session_count_2024 = get_session_count(scores_2024)

# COMMAND ----------

print("Number of sessions played in #ITLOnline2023: ", session_count_2023) # Should be 34
print("Number of sessions played in #ITLOnline2024: ", session_count_2024) # Should be 30

# COMMAND ----------

# 6. Total number of FA+ steps submitted
def get_fa_plus_steps(scores_df):
    df = scores_df.selectExpr("sum(fantastic_plus) as fa_plus_steps")

    fa_plus_steps = df.collect()[0]["fa_plus_steps"]

    return fa_plus_steps

# COMMAND ----------

fa_plus_steps_2023 = get_fa_plus_steps(scores_2023)
fa_plus_steps_2024 = get_fa_plus_steps(scores_2024)

# COMMAND ----------

print("Blue Fantastic steps submitted in #ITLOnline2023: ", fa_plus_steps_2023)
print("Blue Fantastic steps submitted in #ITLOnline2024: ", fa_plus_steps_2024)
print("Difference: ", fa_plus_steps_2024-fa_plus_steps_2023)

# COMMAND ----------

print("Number of scores submitted in #ITLOnline2023: ", scores_2023.count())
print("Number of scores submitted in #ITLOnline2024: ", scores_2024.count())

# COMMAND ----------

# Viz
# SP progression over time
# create table first, with columns: date, SP total
def get_best_scores_for_date(scores_df, date):
    scores_date_df = scores_df.select("score_date", "chart_id", "points")

    scores_filtered_df = scores_date_df.filter(scores_date_df.score_date <= date)

    best_scores_per_chart_df = scores_filtered_df.groupBy("chart_id").agg({'points': 'max'}).withColumnRenamed("max(points)", "points")

    return best_scores_per_chart_df

def get_running_sp_for_date(scores_df, date):
    best_scores_per_chart_df = get_best_scores_for_date(scores_df, date)

    window_spec = Window.orderBy(best_scores_per_chart_df.points.desc())

    ranked_df = best_scores_per_chart_df.withColumn("rank", rank().over(window_spec))

    running_sp_df = ranked_df.filter(ranked_df.rank <= 75)

    running_sp = running_sp_df.selectExpr("sum(points) as sum_points").collect()[0]["sum_points"]

    return running_sp

def create_sp_progression_df(scores_df):
    # for each date, get running SP and store in dataframe
    dates_played = scores_df.select("score_date").groupBy("score_date").count().select("score_date").orderBy("score_date", ascending=True).rdd.flatMap(lambda x: x).collect()

    sp = []

    for date in dates_played:
        sp_at_d = get_running_sp_for_date(scores_df, date)
        sp.append(sp_at_d)

    data = {
        "date_played": dates_played,
        "running_sp": sp
    }

    return data

# COMMAND ----------

sp_progression_2023_df = create_sp_progression_df(scores_2023)

# COMMAND ----------

sp_progression_2024_df = create_sp_progression_df(scores_2024)

# COMMAND ----------

# Plot
# DEBUG fix SP drop issue
sns.lineplot(data=sp_progression_2023_df, x="date_played", y="running_sp")

# COMMAND ----------

sns.lineplot(data=sp_progression_2024_df, x="date_played", y="running_sp")

# COMMAND ----------

sp_progression_2023_df

# COMMAND ----------

sp_progression_2024_df
