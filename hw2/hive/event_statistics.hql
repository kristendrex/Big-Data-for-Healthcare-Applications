-- ***************************************************************************
-- Put events.csv and mortality.csv under hdfs directory 
-- sudo su - hdfs
-- hdfs dfs -mkdir -p /input/events
-- hdfs dfs -chown -R root /input
-- exit 
-- hdfs dfs -put /path-to-events.csv /input/events/
-- Same steps 1 - 5 for mortality.csv, except that the path is /input/mortality
-- ***************************************************************************
-- create events table 
DROP TABLE IF EXISTS events;
CREATE EXTERNAL TABLE events (
  patient_id STRING,
  event_id STRING,
  event_description STRING,
  time DATE,
  value DOUBLE)
ROW FORMAT DELIMITED FIELDS TERMINATED BY ','
STORED AS TEXTFILE
LOCATION '/input/events';

-- create mortality events table 
DROP TABLE IF EXISTS mortality;
CREATE EXTERNAL TABLE mortality (
  patient_id STRING,
  time DATE,
  label INT)
ROW FORMAT DELIMITED FIELDS TERMINATED BY ','
STORED AS TEXTFILE
LOCATION '/input/mortality';

-- ******************************************************
-- Generate two views for alive and dead patients' events
-- ******************************************************
-- find events for alive patients
DROP VIEW IF EXISTS alive_events;
CREATE VIEW alive_events 
AS
SELECT events.patient_id, events.event_id, events.time 
FROM events 
WHERE patient_id NOT IN (
    SELECT patient_id 
    FROM mortality);

-- find events for dead patients
DROP VIEW IF EXISTS dead_events;
CREATE VIEW dead_events 
AS
SELECT events.patient_id, events.event_id, events.time
FROM events
WHERE patient_id IN (
    SELECT patient_id 
    FROM mortality);

-- ************************************************
-- Event count metrics: Avg, min, and max of event counts
-- for alive and dead patients respectively  
-- ************************************************
-- alive patients
INSERT OVERWRITE LOCAL DIRECTORY 'event_count_alive'
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
STORED AS TEXTFILE
SELECT avg(event_count), min(event_count), max(event_count)
FROM (
  SELECT count(event_id) AS event_count
  FROM alive_events
  GROUP BY patient_id) alive_count;

-- dead patients
INSERT OVERWRITE LOCAL DIRECTORY 'event_count_dead'
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
STORED AS TEXTFILE
SELECT avg(event_count), min(event_count), max(event_count)
FROM (
  SELECT count(event_id) as event_count
FROM dead_events
GROUP BY patient_id) dead_count;


-- ************************************************
-- Encounter count metrics: average, median, min and max of encounter counts 
-- for alive and dead patients respectively
-- ************************************************
-- alive
INSERT OVERWRITE LOCAL DIRECTORY 'encounter_count_alive'
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
STORED AS TEXTFILE
SELECT avg(encounter_count), percentile(encounter_count, 0.5), min(encounter_count), max(encounter_count)
FROM (
  SELECT patient_id, COUNT(DISTINCT(time)) as encounter_count
  FROM alive_events
  GROUP BY patient_id) alive_encounter;


-- dead
INSERT OVERWRITE LOCAL DIRECTORY 'encounter_count_dead'
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
STORED AS TEXTFILE
SELECT avg(encounter_count), percentile(encounter_count, 0.5), min(encounter_count), max(encounter_count)
FROM (
  SELECT patient_id, COUNT(DISTINCT(time)) as encounter_count 
  FROM dead_events
  GROUP BY patient_id) dead_encounter;


-- ************************************************
-- Record length metrics: avg, median, min and max of record lengths
-- for alive and dead patients respectively
-- ************************************************
-- alive 
INSERT OVERWRITE LOCAL DIRECTORY 'record_length_alive'
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
STORED AS TEXTFILE
SELECT avg(record_length), percentile(record_length, 0.5), min(record_length), max(record_length)
FROM (
  SELECT patient_id, DATEDIFF(max(time),min(time)) as record_length 
  FROM alive_events GROUP BY patient_id) alive_days;

-- dead
INSERT OVERWRITE LOCAL DIRECTORY 'record_length_dead'
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
STORED AS TEXTFILE
SELECT avg(record_length), percentile(record_length, 0.5), min(record_length), max(record_length)
FROM (
  SELECT patient_id, DATEDIFF(max(time),min(time)) as record_length
  FROM dead_events GROUP BY patient_id) dead_days;


-- ******************************************* 
-- Common diag/lab/med: Compute the 5 most frequently occurring diag/lab/med
-- for alive and dead patients respectively
-- *******************************************
-- alive patients
---- diag
INSERT OVERWRITE LOCAL DIRECTORY 'common_diag_alive'
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
STORED AS TEXTFILE
SELECT event_id, count(*) AS diag_count
FROM alive_events
WHERE event_id LIKE 'DIAG%'
GROUP BY event_id
ORDER BY diag_count DESC
LIMIT 5;

---- lab
INSERT OVERWRITE LOCAL DIRECTORY 'common_lab_alive'
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
STORED AS TEXTFILE
SELECT event_id, count(*) AS lab_count
FROM alive_events
WHERE event_id LIKE 'LAB%'
GROUP BY event_id
ORDER BY lab_count DESC
LIMIT 5;

---- med
INSERT OVERWRITE LOCAL DIRECTORY 'common_med_alive'
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
STORED AS TEXTFILE
SELECT event_id, count(*) AS med_count
FROM alive_events
WHERE event_id LIKE 'DRUG%'
GROUP BY event_id
ORDER BY med_count DESC
LIMIT 5;

-- dead patients
---- diag
INSERT OVERWRITE LOCAL DIRECTORY 'common_diag_dead'
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
STORED AS TEXTFILE
SELECT event_id, count(*) AS diag_count
FROM dead_events
WHERE event_id LIKE 'DIAG%'
GROUP BY event_id
ORDER BY diag_count DESC
LIMIT 5;

---- lab
INSERT OVERWRITE LOCAL DIRECTORY 'common_lab_dead'
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
STORED AS TEXTFILE
SELECT event_id, count(*) AS lab_count
FROM dead_events
WHERE event_id LIKE 'LAB%'
GROUP BY event_id
ORDER BY lab_count DESC
LIMIT 5;

---- med
INSERT OVERWRITE LOCAL DIRECTORY 'common_med_dead'
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
STORED AS TEXTFILE
SELECT event_id, count(*) AS med_count
FROM dead_events
WHERE event_id LIKE 'DRUG%'
GROUP BY event_id
ORDER BY med_count DESC
LIMIT 5;
