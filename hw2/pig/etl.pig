-- ***************************************************************************
-- Aggregate events into features of patient and generate training, testing data for mortality prediction.
-- ***************************************************************************

REGISTER utils.py USING jython AS utils;

-- load events file
events = LOAD '../../data/events.csv' USING PigStorage(',') AS (patientid:int, eventid:chararray, eventdesc:chararray, timestamp:chararray, value:float);
-- select required columns from events
events = FOREACH events GENERATE patientid, eventid, ToDate(timestamp, 'yyyy-MM-dd') AS etimestamp, value;

-- load mortality file
mortality = LOAD '../../data/mortality.csv' USING PigStorage(',') as (patientid:int, timestamp:chararray, label:int);
-- select required columns from mortality
mortality = FOREACH mortality GENERATE patientid, ToDate(timestamp, 'yyyy-MM-dd') AS mtimestamp, label;


-- ***************************************************************************
-- Compute the index dates for dead and alive patients
-- ***************************************************************************
eventswithmort = JOIN events BY patientid LEFT OUTER, mortality BY patientid;

deadevents = FILTER eventswithmort by(mortality::patientid is not null);
deadevents = FOREACH deadevents GENERATE events::patientid AS patientid,events::eventid AS eventid,events::value AS value,mortality::label AS label, DaysBetween(SubtractDuration(mortality::mtimestamp,'P30D'), events::etimestamp) AS time_difference;

aliveevents = FILTER eventswithmort by(mortality::patientid is null);
aliveevents = FOREACH aliveevents GENERATE events::patientid AS patientid, events::eventid AS eventid, events::value AS value, 0 AS label, events::etimestamp AS timestamp;
aliveevents_maxdate = GROUP aliveevents BY patientid;
aliveevents_maxdate = FOREACH aliveevents_maxdate GENERATE group AS patientid, MAX(aliveevents.timestamp) AS maxdate;
aliveevents = JOIN aliveevents BY patientid, aliveevents_maxdate BY patientid;
aliveevents = FOREACH aliveevents GENERATE aliveevents::patientid AS patientid, aliveevents::eventid AS eventid, aliveevents::value AS value,aliveevents::label AS label, DaysBetween(aliveevents_maxdate::maxdate,aliveevents::timestamp) AS time_difference;


-- ***************************************************************************
-- Filter events within the observation window and remove events with missing values
-- ***************************************************************************
-- contains only events for all patients within the observation window of 2000 days and is of the form (patientid, eventid, value, label, time_difference)
allevents = UNION aliveevents, deadevents;
allevents = FILTER allevents BY value is not null;
filtered = FILTER allevents BY (time_difference >= 0 AND time_difference <= 2000);

-- ***************************************************************************
-- Aggregate events to create features
-- ***************************************************************************
-- for group of (patientid, eventid), count the number of  events occurred for the patient and create relation of the form (patientid, eventid, featurevalue)
filterfeat = GROUP filtered BY (patientid,eventid);
featureswithid = FOREACH filterfeat GENERATE FLATTEN(group) AS (patientid, eventid), COUNT(filtered.value) AS featurevalue;

-- ***************************************************************************
-- Generate feature mapping
-- ***************************************************************************
all_features = DISTINCT(FOREACH featureswithid GENERATE eventid);
all_features = RANK all_features BY eventid ASC;
all_features = FOREACH all_features GENERATE $0-1 AS idx, $1;

-- store the features as an output file
STORE all_features INTO 'features' using PigStorage(' ');

-- perform join of featureswithid and all_features by eventid and replace eventid with idx. It is of the form (patientid, idx, featurevalue)
features = JOIN featureswithid BY eventid LEFT, all_features BY eventid;
features = FOREACH features GENERATE featureswithid::patientid AS patientid, all_features::idx as idx, featureswithid::featurevalue AS featurevalue;


-- ***************************************************************************
-- Normalize the values using min-max normalization
-- Use DOUBLE precision
-- ***************************************************************************
-- group events by idx and compute the maximum feature value in each group. I t is of the form (idx, maxvalue)
maxvalues = GROUP features BY idx;
maxvalues = FOREACH maxvalues GENERATE group AS idx, MAX(features.featurevalue) AS maxvalue;
-- join features and maxvalues by idx
normalized = JOIN features BY idx, maxvalues BY idx;
-- compute the final set of normalized features of the form (patientid, idx, normalizedfeaturevalue)
features = FOREACH normalized GENERATE features::patientid AS patientid, features::idx AS idx, ((DOUBLE)features::featurevalue/(DOUBLE)maxvalues::maxvalue) as normalizedfeaturevalue;


-- ***************************************************************************
-- Generate features in svmlight format
-- features is of the form (patientid, idx, normalizedfeaturevalue) and is the output of the previous step
-- e.g.  1,1,1.0
--  	 1,3,0.8
--	     2,1,0.5
--       3,3,1.0
-- ***************************************************************************

grpd = GROUP features BY patientid;
grpd_order = ORDER grpd BY $0;
features = FOREACH grpd_order
{
    sorted = ORDER features BY idx;
    generate group as patientid, utils.bag_to_svmlight(sorted) as sparsefeature;
}

-- ***************************************************************************
-- Split into train and test set
-- labels is of the form (patientid, label) and contains all patientids followed by label of 1 for dead and 0 for alive
-- e.g. 1,1
--	2,0
--      3,1
-- ***************************************************************************

-- create it of the form (patientid, label) for dead and alive patients
labels = FOREACH filtered GENERATE patientid,label;
labels = DISTINCT labels;


--Generate sparsefeature vector relation
samples = JOIN features BY patientid, labels BY patientid;
samples = DISTINCT samples PARALLEL 1;
samples = ORDER samples BY $0;
samples = FOREACH samples GENERATE $3 AS label, $1 AS sparsefeature;

-- randomly split data for training and testing
DEFINE rand_gen RANDOM('6505');
samples = FOREACH samples GENERATE rand_gen() as assignmentkey, *;
SPLIT samples INTO testing IF assignmentkey <= 0.20, training OTHERWISE;
training = FOREACH training GENERATE $1..;
testing = FOREACH testing GENERATE $1..;

-- save training and tesing data
STORE testing INTO 'testing' USING PigStorage(' ');
STORE training INTO 'training' USING PigStorage(' ');