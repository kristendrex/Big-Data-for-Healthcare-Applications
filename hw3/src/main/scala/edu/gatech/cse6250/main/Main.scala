package edu.gatech.cse6250.main

import java.text.SimpleDateFormat

import edu.gatech.cse6250.clustering.Metrics
import edu.gatech.cse6250.features.FeatureConstruction
import edu.gatech.cse6250.helper.{ CSVHelper, SparkHelper }
import edu.gatech.cse6250.model.{ Diagnostic, LabResult, Medication }
import edu.gatech.cse6250.phenotyping.T2dmPhenotype
import org.apache.spark.mllib.clustering.{ GaussianMixture, KMeans, StreamingKMeans }
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.linalg.{ DenseMatrix, Matrices, Vector, Vectors }
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

import scala.io.Source

object Main {
  def main(args: Array[String]) {
    import org.apache.log4j.{ Level, Logger }

    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)

    val spark = SparkHelper.spark
    val sc = spark.sparkContext
    //  val sqlContext = spark.sqlContext

    /** initialize loading of data */
    val (medication, labResult, diagnostic) = loadRddRawData(spark)
    val (candidateMedication, candidateLab, candidateDiagnostic) = loadLocalRawData

    /** conduct phenotyping */
    val phenotypeLabel = T2dmPhenotype.transform(medication, labResult, diagnostic)

    /** feature construction with all features */
    val featureTuples = sc.union(
      FeatureConstruction.constructDiagnosticFeatureTuple(diagnostic),
      FeatureConstruction.constructLabFeatureTuple(labResult),
      FeatureConstruction.constructMedicationFeatureTuple(medication)
    )

    val rawFeatures = FeatureConstruction.construct(sc, featureTuples)

    val (kMeansPurity, gaussianMixturePurity, streamingPurity) = testClustering(phenotypeLabel, rawFeatures)
    println(f"[All feature] purity of kMeans is: $kMeansPurity%.5f")
    println(f"[All feature] purity of GMM is: $gaussianMixturePurity%.5f")
    println(f"[All feature] purity of StreamingKmeans is: $streamingPurity%.5f")

    /** feature construction with filtered features */
    val filteredFeatureTuples = sc.union(
      FeatureConstruction.constructDiagnosticFeatureTuple(diagnostic, candidateDiagnostic),
      FeatureConstruction.constructLabFeatureTuple(labResult, candidateLab),
      FeatureConstruction.constructMedicationFeatureTuple(medication, candidateMedication)
    )

    val filteredRawFeatures = FeatureConstruction.construct(sc, filteredFeatureTuples)

    val (kMeansPurity2, gaussianMixturePurity2, streamingPurity2) = testClustering(phenotypeLabel, filteredRawFeatures)
    println(f"[Filtered feature] purity of kMeans is: $kMeansPurity2%.5f")
    println(f"[Filtered feature] purity of GMM is: $gaussianMixturePurity2%.5f")
    println(f"[Filtered feature] purity of StreamingKmeans is: $streamingPurity2%.5f")
  }

  def testClustering(phenotypeLabel: RDD[(String, Int)], rawFeatures: RDD[(String, Vector)]): (Double, Double, Double) = {
    import org.apache.spark.mllib.linalg.Matrix
    import org.apache.spark.mllib.linalg.distributed.RowMatrix

    println("phenotypeLabel: " + phenotypeLabel.count)
    /** scale features */
    val scaler = new StandardScaler(withMean = true, withStd = true).fit(rawFeatures.map(_._2))
    val features = rawFeatures.map({ case (patientID, featureVector) => (patientID, scaler.transform(Vectors.dense(featureVector.toArray))) })
    println("features: " + features.count)
    val rawFeatureVectors = features.map(_._2).cache()
    println("rawFeatureVectors: " + rawFeatureVectors.count)

    /** reduce dimension */
    val mat: RowMatrix = new RowMatrix(rawFeatureVectors)
    val pc: Matrix = mat.computePrincipalComponents(10) // Principal components are stored in a local dense matrix.
    val featureVectors = mat.multiply(pc).rows

    val densePc = Matrices.dense(pc.numRows, pc.numCols, pc.toArray).asInstanceOf[DenseMatrix]

    def transform(feature: Vector): Vector = {
      val scaled = scaler.transform(Vectors.dense(feature.toArray))
      Vectors.dense(Matrices.dense(1, scaled.size, scaled.toArray).multiply(densePc).toArray)
    }

    /**
     * K Means Clustering using spark mllib
     * Train a k means model using the variabe featureVectors as input
     * Assign each feature vector to a cluster(predicted Class)
     * Obtain an RDD[(Int, Int)] of the form (cluster number, RealClass)
     * Find Purity using that RDD as an input to Metrics.purity
     */
    featureVectors.cache()
    val k_means = new KMeans().setSeed(6205L).setK(3).setMaxIterations(20).run(featureVectors).predict(featureVectors)
    val k_means_pred = features.map(_._1).zip(k_means).join(phenotypeLabel).map(_._2)
    val kMeansPurity = Metrics.purity(k_means_pred)
    
    /**
     * GMMM Clustering using spark mllib
     * Train a Gaussian Mixture model using the variabe featureVectors as input
     * Assign each feature vector to a cluster(predicted Class)
     * Obtain an RDD[(Int, Int)] of the form (cluster number, RealClass)
     * Find Purity using that RDD as an input to Metrics.purity
     */

    val gmm = new GaussianMixture().setSeed(6205L).setK(3).setMaxIterations(20).run(featureVectors).predict(featureVectors)
    val gmm_pred = features.map(_._1).zip(gmm).join(phenotypeLabel).map(_._2)
    val gaussianMixturePurity = Metrics.purity(gmm_pred)
    /*
    
    /**
     * StreamingKMeans Clustering using spark mllib
     * Train a StreamingKMeans model using the variabe featureVectors as input
     * Assign each feature vector to a cluster(predicted Class)
     * Obtain an RDD[(Int, Int)] of the form (cluster number, RealClass)
     * Find Purity using that RDD as an input to Metrics.purity
     */

    val sKmeans = new StreamingKMeans().setK(3).setDecayFactor(1.0).setRandomCenters(10, 0.5, 6250L).latestModel()
    val sKmeans_pred = sKmeans.update(featureVectors, 1.0, "batches").predict(featureVectors)
    val sKmeans_test = features.map(_._1).zip(sKmeans_pred).join(phenotypeLabel).map(_._2)

    val streamKmeansPurity = Metrics.purity(sKmeans_test)

    //val streamKmeansPurity = 0.0

    (kMeansPurity, gaussianMixturePurity, streamKmeansPurity)
  }

  /**
   * load the sets of string for filtering of medication
   * lab result and diagnostics
   *
   * @return
   */
  def loadLocalRawData: (Set[String], Set[String], Set[String]) = {
    val candidateMedication = Source.fromFile("data/med_filter.txt").getLines().map(_.toLowerCase).toSet[String]
    val candidateLab = Source.fromFile("data/lab_filter.txt").getLines().map(_.toLowerCase).toSet[String]
    val candidateDiagnostic = Source.fromFile("data/icd9_filter.txt").getLines().map(_.toLowerCase).toSet[String]
    (candidateMedication, candidateLab, candidateDiagnostic)
  }

  def sqlDateParser(input: String, pattern: String = "yyyy-MM-dd'T'HH:mm:ssX"): java.sql.Date = {
    val dateFormat = new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ssX")
    new java.sql.Date(dateFormat.parse(input).getTime)
  }

  def loadRddRawData(spark: SparkSession): (RDD[Medication], RDD[LabResult], RDD[Diagnostic]) = {
    /* the sql queries in spark required to import sparkSession.implicits._ */
    import spark.implicits._
    val sqlContext = spark.sqlContext

    val dateFormat = new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ssX")

    // medication data
    val medicationOrders = CSVHelper.loadCSVAsTable(spark, "file:///hw3/code/data/medication_orders_INPUT.csv", "MedicationTable")
    val medicationDF = spark.sql("SELECT Member_ID AS patientID, Order_Date AS date, Drug_Name AS medicine FROM MedicationTable")
    val medication: RDD[Medication] = medicationDF.rdd.map(x => { Medication(x(0).toString, sqlDateParser(x(1).toString), x(2).toString.toLowerCase) })

    // lab result data
    val labResultData = CSVHelper.loadCSVAsTable(spark, "file:///hw3/code/data/lab_results_INPUT.csv", "LabTable")
    val labResultDF = spark.sql("SELECT Member_ID AS patientID, Date_Resulted AS date, Result_Name AS testName, Numeric_Result as value FROM LabTable WHERE Numeric_Result != ''")
    val labResult: RDD[LabResult] = labResultDF.rdd.map(x => LabResult(x(0).toString, sqlDateParser(x(1).toString), x(2).toString.toLowerCase, x(3).toString.filterNot(",".toSet).toDouble))

    val IDdata = CSVHelper.loadCSVAsTable(spark, "file:///hw3/code/data/encounter_INPUT.csv", "IDTable")
    val diagnosedata = CSVHelper.loadCSVAsTable(spark, "file:///hw3/code/data/encounter_dx_INPUT.csv", "DiagnoseTable")
    val diagnoseDF = spark.sql("SELECT IDTable.Member_ID AS patientID, IDTable.Encounter_DateTime AS date, DiagnoseTable.code AS code FROM IDTable INNER JOIN DiagnoseTable ON IDTable.Encounter_ID = DiagnoseTable.Encounter_ID")
    val diagnostic: RDD[Diagnostic] = diagnoseDF.rdd.map(x => Diagnostic(x(0).toString, sqlDateParser(x(1).toString), x(2).toString.toLowerCase))

    (medication, labResult, diagnostic)
  }

}
