package edu.gatech.cse6250.features

import edu.gatech.cse6250.model.{ Diagnostic, LabResult, Medication }
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.{ Vector, Vectors }
import org.apache.spark.rdd.RDD


object FeatureConstruction {

  /**
   * ((patient-id, feature-name), feature-value)
   */
  type FeatureTuple = ((String, String), Double)

  /**
   * Aggregate feature tuples from diagnostic with COUNT aggregation,
   *
   * @param diagnostic RDD of diagnostic
   * @return RDD of feature tuples
   */
  def constructDiagnosticFeatureTuple(diagnostic: RDD[Diagnostic]): RDD[FeatureTuple] = {
    val diag_feature = diagnostic.map(x => ((x.patientID, x.code), 1.0)).reduceByKey(_ + _)
    diag_feature

  }

  /**
   * Aggregate feature tuples from medication with COUNT aggregation,
   * @param medication RDD of medication
   * @return RDD of feature tuples
   */
  def constructMedicationFeatureTuple(medication: RDD[Medication]): RDD[FeatureTuple] = {
    val med_feature = medication.map(x => ((x.patientID, x.medicine), 1.0)).reduceByKey(_ + _)
    med_feature
  }

  /**
   * Aggregate feature tuples from lab result, using AVERAGE aggregation
   *
   * @param labResult RDD of lab result
   * @return RDD of feature tuples
   */
  def constructLabFeatureTuple(labResult: RDD[LabResult]): RDD[FeatureTuple] = {
    val lab_sum = labResult.map(x => ((x.patientID, x.testName), x.value)).reduceByKey(_ + _)
    val lab_count = labResult.map(x => ((x.patientID, x.testName), 1.0)).reduceByKey(_ + _)
    val lab_feature = lab_sum.join(lab_count).map(x => (x._1, x._2._1 / x._2._2))
    lab_feature
  }

  /**
   * Aggregate feature tuple from diagnostics with COUNT aggregation, but use code that is
   * available in the given set only and drop all others.
   *
   * @param diagnostic   RDD of diagnostics
   * @param candiateCode set of candidate code, filter diagnostics based on this set
   * @return RDD of feature tuples
   */
  def constructDiagnosticFeatureTuple(diagnostic: RDD[Diagnostic], candiateCode: Set[String]): RDD[FeatureTuple] = {
    val diag_features = diagnostic.filter(x => candiateCode.contains(x.code)).map(x => ((x.patientID, x.code), 1.0)).reduceByKey(_ + _)
    diag_features
  }

  /**
   * Aggregate feature tuples from medication with COUNT aggregation, use medications from
   * given set only and drop all others.
   *
   * @param medication          RDD of diagnostics
   * @param candidateMedication set of candidate medication
   * @return RDD of feature tuples
   */
  def constructMedicationFeatureTuple(medication: RDD[Medication], candidateMedication: Set[String]): RDD[FeatureTuple] = {
    val med_features = medication.filter(x => candidateMedication.contains(x.medicine)).map(x => ((x.patientID, x.medicine), 1.0)).reduceByKey(_ + _)
    med_features

  }

  /**
   * Aggregate feature tuples from lab result with AVERAGE aggregation, use lab from
   * given set of lab test names only and drop all others.
   *
   * @param labResult    RDD of lab result
   * @param candidateLab set of candidate lab test name
   * @return RDD of feature tuples
   */
  def constructLabFeatureTuple(labResult: RDD[LabResult], candidateLab: Set[String]): RDD[FeatureTuple] = {
    val lab_sum = labResult.map(x => ((x.patientID, x.testName), x.value)).reduceByKey(_ + _)
    val lab_count = labResult.map(x => ((x.patientID, x.testName), 1.0)).reduceByKey(_ + _)
    val lab_feature = lab_sum.join(lab_count).map(x => (x._1, x._2._1 / x._2._2))
    val lab_type = lab_feature.filter(x => (candidateLab.contains(x._1._2)))
    lab_type

  }

  /**
   * Given a feature tuples RDD, construct features in vector
   * format for each patient. feature name should be mapped
   * to some index and convert to sparse feature format.
   *
   * @param sc      SparkContext to run
   * @param feature RDD of input feature tuples
   * @return
   */
  def construct(sc: SparkContext, feature: RDD[FeatureTuple]): RDD[(String, Vector)] = {

    /** save for later usage */
    feature.cache()

    /** create a feature name to id map */
    val feature_index = feature.map(_._1._2).distinct().collect.zipWithIndex.toMap
    val scFeatureMap = sc.broadcast(feature_index)

    /** transform input feature */

    val patientAndFeatures = feature.map(x => (x._1._1, (x._1._2, x._2))).groupByKey()

    val result = patientAndFeatures.map {
      case (target, features) =>
        val num_features = scFeatureMap.value.size
        val index_features = features.toList.map { case (featureName, featureValue) => (scFeatureMap.value(featureName), featureValue) }
        val featureVector = Vectors.sparse(num_features, index_features)
        val label_point = (target, featureVector)
        label_point
    }
    result

    /** The feature vectors returned can be sparse or dense. It is advisable to use sparse */
  }
}
