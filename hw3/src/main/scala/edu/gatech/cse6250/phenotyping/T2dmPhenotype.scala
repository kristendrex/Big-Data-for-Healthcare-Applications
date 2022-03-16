package edu.gatech.cse6250.phenotyping

import edu.gatech.cse6250.model.{ Diagnostic, LabResult, Medication }
import org.apache.spark.rdd.RDD

/**
 * @author Hang Su <hangsu@gatech.edu>,
 * @author Sungtae An <stan84@gatech.edu>,
 */
object T2dmPhenotype {

  /** Hard code the criteria */
  val T1DM_DX = Set("250.01", "250.03", "250.11", "250.13", "250.21", "250.23", "250.31", "250.33", "250.41", "250.43",
    "250.51", "250.53", "250.61", "250.63", "250.71", "250.73", "250.81", "250.83", "250.91", "250.93")

  val T2DM_DX = Set("250.3", "250.32", "250.2", "250.22", "250.9", "250.92", "250.8", "250.82", "250.7", "250.72", "250.6",
    "250.62", "250.5", "250.52", "250.4", "250.42", "250.00", "250.02")

  val T1DM_MED = Set("lantus", "insulin glargine", "insulin aspart", "insulin detemir", "insulin lente", "insulin nph", "insulin reg", "insulin,ultralente")

  val T2DM_MED = Set("chlorpropamide", "diabinese", "diabanase", "diabinase", "glipizide", "glucotrol", "glucotrol xl",
    "glucatrol ", "glyburide", "micronase", "glynase", "diabetamide", "diabeta", "glimepiride", "amaryl",
    "repaglinide", "prandin", "nateglinide", "metformin", "rosiglitazone", "pioglitazone", "acarbose",
    "miglitol", "sitagliptin", "exenatide", "tolazamide", "acetohexamide", "troglitazone", "tolbutamide",
    "avandia", "actos", "actos", "glipizide")

  /**
   * Transform given data set to a RDD of patients and corresponding phenotype
   *
   * @param medication medication RDD
   * @param labResult  lab result RDD
   * @param diagnostic diagnostic code RDD
   * @return tuple in the format of (patient-ID, label). label = 1 if the patient is case, label = 2 if control, 3 otherwise
   */
  def transform(medication: RDD[Medication], labResult: RDD[LabResult], diagnostic: RDD[Diagnostic]): RDD[(String, Int)] = {
    /**
     * Remove the place holder and implement your code here.
     * Hard code the medication, lab, icd code etc. for phenotypes like example code below.
     * When testing your code, we expect your function to have no side effect,
     * i.e. do NOT read from file or write file
     *
     * //You don't need to follow the example placeholder code below exactly, but do have the same return type.
     * Hint: Consider case sensitivity when doing string comparisons.
     */

    val sc = medication.sparkContext

    /** Hard code the criteria */
    val type1_dm_dx = Set("code1", "250.01", "250.03", "250.11", "250.13", "250.21", "250.23", "250.31", "250.33", "250.41", "250.43", "250.51", "250.53", "250.61", "250.63", "250.71", "250.73", "250.81", "250.83", "250.91", "250.93")
    val type1_dm_med = Set("med1", "lantus", "insulin glargine", "insulin aspart", "insulin detemir", "insulin lente", "insulin nph", "insulin reg", "insulin,ultralente")
    val type2_dm_dx = Set("250.3", "250.32", "250.2", "250.22", "250.9", "250.92", "250.8", "250.82", "250.7", "250.72", "250.6", "250.62", "250.5", "250.52", "250.4", "250.42", "250.00", "250.02")
    val type2_dm_med = Set("chlorpropamide", "diabinese", "diabanase", "diabinase", "glipizide", "glucotrol", "glucotrol xl", "glucatrol ", "glyburide", "micronase", "glynase", "diabetamide", "diabeta", "glimepiride", "amaryl", "repaglinide", "prandin", "nateglinide", "metformin", "rosiglitazone", "pioglitazone", "acarbose", "miglitol", "sitagliptin", "exenatide", "tolazamide", "acetohexamide", "troglitazone", "tolbutamide", "avandia", "actos", "ACTOS", "glipizide")
    val dm_related_dx = Set("790.21", "790.22", "790.2", "790.29", "648.81", "648.82", "648.83", "648.84", "648", "648", "648.01", "648.02", "648.03", "648.04", "791.5", "277.7", "V77.1", "256.4")

    //all patients
    val patients = diagnostic.map(_.patientID).union(labResult.map(_.patientID)).union(medication.map(_.patientID)).distinct()

    //create groupings
    val T1DiagYes = diagnostic.filter(x => type1_dm_dx.contains(x.code)).map(_.patientID).distinct()
    val T1DiagNo = patients.subtract(T1DiagYes).distinct()
    val T1MedYes = medication.filter(x => type1_dm_med.contains(x.medicine.toLowerCase)).map(_.patientID).distinct()
    val T1MedNo = patients.subtract(T1MedYes).distinct()
    val T2DiagYes = diagnostic.filter(x => type2_dm_dx.contains(x.code)).map(_.patientID).distinct()
    val T2DiagNo = patients.subtract(T2DiagYes).distinct()
    val T2MedYes = medication.filter(x => type2_dm_med.contains(x.medicine.toLowerCase)).map(_.patientID).distinct()
    val T2MedNo = patients.subtract(T2MedYes).distinct()

    /** Find CASE Patients */
    val cp1 = T1DiagNo.intersection(T2DiagYes).intersection(T1MedNo)
    val cp2 = T1DiagNo.intersection(T2DiagYes).intersection(T1MedYes).intersection(T2MedNo)
    val cp3 = T1DiagNo.intersection(T2DiagYes).intersection(T1MedYes).intersection(T2MedYes)
    val cp3Med = medication.map(x => (x.patientID, x)).join(cp3.map(x => (x, 0))).map(x => Medication(x._2._1.patientID, x._2._1.date, x._2._1.medicine))
    val cp3_T1Med = cp3Med.filter(x => type1_dm_med.contains(x.medicine.toLowerCase)).map(x => (x.patientID, x.date.getTime())).reduceByKey(Math.min)
    val cp3_T2Med = cp3Med.filter(x => type2_dm_med.contains(x.medicine.toLowerCase)).map(x => (x.patientID, x.date.getTime())).reduceByKey(Math.min)
    val cp3_filter = cp3_T1Med.join(cp3_T2Med).filter(x => x._2._1 > x._2._2).map(_._1)
    val caseP = sc.union(cp1, cp2, cp3_filter).distinct()
    val casePatients = caseP.map((_, 1))

    /** Find CONTROL Patients */
    val glucose2 = labResult.filter(x => x.testName.toLowerCase.contains("glucose")).map(x => x.patientID).distinct()
    val ablab = labResult.filter(x => abnormal(x)).map(x => x.patientID).distinct()
    val normlab = glucose2.subtract(ablab)
    val mellitus = diagnostic.filter(x => dm_related_dx.contains(x.code) || x.code.contains("250.")).map(x => x.patientID).distinct()
    val noMellitus = patients.subtract(mellitus).distinct()
    val controlP = normlab.intersection(noMellitus)
    val controlPatients = controlP.map((_, 2))

    /** Find OTHER Patients */
    val otherP = patients.subtract(caseP).subtract(controlP).distinct()
    val others = otherP.map((_, 3))

    /** Once you find patients for each group, make them as a single RDD[(String, Int)] */
    val phenotypeLabel = sc.union(casePatients, controlPatients, others)

    /** Return */
    phenotypeLabel
  }

  def abnormal(item: LabResult): Boolean = {
    item.testName match {
      case "hba1c"                  => item.value >= 6
      case "hemoglobin a1c"         => item.value >= 6
      case "fasting glucose"        => item.value >= 110
      case "fasting blood glucose"  => item.value >= 110
      case "fasting plasma glucose" => item.value >= 110
      case "glucose"                => item.value > 110
      case "glucose, serum"         => item.value > 110
      case _                        => false
    }
  }

  /**
   * calculate specific stats given phenotype labels and corresponding data set rdd
   * @param labResult  lab result RDD
   * @param phenotypeLabel phenotype label return from T2dmPhenotype.transfrom
   * @return tuple in the format of (case_mean, control_mean, other_mean).
   *         case_mean = mean Glucose lab test results of case group
   *         control_mean = mean Glucose lab test results of control group
   *         other_mean = mean Glucose lab test results of unknown group
   *         Attention: order of the three stats in the returned tuple matters!
   */
  def stat_calc(labResult: RDD[LabResult], phenotypeLabel: RDD[(String, Int)]): (Double, Double, Double) = {
    /**
     * you need to hardcode the feature name and the type of stat:
     * e.g. calculate "mean" of "Glucose" lab test result of each group: case, control, unknown
     *
     * The feature name should be "Glucose" exactly with considering case sensitivity.
     * i.e. "Glucose" and "glucose" are counted, but features like "fasting glucose" should not be counted.
     *
     * Hint: rdd dataset can directly call statistic method. Details can be found on course website.
     *
     */
    val case1 = phenotypeLabel.filter(x => x._2 == 1).map(x => x._1).collect.toSet
    val case_mean = labResult.filter(x => x.testName.toLowerCase.contains("glucose")).filter(x => case1(x.patientID)).map(x => x.value).mean()
    val case2 = phenotypeLabel.filter(x => x._2 == 2).map(x => x._1).collect.toSet
    val control_mean = labResult.filter(x => x.testName.toLowerCase.contains("glucose")).filter(x => case2(x.patientID)).map(x => x.value).mean()
    val case3 = phenotypeLabel.filter(x => x._2 == 3).map(x => x._1).collect.toSet
    val other_mean = labResult.filter(x => x.testName.toLowerCase.contains("glucose")).filter(x => case3(x.patientID)).map(x => x.value).mean()

    //val case_mean = 1.0
    //val control_mean = 1.0
    //val other_mean = 1.0
    (case_mean, control_mean, other_mean)
  }
}