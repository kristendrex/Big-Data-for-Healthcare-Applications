/**
 * @author Hang Su <hangsu@gatech.edu>.
 */

package edu.gatech.cse6250.graphconstruct

import edu.gatech.cse6250.model._
import org.apache.spark.graphx._
import org.apache.spark.rdd.RDD

object GraphLoader {
  /**
   * Generate Bipartite Graph using RDDs
   *
   * @input: RDDs for Patient, LabResult, Medication, and Diagnostic
   * @return: Constructed Graph
   *
   */
  var vertexPatient: RDD[(VertexId, VertexProperty)] = _
  var vertexLabResult: RDD[(VertexId, VertexProperty)] = _
  var vertexMedication: RDD[(VertexId, VertexProperty)] = _
  var vertexDiagnostic: RDD[(VertexId, VertexProperty)] = _

  var edgeLabResult: RDD[Edge[EdgeProperty]] = _
  var edgeMedication: RDD[Edge[EdgeProperty]] = _
  var edgeDiagnostic: RDD[Edge[EdgeProperty]] = _

  def load(patients: RDD[PatientProperty], labResults: RDD[LabResult],
    medications: RDD[Medication], diagnostics: RDD[Diagnostic]): Graph[VertexProperty, EdgeProperty] = {

    /** HINT: See Example of Making Patient Vertices Below */
    //vertexPatient = patients.map(patient => (patient.patientID.toLong, patient.asInstanceOf[VertexProperty]))
    vertexPatient = patients.map(patient => (patient.patientID.toLong, patient.asInstanceOf[VertexProperty]))
    val patientCount = vertexPatient.count() + 1
    //lab
    var labRDD = labResults.map(_.labName).distinct().zipWithIndex().map(x => (x._1, x._2 + patientCount))
    vertexLabResult = labRDD.map(x => (x._2, LabResultProperty(x._1).asInstanceOf[VertexProperty]))
    val labCount = vertexLabResult.count() + patientCount + 1
    val labMapBC = sc.broadcast(labRDD.collect.toMap)
    //medication
    var vertexMedicationRDD = medications.map(_.medicine).distinct().zipWithIndex().map(x => (x._1, x._2 + labCount))
    vertexMedication = vertexMedicationRDD.map(x => (x._2, MedicationProperty(x._1).asInstanceOf[VertexProperty]))
    val medCount = vertexMedication.count() + patientCount + labCount + 1
    val medMapBC = sc.broadcast(vertexMedicationRDD.collect.toMap)
    //diagnostic
    var vertexDiagnosticRDD = diagnostics.map(_.icd9code).distinct().zipWithIndex().map(x => (x._1, x._2 + medCount))
    vertexDiagnostic = vertexDiagnosticRDD.map(x => (x._2, DiagnosticProperty(x._1).asInstanceOf[VertexProperty]))
    val diagMapBC = sc.broadcast(vertexDiagnosticRDD.collect.toMap)

    val vertices = vertexPatient.union(vertexLabResult).union(vertexMedication).union(vertexDiagnostic)
    /**
     * HINT: See Example of Making PatientPatient Edges Below
     *
     * This is just sample edges to give you an example.
     * You can remove this PatientPatient edges and make edges you really need
     */
    edgeDiagnostic = diagnostics.map(x => ((x.patientID, x.icd9code), x)).groupByKey()
      .map(x => ((x._1._1, x._1._2), x._2.last))
      .flatMap(y => List(
        Edge(y._1._1.toLong, diagMapBC.value(y._1._2), PatientDiagnosticEdgeProperty(y._2)),
        Edge(diagMapBC.value(y._1._2), y._1._1.toLong, PatientDiagnosticEdgeProperty(y._2))))
    //lab edge
    edgeLabResult = labResults.map(l => ((l.patientID, l.labName), l)).groupByKey()
      .map(z => ((z._1._1, z._1._2), z._2.last))
      .flatMap(p => List(
        Edge(p._1._1.toLong, labMapBC.value(p._1._2), PatientLabEdgeProperty(p._2)),
        Edge(labMapBC.value(p._1._2), p._1._1.toLong, PatientLabEdgeProperty(p._2))))
    //medication edge
    edgeMedication = medications.map(m => ((m.patientID, m.medicine), m)).groupByKey()
      .map(p => ((p._1._1, p._1._2), p._2.last))
      .flatMap(e => List(
        Edge(e._1._1.toLong, medMapBC.value(e._1._2), PatientMedicationEdgeProperty(e._2)),
        Edge(medMapBC.value(e._1._2), e._1._1.toLong, PatientMedicationEdgeProperty(e._2))))    

    //combine edges
    val edges = edgeDiagnostic.union(edgeLabResult).union(edgeMedication)

    // Making Graph
    val graph: Graph[VertexProperty, EdgeProperty] = Graph[VertexProperty, EdgeProperty](vertices, edges)

    graph
  }
}
