/**
 *
 * students: please put your implementation in this file!
 */
package edu.gatech.cse6250.jaccard

import edu.gatech.cse6250.model._
import edu.gatech.cse6250.model.{ EdgeProperty, VertexProperty }
import org.apache.spark.graphx._
import org.apache.spark.rdd.RDD

object Jaccard {

  def jaccardSimilarityOneVsAll(graph: Graph[VertexProperty, EdgeProperty], patientID: Long): List[Long] = {
    /**
     * Given a patient ID, compute the Jaccard similarity w.r.t. to all other patients.
     * Return a List of top 10 patient IDs ordered by the highest to the lowest similarity.
     * For ties, random order is okay. The given patientID should be excluded from the result.
     */
    val all_neighbors = graph.collectNeighborIds(EdgeDirection.Out)
    val setPatID = all_neighbors.lookup(patientID).head.toSet
    val patientVerts = graph.subgraph(vpred = (id, attr) => attr.isInstanceOf[PatientProperty]).collectNeighborIds(EdgeDirection.Out)
      .map(x => x._1).filter(x => x != patientID).collect().toSet
    var pVtopE = all_neighbors.filter(x => patientVerts.contains(x._1))
    var jaccardTest = pVtopE.map(x => (jaccard(setPatID, x._2.toSet), x._1)).sortBy(_._1, false).take(10).map(_._2.toLong).toList
    jaccardTest
    /** Remove this placeholder and implement your code */

  }

  def jaccardSimilarityAllPatients(graph: Graph[VertexProperty, EdgeProperty]): RDD[(Long, Long, Double)] = {
    /**
     * Given a patient, med, diag, lab graph, calculate pairwise similarity between all
     * patients. Return a RDD of (patient-1-id, patient-2-id, similarity) where
     * patient-1-id < patient-2-id to avoid duplications
     */

    /** Remove this placeholder and implement your code */
    val sc = graph.edges.sparkContext
    var all_neighbors = graph.collectNeighborIds(EdgeDirection.Out)
    var filtered = graph.subgraph(vpred = (id, attr) => attr.isInstanceOf[PatientProperty]).collectNeighborIds(EdgeDirection.Out).map(x => x._1).collect().toSet
    var patients_neighbors = all_neighbors.filter(x => filtered.contains(x._1))
    var idFilter = patients_neighbors.cartesian(patients_neighbors).filter(x => x._1._1 < x._2._1)
    val jacAll = idFilter.map(x => (x._1._1, x._2._1, jaccard(x._1._2.toSet, x._2._2.toSet)))
    jacAll   

  }

  def jaccard[A](a: Set[A], b: Set[A]): Double = {
    /**
     * Helper function
     *
     * Given two sets, compute its Jaccard similarity and return its result.
     * If the union part is zero, then return 0.
     */

    /** Remove this placeholder and implement your code */
    val numerator = a.intersect(b).size.toDouble
    val denominator = a.union(b).size.toDouble
    val jac = numerator / denominator
    if (jac.isNaN) 0.0 else jac
  }
}
