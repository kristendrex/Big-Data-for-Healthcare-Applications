package edu.gatech.cse6250.randomwalk

import edu.gatech.cse6250.model.{ PatientProperty, EdgeProperty, VertexProperty }
import org.apache.spark.graphx._

object RandomWalk {

  def randomWalkOneVsAll(graph: Graph[VertexProperty, EdgeProperty], patientID: Long, numIter: Int = 100, alpha: Double = 0.15): List[Long] = {
    /**
     * Given a patient ID, compute the random walk probability w.r.t. to all other patients.
     * Return a List of patient IDs ordered by the highest to the lowest similarity.
     * For ties, random order is okay
     */
    val rankedGraph = runRW(graph, numIter, alpha, patientID)

    val patientsRDD = graph.vertices.filter { elt => elt._2.getClass.getName == "edu.gatech.cse6250.model.PatientProperty" }
    val patientsList = patientsRDD.map { x => x._1 }.collect.toList

    val rankedMap = rankedGraph.vertices.collect.toList.toMap

    val resultList = patientsList.map { x => (x, rankedMap(x)) }.sortBy(-_._2).map { y => y._1 }.filter { z => z != patientID }

    resultList
  }
  def runRW(graph: Graph[VertexProperty, EdgeProperty], numIter: Int, resetProb: Double = 0.15,
    patientID: Long): Graph[Double, Double] =
    {

      val src: VertexId = patientID

      var rankGraph: Graph[Double, Double] = graph
        .outerJoinVertices(graph.outDegrees) { (vid, vdata, deg) => deg.getOrElse(0) }
        .mapTriplets(edge => 1.0 / edge.srcAttr, TripletFields.Src)
        .mapVertices { (id, attr) =>
          if (!(id != src)) resetProb else 0.0
        }

      var iteration = 0
      var prevRankGraph: Graph[Double, Double] = null
      while (iteration < numIter) {
        rankGraph.cache()

        val rankUpdates = rankGraph.aggregateMessages[Double](
          ctx => ctx.sendToDst(ctx.srcAttr * ctx.attr), _ + _, TripletFields.Src)

        prevRankGraph = rankGraph

        rankGraph = rankGraph.joinVertices(rankUpdates) {
          (id, oldRank, msgSum) => if (id == patientID) resetProb + (1.0 - resetProb) * msgSum else (1.0 - resetProb) * msgSum
        }.cache()

        rankGraph.edges.foreachPartition(x => {})
        prevRankGraph.vertices.unpersist(false)
        prevRankGraph.edges.unpersist(false)

        iteration += 1
      }

      rankGraph
    }
}
