package edu.gatech.cse6250.model

import java.sql.Date

case class Diagnostic(patientID: String, date: Date, code: String)

case class LabResult(patientID: String, date: Date, testName: String, value: Double)

case class Medication(patientID: String, date: Date, medicine: String)
