package com.naveen.test.utility
import org.apache.spark.sql.functions._
import org.apache.spark.sql.DataFrame
class ModelUtility {
  
  def getPerformance(df: DataFrame):Map[String,Double]= {
    
    var accuracy :Double= -0.1
    var recal :Double= -0.1
    var precision :Double= -0.1
    var f1score:Double= -0.1

    var tp :Double= df.where(df("label") === "1" && df("prediction") === "1").count()
    var tn :Double= df.where(df("label") === "0" && df("prediction") === "0").count()
    var fp :Double= df.where(df("label") === "0" && df("prediction") === "1").count()
    var fn :Double= df.where(df("label") === "1" && df("prediction") === "0").count()
    
    println("tp-"+tp+":tn-"+tn+":fp-"+fp+":fn-"+fn)
    
    accuracy = (tp+tn)/(tp+tn+fp+fn)
    recal =tp/(tp+fn)
    precision=tp/(tp+fp)
    f1score=(recal*precision)/(recal+precision)
    
   return Map("accuracy" -> accuracy,"recal"->recal,"precision"->precision,"f1score"->f1score)

  }
}