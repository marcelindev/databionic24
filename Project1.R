library(dbt.DataIO)
library(AdaptLGT)
library(AdaptGauss)
library(caret)
library(Classifiers)
require(FCPS)
require(neuralnet)
require(DataVisualizations)

set.seed(123)

#Aufgabe1

dataset <- dbt.DataIO::ReadLRN("Flowcytometry.lrn")
data <- dataset$Data


result1 <- AdaptLGT::AdaptLGT(data[,1])
qq1 <- QQplot4Mixture(data[,1],result1$DisTypes,result1$Params,result1$Weights)
chi1 <- ChiSquareTest4Mixture(data[,1],result1$DisTypes,result1$Params,result1$Weights)

result2 <- AdaptLGT::AdaptLGT(data[,2])
qq2 <- QQplot4Mixture(data[,2],result2$DisTypes,result2$Params,result2$Weights)
chi2 <- ChiSquareTest4Mixture(data[,2],result2$DisTypes,result2$Params,result2$Weights)

result3 <- AdaptLGT::AdaptLGT(data[,3])
qq3 <- QQplot4Mixture(data[,3],result3$DisTypes,result3$Params,result3$Weights)
chi3 <- ChiSquareTest4Mixture(data[,3],result3$DisTypes,result3$Params,result3$Weights)

result4 <- AdaptLGT::AdaptLGT(data[,4])
qq4 <- QQplot4Mixture(data[,4],result4$DisTypes,result4$Params,result4$Weights)
chi4 <- ChiSquareTest4Mixture(data[,4],result4$DisTypes,result4$Params,result4$Weights)

#We managed to perform the gmm with results rms less or equal to 0.2 , the qq plots validate the model
#We have big chi2Value because of the great amount of data but the p value is near zero


#Aufgabe2

x <- ReadLRN("WisconsinDiagnosis.lrn")[2][[1]]
y <- ReadCLS("WisconsinDiagnosis.cls")[2][[1]]

# Scale the features
preProcValues <- preProcess(as.data.frame(x), method = c("center", "scale"))
x_scaled <- predict(preProcValues, as.data.frame(x))

# Combine x_scaled and y into a single data frame
dataset <- cbind(x_scaled, Diagnosis = as.factor(y))

# Function to train and evaluate the model
evaluate_model <- function(data, n_iterations = 100) {
  accuracies <- numeric(n_iterations)
  precision_vals <- numeric(n_iterations)
  recall_vals <- numeric(n_iterations)
  f1_scores <- numeric(n_iterations)
  
  for (i in 1:n_iterations) {
    set.seed(123 + i) # Ensure reproducibility with a different seed each iteration
    splitIndex <- createDataPartition(data$Diagnosis, p = 0.8, list = FALSE, times = 1)
    trainData <- data[splitIndex,]
    testData <- data[-splitIndex,]
    
    # Define the formula for neural network training
    formula <- as.formula(paste("Diagnosis ~", paste(names(trainData)[-ncol(trainData)], collapse = " + ")))
    
    # Train the neural network
    model <- neuralnet(formula, data = trainData, hidden = c(5, 3), linear.output = FALSE)
    
    # Predict on test data
    predictions <- neuralnet::compute(model, testData[, -ncol(testData)])$net.result
    
    # Determine predicted classes from probabilities
    predicted_classes <- apply(predictions, 1, function(row) ifelse(row[1] > row[2], 1, 2))
    
    # Ensure lengths match
    if (length(predicted_classes) == nrow(testData)) {
      predicted_classes <- as.factor(predicted_classes)
      actual_classes <- as.factor(testData$Diagnosis)
      
      # Calculate accuracy
      accuracies[i] <- sum(predicted_classes == actual_classes) / nrow(testData)
      
      # Confusion matrix
      conf_matrix <- table(Predicted = predicted_classes, Actual = actual_classes)
      
      # Calculate precision, recall, and F1-score
      tp <- conf_matrix[2, 2]
      tn <- conf_matrix[1, 1]
      fp <- conf_matrix[2, 1]
      fn <- conf_matrix[1, 2]
      
      precision_vals[i] <- tp / (tp + fp)
      recall_vals[i] <- tp / (tp + fn)
      f1_scores[i] <- 2 * (precision_vals[i] * recall_vals[i]) / (precision_vals[i] + recall_vals[i])
    } else {
      accuracies[i] <- NA
      precision_vals[i] <- NA
      recall_vals[i] <- NA
      f1_scores[i] <- NA
    }
  }
  
  results <- list(
    accuracies = accuracies,
    mean_accuracy = mean(accuracies, na.rm = TRUE),
    sd_accuracy = sd(accuracies, na.rm = TRUE),
    mean_precision = mean(precision_vals, na.rm = TRUE),
    mean_recall = mean(recall_vals, na.rm = TRUE),
    mean_f1_score = mean(f1_scores, na.rm = TRUE)
  )
  
  return(results)
}

# Run the evaluation 100 times
set.seed(123)
resultstask2 <- evaluate_model(dataset, 100)

# Print the results
print(paste("Mean Accuracy:", resultstask2$mean_accuracy))
print(paste("Standard Deviation of Accuracy:", resultstask2$sd_accuracy))
print(paste("Mean Precision:", resultstask2$mean_precision))
print(paste("Mean Recall:", resultstask2$mean_recall))
print(paste("Mean F1-Score:", resultstask2$mean_f1_score))

#Evaluating multiple (100) times reduces variance, helps detect overfitting, gives statistical confidence and uses different parts of the data for training.

#Evaluating the training Data provides an indication how well the model learned.

#Evaluating on the test data assesses how well the model generalizes to new, unseen data.

#Accuracy might be misleading, especially with data sets with class imbalance where eg the model learned to always predict the most common class which leads to a high accuracy score but otherwise bad performance

#Alternative quality measures are among aothers:
#Precision (positive predictive value) and recall (sensitivity)

#F1-Score is the harmonic mean of Prcision and Recall and robust against unbalanced classes




#Aufgabe 3

require(Umatrix)
 
dataset3 <- scaledData

neuralmap = Umatrix::esomTrain(Data = dataset3,  
Columns = 80,
Lines = 60,
NeighbourhoodFunction = "mexicanhat",
Toroid = F,
NeighbourhoodCooling = "linear",
LearningRateCooling = "linear",
#DataPerEpoch = 1,
StartLearningRate = 0.85,
EndLearningRate = 0.2,
StartRadius = 24, 
EndRadius = 4,
Key = dataset2$Key)

BMUs = neuralmap$BestMatches
ggobj=ProjectionBasedClustering::PlotProjectedPoints(BMUs,
                                                     Cls = targetvar$Cls,
                                                     BMUorProjected = T,
                                                     main = 'Plot of Neural map, marked are BMUs')

UmatrixInformation = Umatrix::umatrixForEsom(
  neuralmap$Weights, 
  Columns = 80,
  Lines = 60,
  Toroid = F
)

GeneralizedUmatrix::plotTopographicMap(UmatrixInformation, Cls = targetvar$Cls,
                                       neuralmap$BestMatches,
                                       BmSize = 1
)

#Based on the visual output from the UMap there is no distinct seperation into clusters , so one would not 
#trust the results from taks 2 , this could be due to the small amount of data or a false application of the 
#Umatrix
