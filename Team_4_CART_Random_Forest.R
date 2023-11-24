#Diabetes Prediction using Linear Regression, kNN and CART
#Data Ingestion and Pre-Processing

rm(list=ls()); gc()
library(rpart); library(rpart.plot)
setwd("C:/Users/aakarshsurendra/Desktop/ISDS 574/Project/Final Dataset/")
dat = read.csv('Diabetes.csv', stringsAsFactors=T, head=T)

#Explore Dataset
str(dat)

hist(dat$Diabetes_012)
hist(dat$BMI)
hist(dat$PhysHlth)
hist(dat$Age)
hist(dat$Education)

#Dimension of the dataset
dim(dat)

#Summary of dataset
summary(dat)

#Columns
colnames(dat)

#Rename the Column Diabetes_012 to Diabetes_Type
colnames(dat)[1] <- "Diabetes_Type"

#Cleaning Dataset

#Missing Value Count
sum(is.na(dat))

#Unique Values

#BMI
sort(unique(dat$BMI))

#General Health
sort(unique(dat$GenHlth))
# 1 = Excellent, 2 = Very Good, 3 = Good, 4 = Fair,  5 = Poor

#Mental Health Scale
sort(unique(dat$MentHlth))

#Physical Health
sort(unique(dat$PhysHlth))

#Age
sort(unique(dat$Age))

#Education
sort(unique(dat$Education))

#Income
sort(unique(dat$Income))

#Diabetes Type
sort(unique(dat$Diabetes_Type))


library(dplyr)

#Replacing the levels of General Health
#Before : 1= Excellent, 2 = Very Good, 3=Good, 4=Fair, 5=Poor
#After : 5= Excellent, 4= Very Good, 3 = Good, 2= Fair, 1=Poor
dat <- dat %>% mutate(GenHlth = recode(GenHlth, `1` = 5, `2` = 4, `3` = 3, `4` = 2, `5` = 1))

counts <- table(dat$Diabetes_Type)

#Changing the values of '2.0' to '1.0' to make it Binary Logistic

dat$Diabetes_Type[dat$Diabetes_Type== 2.0] <- 1.0

counts_before <- table(dat$Diabetes_Type)

#We see a class imbalance of the target variable
#We try to both Oversample minority class(1's) using SMOTE
#Also we do Undersampling of the majority class(0's)

#First we apply SMOTE (Oversampling technique)

library(smotefamily)
library(dplyr)
library(tidyverse)
sum(is.na(dat))
smote <- SMOTE(dat, dat$Diabetes_Type)
oversampled_dataset <- smote$data[,-23]
counts_after_oversampling <- table(oversampled_dataset$Diabetes_Type)

#Exploring dataset after oversampling
sum(is.na(oversampled_dataset))

str(oversampled_dataset)

#We floor the values of variables because after SMOTE, so the synthetic values generated are consistent
oversampled_dataset <- floor(oversampled_dataset)

#Checking the correlation
library(gplots)
correlation_matrix_oversampled <- cor(oversampled_dataset)
heatmap.2(correlation_matrix_oversampled,
        col = colorRampPalette(c("blue", "white", "red"))(20),
        main = "Correlation Heatmap"
        )


#Replacing the floating point variables of Diabetes_Type to Integers
oversampled_dataset$Diabetes_Type <- as.integer(oversampled_dataset$Diabetes_Type)


######Undersampling########
# Identify the indices of the majority class (0's)
indices_majority <- which(dat$Diabetes_Type == 0)

# Sample a subset of indices from the majority class
indices_majority_undersampled <- sample(indices_majority, length(which(dat$Diabetes_Type == 1)))

# Combine indices of the majority and minority classes
indices_undersampled <- c(indices_majority_undersampled, which(dat$Diabetes_Type == 1))

# Create the undersampled dataset
undersampled_dataset <- dat[indices_undersampled, ]


counts_after_undersampling <- table(undersampled_dataset$Diabetes_Type)

install.packages(c("gplots", "RColorBrewer"))
library(gplots)
library(RColorBrewer)
correlation_matrix_undersampled <- cor(undersampled_dataset)
heatmap.2(correlation_matrix_undersampled,
          col = colorRampPalette(c("blue", "white", "red"))(20),
          main = "Correlation Heatmap"
)


#Replacing the floating point variables of Diabetes_Type to Integers
undersampled_dataset$Diabetes_Type <- as.integer(undersampled_dataset$Diabetes_Type)


#Comparing the Target Variable Counts
counts_before
counts_after_undersampling
counts_after_oversampling

##Variable names to be used for further##
dat #Original Dataset
undersampled_dataset #undersampled Dataset
oversampled_dataset #SMOTE oversampled Dataset


#***********************************************************************#
CART
#***********************************************************************#
#*UnderSampled*#
library(rpart)
library(rpart.plot)
#Splitting data
set.seed(151)  # for reproducibility
split_index <- createDataPartition(undersampled_dataset$Diabetes_Type, p = 0.8, list = FALSE)
train_data_cart <- undersampled_dataset[split_index, ]
test_data_cart <- undersampled_dataset[-split_index, ]

# Classification Tree with rpart
fit = rpart(Diabetes_Type ~ ., method="class", data=train_data_cart, minsplit=5) # same as using all other variables as predictors

# Minimum Error Tree
pfit.me = prune(fit, cp = fit$cptable[which.min(fit$cptable[,"xerror"]),"CP"])
rpart.plot(pfit.me, main = 'Min Error Tree')

# Best Pruned Tree
ind = which.min(fit$cptable[,"xerror"]) # xerror: cross-validation error
se1 = fit$cptable[ind,"xstd"]/sqrt(K) # 1 standard error
xer1 = min(fit$cptable[,"xerror"]) + se1 # targeted error: min + 1 SE
ind0 = which.min(abs(fit$cptable[1:ind,"xerror"] - xer1)) # select the tree giving closest xerror to xer1
pfit.bp = prune(fit, cp = fit$cptable[ind0,"CP"])
rpart.plot(pfit.bp, main = 'Best Pruned Tree')

## Prediction
# Using the default threshold of 0.5
yhat = predict(pfit.me, test_data_cart, type = "class")

# Check the lengths of yhat and undersampled_dataset$Diabetes_Type
if (length(yhat) != length(undersampled_dataset$Diabetes_Type)) {
  stop("Lengths of yhat and undersampled_dataset$Diabetes_Type do not match.")
}

# Check for missing values
if (any(is.na(yhat)) || any(is.na(undersampled_dataset$Diabetes_Type))) {
  stop("There are missing values in yhat or undersampled_dataset$Diabetes_Type.")
}

# Create confusion matrix
conf_matrix = table(yhat, test_data_cart$Diabetes_Type)

# Display confusion matrix
print(conf_matrix)



# Calculate specificity and sensitivity
TN = conf_matrix[1, 1]  # True Negatives
FP = conf_matrix[1, 2]  # False Positives
FN = conf_matrix[2, 1]  # False Negatives
TP = conf_matrix[2, 2]  # True Positives

specificity = TN / (TN + FP)
sensitivity = TP / (TP + FN)

accuracy = (TP + TN) / sum(conf_matrix)

# Output specificity and sensitivity
cat("Specificity:", specificity, "\n")
cat("Sensitivity:", sensitivity, "\n")

accuracy

# If you want to use a different cutoff (0.5 in this case)
prob1 = predict(pfit.me, test_data_cart, type = "prob")[, 2]
pred.class = as.numeric(prob1 > 0.1)
ytest = as.numeric(test_data_cart$Diabetes_Type) - 1
err.me.newCut = mean(pred.class != ytest)

# Calculate specificity and sensitivity with the new cutoff
conf_matrix_newCut = table(pred.class, ytest)
TN_newCut = conf_matrix_newCut[1, 1]
FP_newCut = conf_matrix_newCut[1, 2]
FN_newCut = conf_matrix_newCut[2, 1]
TP_newCut = conf_matrix_newCut[2, 2]

specificity_newCut = TN_newCut / (TN_newCut + FP_newCut)
sensitivity_newCut = TP_newCut / (TP_newCut + FN_newCut)

accuracy_newCut = (TP + TN) / sum(conf_matrix)

# Output specificity and sensitivity with the new cutoff
cat("Specificity with new cutoff:", specificity_newCut, "\n")
cat("Sensitivity with new cutoff:", sensitivity_newCut, "\n")
accuracy_newCut
#*****************************************************************#
#*Oversampled*#

#Splitting data
set.seed(161)  # for reproducibility
split_index <- createDataPartition(oversampled_dataset$Diabetes_Type, p = 0.8, list = FALSE)
train_data_cart <- oversampled_dataset[split_index, ]
test_data_cart <- oversampled_dataset[-split_index, ]

# Classification Tree with rpart
fit = rpart(Diabetes_Type ~ ., method="class", data=oversampled_dataset, minsplit=5) # same as using all other variables as predictors

# Minimum Error Tree
pfit.me = prune(fit, cp = fit$cptable[which.min(fit$cptable[,"xerror"]),"CP"])
rpart.plot(pfit.me, main = 'Min Error Tree')

# Best Pruned Tree
ind = which.min(fit$cptable[,"xerror"]) # xerror: cross-validation error
se1 = fit$cptable[ind,"xstd"]/sqrt(K) # 1 standard error
xer1 = min(fit$cptable[,"xerror"]) + se1 # targeted error: min + 1 SE
ind0 = which.min(abs(fit$cptable[1:ind,"xerror"] - xer1)) # select the tree giving closest xerror to xer1
pfit.bp = prune(fit, cp = fit$cptable[ind0,"CP"])
rpart.plot(pfit.bp, main = 'Best Pruned Tree')

## Prediction
# Using the default threshold of 0.5
yhat = predict(pfit.bp, test_data_cart, type = "class")

# Check the lengths of yhat and undersampled_dataset$Diabetes_Type
if (length(yhat) != length(undersampled_dataset$Diabetes_Type)) {
  stop("Lengths of yhat and undersampled_dataset$Diabetes_Type do not match.")
}

# Check for missing values
if (any(is.na(yhat)) || any(is.na(undersampled_dataset$Diabetes_Type))) {
  stop("There are missing values in yhat or undersampled_dataset$Diabetes_Type.")
}

# Create confusion matrix
conf_matrix = table(yhat, test_data_cart$Diabetes_Type)

# Display confusion matrix
print(conf_matrix)



# Calculate specificity and sensitivity
TN = conf_matrix[1, 1]  # True Negatives
FP = conf_matrix[1, 2]  # False Positives
FN = conf_matrix[2, 1]  # False Negatives
TP = conf_matrix[2, 2]  # True Positives

specificity = TN / (TN + FP)
sensitivity = TP / (TP + FN)

accuracy = (TP + TN) / sum(conf_matrix)

# Output specificity and sensitivity
cat("Specificity:", specificity, "\n")
cat("Sensitivity:", sensitivity, "\n")

accuracy

# If you want to use a different cutoff (0.5 in this case)
prob1 = predict(pfit.bp, test_data_cart, type = "prob")[, 2]
pred.class = as.numeric(prob1 > 0.1)
ytest = as.numeric(test_data_cart$Diabetes_Type) - 1
err.bp.newCut = mean(pred.class != ytest)

# Calculate specificity and sensitivity with the new cutoff
conf_matrix_newCut = table(pred.class, ytest)
TN_newCut = conf_matrix_newCut[1, 1]
FP_newCut = conf_matrix_newCut[1, 2]
FN_newCut = conf_matrix_newCut[2, 1]
TP_newCut = conf_matrix_newCut[2, 2]

specificity_newCut = TN_newCut / (TN_newCut + FP_newCut)
sensitivity_newCut = TP_newCut / (TP_newCut + FN_newCut)

accuracy_newCut = (TP + TN) / sum(conf_matrix)

# Output specificity and sensitivity with the new cutoff
cat("Specificity with new cutoff:", specificity_newCut, "\n")
cat("Sensitivity with new cutoff:", sensitivity_newCut, "\n")
accuracy_newCut


##***********************************##
RANDOM FOREST
##***********************************##
##*Undersampled*##
# Install and load the randomForest package
install.packages("randomForest")
install.packages("caret")
library(caret)
library(randomForest)


# Split your data into training and testing sets
set.seed(123)  # for reproducibility
split_index <- createDataPartition(undersampled_dataset$Diabetes_Type, p = 0.8, list = FALSE)
train_data <- undersampled_dataset[split_index, ]
test_data <- undersampled_dataset[-split_index, ]

train_data$Diabetes_Type <- as.factor(train_data$Diabetes_Type)

# Train the Random Forest model
# Adjust the formula based on your actual variable names
rf_model <- randomForest(Diabetes_Type ~ ., data = train_data, ntree = 100)

# Make predictions on the test set
predictions <- predict(rf_model, newdata = test_data)

# Evaluate the model performance
confusion_matrix_rf <- table(predictions, test_data$Diabetes_Type)
print(confusion_matrix_rf)

# Calculate accuracy
accuracy_rf_undersampling <- sum(diag(confusion_matrix_rf)) / sum(confusion_matrix_rf)
print(paste("Accuracy:", round(accuracy, 4)))

sensitivity <- confusion_matrix_rf[2, 2] / sum(confusion_matrix_rf[2, ])
specificity <- confusion_matrix_rf[1, 1] / sum(confusion_matrix_rf[1, ])

print(paste("Sensitivity:", round(sensitivity, 4)))
print(paste("Specificity:", round(specificity, 4)))
#**********Oversampled***********#

# Split your data into training and testing sets
set.seed(129)  # for reproducibility
split_index <- createDataPartition(oversampled_dataset$Diabetes_Type, p = 0.8, list = FALSE)
train_data <- oversampled_dataset[split_index, ]
test_data <- oversampled_dataset[-split_index, ]

train_data$Diabetes_Type <- as.factor(train_data$Diabetes_Type)

# Train the Random Forest model
# Adjust the formula based on your actual variable names
rf_model_oversampling <- randomForest(Diabetes_Type ~ ., data = train_data, ntree = 100)

# Make predictions on the test set
predictions <- predict(rf_model_oversampling, newdata = test_data)

# Evaluate the model performance
confusion_matrix_rf <- table(predictions, test_data$Diabetes_Type)
print(confusion_matrix_rf)

# Calculate accuracy
accuracy_rf_smote <- sum(diag(confusion_matrix_rf)) / sum(confusion_matrix_rf)
print(paste("Accuracy:", round(accuracy, 4)))

sensitivity <- confusion_matrix_rf[2, 2] / sum(confusion_matrix_rf[2, ])
specificity <- confusion_matrix_rf[1, 1] / sum(confusion_matrix_rf[1, ])

print(paste("Sensitivity:", round(sensitivity, 4)))
print(paste("Specificity:", round(specificity, 4)))

###**Complete Dataset**###

# Split your data into training and testing sets
set.seed(131)  # for reproducibility
split_index <- createDataPartition(dat$Diabetes_Type, p = 0.8, list = FALSE)
train_data <- oversampled_dataset[split_index, ]
test_data <- oversampled_dataset[-split_index, ]

train_data$Diabetes_Type <- as.factor(train_data$Diabetes_Type)

# Train the Random Forest model
# Adjust the formula based on your actual variable names
rf_model_complete <- randomForest(Diabetes_Type ~ ., data = train_data, ntree = 100)

# Make predictions on the test set
predictions <- predict(rf_model_complete, newdata = test_data)

# Evaluate the model performance
confusion_matrix_rf <- table(predictions, test_data$Diabetes_Type)
print(confusion_matrix)

# Calculate accuracy
accuracy_rf_fullmodel <- sum(diag(confusion_matrix_rf)) / sum(confusion_matrix_rf)
print(paste("Accuracy:", round(accuracy, 4)))

sensitivity <- confusion_matrix_rf[2, 2] / sum(confusion_matrix_rf[2, ])
specificity <- confusion_matrix_rf[1, 1] / sum(confusion_matrix_rf[1, ])

print(paste("Sensitivity:", round(sensitivity, 4)))
print(paste("Specificity:", round(specificity, 4)))
