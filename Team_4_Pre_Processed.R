#Diabetes Prediction using Linear Regression, kNN and CART
#Data Ingestion and Pre-Processing

rm(list=ls()); gc()
library(rpart); library(rpart.plot)
setwd("C:/Users/aakarshsurendra/Desktop/ISDS 574/Project/Final Dataset/")
dat = read.csv('Diabetes.csv', stringsAsFactors=T, head=T)

#Explore Dataset
str(dat)

#Dimension of the dataset
dim(dat)

#Summary of dataset
summary(dat)

#Taking sample of 1000 from the dataset
sample_index <- sample(nrow(dat),1000)
sample_dataset <- dat[sample_index,]
rownames(sample_dataset) <- NULL

#Columns
colnames(sample_dataset)

#Rename the Column Diabetes_012 to Diabetes_Type
colnames(sample_dataset)[1] <- "Diabetes_Type"

#Cleaning Dataset

#Missing Value Count
sum(is.na(sample_dataset))

#Unique Values

#BMI
sort(unique(sample_dataset$BMI))

#General Health
sort(unique(sample_dataset$GenHlth))
# 1 = Excellent, 2 = Very Good, 3 = Good, 4 = Fair,  5 = Poor

#Mental Health Scale
sort(unique(sample_dataset$MentHlth))

#Physical Health
sort(unique(sample_dataset$PhysHlth))

#Age
sort(unique(sample_dataset$Age))

#Education
sort(unique(sample_dataset$Education))

#Income
sort(unique(sample_dataset$Income))

#Diabetes Type
sort(unique(sample_dataset$Diabetes_Type))


library(dplyr)
counts <- table(sample_dataset$Diabetes_Type)

#Changing the values of '2.0' to '1.0' to make it Binary Logistic
#sample_dataset$Diabetes_Type <- ifelse(sample_dataset$Diabetes_Type == 2.0, 1.0, sample_dataset$Diabetes_Type)

sample_dataset$Diabetes_Type[sample_dataset$Diabetes_Type== 2.0] <- 1.0

counts <- table(sample_dataset$Diabetes_Type)

#Preprocessing
#Visualizing the Correlation/Missing Values
#install.packages('gplots')
library(gplots)
correlation_matrix <- cor(sample_dataset)
heatmap.2(correlation_matrix,
        col = colorRampPalette(c("blue", "white", "red"))(20),
        main = "Correlation Heatmap",
        )

#Visualizing for Outliers
#Continuous variables in the dataset - BMI and Age

hist(sample_dataset$BMI)

hist(sample_dataset$Age)

#Replacing 2's with 1's so having only 2 categories : 0 - Non Diabetic, 1 - Diabetic
sample_dataset$Diabetes_Type <- replace(sample_dataset$Diabetes_Type, 2, 1)

#Replacing the floating point variables of Diabetes_Type to Integers
sample_dataset$Diabetes_Type <- as.integer(sample_dataset$Diabetes_Type)

#Replacing the levels of General Health
#Before : 1= Excellent, 2 = Very Good, 3=Good, 4=Fair, 5=Poor
#After : 5= Excellent, 4= Very Good, 3 = Good, 2= Fair, 1=Poor
sample_dataset$GH <- sample_dataset$GenHlth
sample_dataset$GH <- match(sample_dataset$GH, c(1, 2, 3, 4, 5), nomatch = NA)


# Count of Diabetes_Type and HighBP table
diabetes_bp <- as.data.frame(table(sample_dataset$Diabetes_Type, sample_dataset$HighBP))
colnames(diabetes_bp) <- c("Diabetes_Type", "HighBP", "Count")
print(diabetes_bp)

#Changing the name of the sample_dataset to be used for further model building
dataset <- sample_dataset
