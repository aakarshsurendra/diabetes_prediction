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

dataset_2 <- dat %>% filter(Diabetes_Type == 0) %>% slice(sample(n(), 39978))
dataset_1 <- dat %>% filter(Diabetes_Type ==1) %>% slice(sample(n(), 39978))
undersampled_dataset <- rbind(dataset_1,dataset_2)
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