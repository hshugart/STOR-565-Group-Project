library(readr)
library(tidyverse)
library(jpeg)
library(grid)
library(gridExtra)
library(OpenImageR)
library(BiocManager)
library("EBImage")
library(randomForest)
library(dplyr)


set.seed(1)


### DATA PRE-PROCESSING

# Set your own path ! 

setwd("C:/Users/Joe/Desktop/UNC/STOR_565_ML/Final_Project/data")  


# Load data

birds <- read.csv("birds.csv")
n_birds = birds %>% filter(data.set == "train") %>% group_by(labels) %>% summarize(n = n()) %>% arrange(desc(n))
birds_train = birds %>% filter(data.set == "train")
birds_test = birds %>% filter(data.set == "test")


## Preprocess Train Data

# Make classification labels a factor
birds_train$labels <- as.factor(birds_train$labels)

# create train dataset and train label vector, which will be inputs for randomForest function later
x_train <- data.frame() 
y_train <- birds_train$labels

# access image data and store each image as observation in x_train
for(img_path in birds_train$filepaths){
  
  img <- readJPEG(img_path)
  img <- resize(img, 64)  # only include this line to make images smaller and processing faster
  img <- as.vector(img)
  img <- img / 255.0  # scale pixel values 
  x_train <- rbind(x_train, img) # add image as observation to dataset
}

dat_train <- cbind(y_train, x_train, deparse.level=1 )


## Preprocess Test Data  (same procedure as for train data)

# Make classification labels a factor
birds_test$labels <- as.factor(birds_test$labels)

x_test <- data.frame() 
y_test <- birds_train$labels

# access image data and store each image as observation in x_train
for(img_path in birds_test$filepaths){
  
  img <- readJPEG(img_path)
  img <- resize(img, 64)  # only include this line to make images smaller and processing faster
  img <- as.vector(img)
  img <- img / 255.0  # scale pixel values 
  x_test <- rbind(x_test, img) # add image as observation to dataset
}



### FIT RANDOM FOREST 

rf_mod <- randomForest(y_train???., data = dat_train, mtry = 12, importance = TRUE)


### TUNE RANDOM FOREST MODEL

### TEST RANDOM FOREST MODEL

# predict bird classes of test data
yhat <- predict(rf_mod, newdata = x_test)

# calculate misclassification error
sum(as.numeric(yhat != y_test)) / length(y_test) 
  

