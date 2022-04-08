library(keras)
library(tensorflow)
library(tidyverse)
library(dplyr)

setwd("~/downloads/archive")
birds = read_csv("birds.csv")
birds$filepaths = str_replace(birds$filepaths, "BLACK & YELLOW  BROADBILL", "BLACK & YELLOW BROADBILL")
birds$labels = str_replace(birds$labels, "BLACK & YELLOW  BROADBILL", "BLACK & YELLOW BROADBILL")

bird_list = unique(birds$labels)

output_n <- length(bird_list)
img_width <- 80
img_height <- 80
target_size <- c(img_width, img_height)

# RGB = 3 channels
channels <- 3

# path to image folders
train_image_files_path <- "/Users/henryshugart/Downloads/archive/train/"
valid_image_files_path <- "/Users/henryshugart/Downloads/archive/test/"

train_data_gen = image_data_generator(
  rescale = 1/255 ,
  rotation_range = 40,
  width_shift_range = 0.2,
  height_shift_range = 0.2,
  shear_range = 0.2,
  zoom_range = 0.2,
  horizontal_flip = TRUE,
  fill_mode = "nearest"
)

# Validation data shouldn't be augmented! But it should also be scaled.
valid_data_gen <- image_data_generator(
  rescale = 1/255
)  


train_image_array_gen <- flow_images_from_directory(train_image_files_path, 
                                                    train_data_gen,
                                                    target_size = target_size,
                                                    class_mode = "categorical",
                                                    classes = bird_list
                                                    )

# validation images
valid_image_array_gen <- flow_images_from_directory(valid_image_files_path, 
                                                    valid_data_gen,
                                                    target_size = target_size,
                                                    class_mode = "categorical",
                                                    classes = bird_list
                                                    )
train_samples <- train_image_array_gen$n
# number of validation samples
valid_samples <- valid_image_array_gen$n

# define batch size and number of epochs
batch_size <- 256
epochs <- 10

model <- keras_model_sequential()

# add layers
model %>%
  layer_conv_2d(filter = 32, kernel_size = c(3,3), padding = "same", input_shape = c(img_width, img_height, channels)) %>%
  layer_activation("relu") %>%
  
  # Second hidden layer
  layer_conv_2d(filter = 16, kernel_size = c(3,3), padding = "same") %>%
  layer_activation_leaky_relu(0.3) %>%
  layer_batch_normalization() %>%
  layer_conv_2d(filter = 16, kernel_size = c(3,3), padding = "same") %>%
  layer_activation_leaky_relu(0.3) %>%
  layer_batch_normalization() %>%
  
  # Use max pooling
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(0.25) %>%
  
  # Flatten max filtered output into feature vector 
  # and feed into dense layer
  layer_flatten() %>%
  layer_dense(500) %>%
  layer_activation("relu") %>%
  layer_dense(500) %>%
  layer_activation("relu") %>%
  layer_dropout(0.3) %>%
  
  # Outputs from dense layer are projected onto output layer
  layer_dense(output_n, activation = ) %>% 
  layer_activation("softmax")

# compile
model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = "adam",
  metrics = "accuracy"
)

hist <- model %>% fit(
  # training data
  train_image_array_gen,
  
  # epochs
  batch_size=batch_size, 
  epochs = epochs, 
  
  # validation data
  validation_data = valid_image_array_gen,
  validation_steps = as.integer(valid_samples / batch_size),
  
  # print progress
  verbose = 2
  
)

