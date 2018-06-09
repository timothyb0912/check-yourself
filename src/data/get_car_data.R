# Set the current working directory
setwd("~/Documents/model_checking/code/")

# Load the mlogit package
library(mlogit)

# Load the Car dataset
data(Car)

# Write the Car dataset to file as a wide-format CSV
write.csv(Car,
          file='../data/car_wide_format.csv',
          row.names = FALSE)
