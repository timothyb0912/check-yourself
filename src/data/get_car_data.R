# Load the mlogit package
library(mlogit)

# Load the Car dataset
data(Car)

# Declare a path where the data will be saved
data_path = '../../data/raw/car_wide_format.csv'

# Write the Car dataset to file as a wide-format CSV
write.csv(Car, file=data_path, row.names = FALSE)
