library(Boruta)
library(randomForest)
library(readr)
# # install.packages("mlbench")
# library(mlbench)
# data("Ozone")
# Ozone <- na.omit(Ozone)
# set.seed(1)
# Boruta.Ozone <- Boruta(V4 ~ ., data = Ozone, doTrace = 2, ntree = 500)
# Load the data
player <- read_csv("../datafrom200/players.csv")

# Handle missing data
player <- na.omit(player)

# Select the columns for X and Y
x_list <- names(player)[8:(ncol(player)-5)]
X <- player[x_list]
Y <- player["value"]
print(x_list)

# Combine X and Y into a data frame
data <- cbind(X,Y)
 
# Run Boruta
set.seed(123)
boruta.output <- Boruta(value ~ ., data=data, doTrace=2)

print(boruta.output)
plot(boruta.output)
print(getConfirmedFormula(boruta.output))
print(attStats(boruta.output))