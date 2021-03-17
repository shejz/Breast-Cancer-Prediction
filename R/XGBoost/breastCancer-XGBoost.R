# DataSet : Breast Cancer DataSet from UCI
# Cancer Identification using Boosting Ensemble Techniques

# Boosting
# 01. XGBoost      : xgbLinear - Binary & MultiClass Classification
# 02. XGBoost      : xgbTree   - Binary & MultiClass Classification
# 03. Boosted Tree : bstTree   - Binary & MultiClass Classification
# 04. SGB          : gbm       - Binary & MultiClass Classification
# 05. AdaBoost     : adaboost  - Binary & MultiClass Classification
# 

# Load necessary Libraries
library(DBI)
library(RMySQL)
library(corrgram)
library(caret)

# For MAC OSX and Unix like System
# library(doMC)
# registerDoMC(cores = 4)

# set working directory where CSV is located
getwd()
setwd("/Users/shielaj/Desktop/Breast Cancer")
getwd()

# Load the DataSets: 
dataSet <- read.csv("BreastCancerWisconsin.data.csv", header = FALSE, sep = ',')
colnames(dataSet) <- c('SampleCodeNumber', 
                       'ClumpThickness', 
                       'CellSize', 
                       'CellShape',
                       'MarginalAdhesion', 
                       'EpithelialCellSize', 
                       'BareNuclei',
                       'BlandChromatin', 
                       'NormalNucleoli', 
                       'Mitoses',
                       'Class')  # Class:2 for benign
                                 # Class:4 for malignant

# Print top 10 rows in the dataSet
head(dataSet, 10)

# Print last 10 rows in the dataSet
tail(dataSet, 10)

# Dimention of Dataset
dim(dataSet)

# Check Data types of each column
table(unlist(lapply(dataSet, class)))

#Check column names
colnames(dataSet)

# Check Data types of individual column
data.class(dataSet$SampleCodeNumber)
data.class(dataSet$ClumpThickness)
data.class(dataSet$CellSize)
data.class(dataSet$CellShape)
data.class(dataSet$MarginalAdhesion)
data.class(dataSet$EpithelialCellSize)
data.class(dataSet$BareNuclei)
data.class(dataSet$BlandChromatin)
data.class(dataSet$NormalNucleoli)
data.class(dataSet$Mitoses)
data.class(dataSet$Class)

# Change the Data type of variables from numeric to character and vice versa
dataSet$BareNuclei = as.numeric(dataSet$BareNuclei)
dataSet$Class = as.character(dataSet$Class)

data.class(dataSet$BareNuclei)
data.class(dataSet$Class)

## Connect to a MySQL Database 
# create a MySQL driver 
m = dbDriver("MySQL")
myHost <- 'localhost' #'127.0.0.1'
myUsername = 'root'
myDbname = 'datasciencerecipes'
myPort = 3306
myPassword = 'root888'
con = dbConnect(m, user= myUsername, host= myHost, password= myPassword, dbname= myDbname, port= myPort)

if(dbIsValid(con)) {
  print('MySQL Connection is Successful')
} else {print('MySQL Connection is Unsuccessful')}

# Export DataFrame to a MySQL table 
response <- dbWriteTable(conn = con, name = 'breastcancerdata', value = dataSet, 
                         row.names = FALSE, overwrite = TRUE)
if(response) {print('Data export to MySQL is successful')
} else {print('Data export to MySQL is unsuccessful')}

## Write a query here and execute it to retrive data from MySQL Database
sql = 'SELECT * 
FROM breastcancerdata;'
result = dbSendQuery(conn = con, statement = sql)
dataset <- dbFetch(res = result, n = -1)
dbClearResult(result)
dbDisconnect(conn = con)

## Check dataset that retrived from MySQL database
# Print top 10 rows in the dataSet
head(dataset, 10)
# Print last 10 rows in the dataSet
tail(dataset, 10)
# Dimention of Dataset
dim(dataset)
# Check Data types of each column
table(unlist(lapply(dataset, class)))
#Check column names
colnames(dataset)

# Change the Data type of Class variables to "character"
dataset$Class = as.character(dataset$Class)
data.class(dataset$Class)

## Exploring or Summarising dataset with descriptive statistics
#  Find out if there is missing value
rowSums(is.na(dataset))
colSums(is.na(dataset))
# Missing data treatment if exists
#dataset[dataset$columnName=="& ","columnName"] <- NA 
#drop columns
#dataset <- within(dataset, rm(columnName))

# Summary of dataset
#lapply - When you want to apply a function to each element of a list in turn and get a list back.
lapply(dataset[2:10], FUN = sum)
lapply(dataset[2:10], FUN = mean)
lapply(dataset[2:10], FUN = median)
lapply(dataset[2:10], FUN = min)
lapply(dataset[2:10], FUN = max)
lapply(dataset[2:10], FUN = length)

#sapply - When you want to apply a function to each element of a list in turn, 
#but you want a vector back, rather than a list.
sapply(dataset[2:10], FUN = sum)
sapply(dataset[2:10], FUN = mean)
sapply(dataset[2:10], FUN = median)
sapply(dataset[2:10], FUN = min)
sapply(dataset[2:10], FUN = max)
sapply(dataset[2:10], FUN = length)

# Using Aggregate FUNCTION
aggregate(dataset$ClumpThickness, list(dataset$Class), summary)
aggregate(dataset$CellSize, list(dataset$Class), summary)
aggregate(dataset$CellShape, list(dataset$Class), summary)
aggregate(dataset$MarginalAdhesion, list(dataset$Class), summary)
aggregate(dataset$EpithelialCellSize, list(dataset$Class), summary)
aggregate(dataset$BareNuclei, list(dataset$Class), summary)
aggregate(dataset$BlandChromatin, list(dataset$Class), summary)
aggregate(dataset$NormalNucleoli, list(dataset$Class), summary)
aggregate(dataset$Mitoses, list(dataset$Class), summary)

# Using "by"
by(dataset[2:10], dataset[11], FUN = summary)
by(dataset[2:10], dataset$Class, FUN = summary)

## Visualising DataSet
# Print Column Names
colnames(dataset)
# Print Data Types of each column
for(i in 2:length(dataset)) {
  print(data.class(dataset[,i]))
}

# Histogram
par(mfrow=c(3,3))

x <- dataset$ClumpThickness
hist(x,  xlab = "ClumpThickness", ylab = "Count", main = "")
x <- dataset$CellSize
hist(x,  xlab = "CellSize", ylab = "Count", main = "")
x <- dataset$CellShape
hist(x,  xlab = "CellShape", ylab = "Count", main = "")

x <- dataset$MarginalAdhesion
hist(x,  xlab = "MarginalAdhesion", ylab = "Count", main = "")
x <- dataset$EpithelialCellSize
hist(x,  xlab = "EpithelialCellSize", ylab = "Count", main = "")
x <- dataset$BareNuclei
hist(x,  xlab = "BareNuclei", ylab = "Count", main = "")

x <- dataset$BlandChromatin
hist(x,  xlab = "BlandChromatin", ylab = "Count", main = "")
x <- dataset$NormalNucleoli
hist(x,  xlab = "NormalNucleoli", ylab = "Count", main = "")
x <- dataset$Mitoses
hist(x,  xlab = "Mitoses", ylab = "Count", main = "")


# Histogram with Density graph
par(mfrow=c(3,3))

x <- dataset$ClumpThickness
h <- hist(x,  xlab = "", ylab = "Count", main = "")
xfit<-seq(min(x),max(x),length=40) 
yfit<-dnorm(xfit,mean=mean(x),sd=sd(x)) 
yfit <- yfit*diff(h$mids[1:2])*length(x) 
lines(xfit, yfit, col="darkblue", lwd=2)

x <- dataset$CellSize
h <- hist(x,  xlab = "", ylab = "Count", main = "")
xfit<-seq(min(x),max(x),length=40) 
yfit<-dnorm(xfit,mean=mean(x),sd=sd(x)) 
yfit <- yfit*diff(h$mids[1:2])*length(x) 
lines(xfit, yfit, col="darkblue", lwd=2)

x <- dataset$CellShape
h <- hist(x,  xlab = "", ylab = "Count", main = "")
xfit<-seq(min(x),max(x),length=40) 
yfit<-dnorm(xfit,mean=mean(x),sd=sd(x)) 
yfit <- yfit*diff(h$mids[1:2])*length(x) 
lines(xfit, yfit, col="darkblue", lwd=2)

x <- dataset$MarginalAdhesion
h <- hist(x,  xlab = "", ylab = "Count", main = "")
xfit<-seq(min(x),max(x),length=40) 
yfit<-dnorm(xfit,mean=mean(x),sd=sd(x)) 
yfit <- yfit*diff(h$mids[1:2])*length(x) 
lines(xfit, yfit, col="darkblue", lwd=2)

x <- dataset$EpithelialCellSize
h <- hist(x,  xlab = "", ylab = "Count", main = "")
xfit<-seq(min(x),max(x),length=40) 
yfit<-dnorm(xfit,mean=mean(x),sd=sd(x)) 
yfit <- yfit*diff(h$mids[1:2])*length(x) 
lines(xfit, yfit, col="darkblue", lwd=2)

x <- dataset$BareNuclei
h <- hist(x,  xlab = "", ylab = "Count", main = "")
xfit<-seq(min(x),max(x),length=40) 
yfit<-dnorm(xfit,mean=mean(x),sd=sd(x)) 
yfit <- yfit*diff(h$mids[1:2])*length(x) 
lines(xfit, yfit, col="darkblue", lwd=2)

x <- dataset$BlandChromatin
h <- hist(x,  xlab = "", ylab = "Count", main = "")
xfit<-seq(min(x),max(x),length=40) 
yfit<-dnorm(xfit,mean=mean(x),sd=sd(x)) 
yfit <- yfit*diff(h$mids[1:2])*length(x) 
lines(xfit, yfit, col="darkblue", lwd=2)

x <- dataset$NormalNucleoli
h <- hist(x,  xlab = "", ylab = "Count", main = "")
xfit<-seq(min(x),max(x),length=40) 
yfit<-dnorm(xfit,mean=mean(x),sd=sd(x)) 
yfit <- yfit*diff(h$mids[1:2])*length(x) 
lines(xfit, yfit, col="darkblue", lwd=2)

x <- dataset$Mitoses
h <- hist(x,  xlab = "", ylab = "Count", main = "")
xfit<-seq(min(x),max(x),length=40) 
yfit<-dnorm(xfit,mean=mean(x),sd=sd(x)) 
yfit <- yfit*diff(h$mids[1:2])*length(x) 
lines(xfit, yfit, col="darkblue", lwd=2)

# Barplot of categorical data
par(mfrow=c(2,2))
barplot(table(dataset$Class), ylab = "Count", 
        col=c("darkblue","red", "green"))
barplot(prop.table(table(dataset$Class)), ylab = "Proportion", 
        col=c("darkblue","red", "green"))
barplot(table(dataset$Class), xlab = "Count", horiz = TRUE, 
        col=c("darkblue","red", "green"))
barplot(prop.table(table(dataset$Class)), xlab = "Proportion", horiz = TRUE, 
        col=c("darkblue","red", "green"))

# Box Plot of Numerical Data
par(mfrow=c(3,3))
boxplot(dataset$ClumpThickness, ylab = "")
boxplot(dataset$CellSize, ylab = "")
boxplot(dataset$CellShape, ylab = "")
boxplot(dataset$MarginalAdhesion, ylab = "")
boxplot(dataset$EpithelialCellSize, ylab = "")
boxplot(dataset$BareNuclei, ylab = "")
boxplot(dataset$BlandChromatin, ylab = "")
boxplot(dataset$NormalNucleoli, ylab = "")
boxplot(dataset$Mitoses, ylab = "")

# Scatter Plots
par(mfrow=c(2,2))
plot(dataset$ClumpThickness, pch = 20)
plot(dataset$CellSize, pch = 20)
plot(dataset$CellShape, pch = 20)
plot(dataset$NormalNucleoli, pch = 20)

par(mfrow=c(2,2))
plot(dataset$ClumpThickness, dataset$CellSize, pch = 20)
plot(dataset$CellSize, dataset$CellShape, pch = 20)
plot(dataset$MarginalAdhesion, dataset$EpithelialCellSize, pch = 20)
plot(dataset$Mitoses, dataset$NormalNucleoli, pch = 20)

# Corelation Diagram using "corrgram" package
x <- dataset[2:10]

#x is a data frame with one observation per row.
corrgram(x)

#order=TRUE will cause the variables to be ordered using principal component analysis of the correlation matrix.
corrgram(x, order = TRUE)

# lower.panel= and upper.panel= to choose different options below and above the main diagonal respectively. 

# (the filled portion of the pie indicates the magnitude of the correlation)
# lower.panel=  
corrgram(x, order = TRUE, lower.panel = panel.pie, upper.panel = NULL)
corrgram(x, order = TRUE, lower.panel = panel.shade, upper.panel = NULL)
corrgram(x, order = TRUE, lower.panel = panel.ellipse, upper.panel = NULL)
corrgram(x, order = TRUE, lower.panel = panel.pts, upper.panel = NULL)

#off diagonal panels
# lower.panel= & upper.panel=
corrgram(x, order = TRUE, lower.panel = panel.pie, upper.panel = panel.pts)
corrgram(x, order = TRUE, lower.panel = panel.shade, upper.panel = panel.pie)
corrgram(x, order = TRUE, lower.panel = panel.ellipse, upper.panel = panel.shade)
corrgram(x, order = TRUE, lower.panel = panel.pts, upper.panel = panel.pie)

# upper.panel=
corrgram(x, order = TRUE, lower.panel = NULL, upper.panel = panel.pts)
corrgram(x, order = TRUE, lower.panel = NULL, upper.panel = panel.pie)
corrgram(x, order = TRUE, lower.panel = NULL, upper.panel = panel.shade)
corrgram(x, order = TRUE, lower.panel = NULL, upper.panel = panel.ellipse)

#text.panel= and diag.panel= refer to the main diagnonal. 
corrgram(x, order = TRUE, lower.panel = panel.pie, upper.panel = panel.pts,
         text.panel=panel.txt)
corrgram(x, order = TRUE, lower.panel = panel.pie, upper.panel = panel.pts,
         text.panel=panel.txt, diag.panel=panel.minmax,
         main="Correlation Diagram")

# Pie Chart
par(mfrow=c(1,1))
x <- table(dataset$Class)
lbls <- paste(names(x), "\nTotal Count:", x, sep="")
pie(x, labels = lbls, main="Pie Chart of Different Classes", col = c("red","blue","green"))

## Pre-Processing of DataSet i.e. train : test split
train_test_index <- createDataPartition(dataset$Class, p=0.67, list=FALSE)
training_dataset <- dataset[train_test_index,]
testing_dataset <- dataset[-train_test_index,]

dim(training_dataset)
dim(testing_dataset)

## Evaluating Algorithm i.e. training, testing and evaluation
# Check available ML Algorithms
names(getModelInfo())

## Turn Off warnings
options( warn = -1 )

## Turn On warnings
#options( warn = 0 )

##############################################################
# cross Validation and control parameter setup
##############################################################
control <- trainControl(method="repeatedcv", # repeatedcv / adaptive_cv
                        number=3, repeats = 3, verbose = TRUE, 
                        search = "grid",
                        allowParallel = TRUE)

###################################################################
###################################################################
# Machine Learning Algorithm and parameter tuning 
# 1. Without paramet tuning or using default

## There three ways of parameter tuning
# 2. Using Data Pre-Processing: 
# caret method -> preProcess 
# default value is NULL
# other value ["BoxCox", "YeoJohnson", "expoTrans", "center", 
#              "scale", "range", "knnImpute", "bagImpute", 
#              "medianImpute", "pca", "ica" and "spatialSign"]

# 3. Using Automatic Grid
# caret method -> tuneLength [Note: it takes INTEGER Val]
# Example: tuneLength = 3

# 4. Using Manual Grid
# caret method -> tuneGrid [Note: grid needs to be defined manually]
# Example: grid <- expand.grid(size=c(5,10), k=c(3,4,5)) [parameters of LVQ]
#          tuneGrid = grid   

###################################################################
###################################################################

# -----------------------------------------------------------------------------
## 1. Training - without explicit parameter tuning / using default
# Training using Boosting and Neural Network Algorithms
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# 01. XGBoost     : xgbLinear - Binary & MultiClass Classification
# 02. XGBoost     : xgbTree   - Binary & MultiClass Classification
# 03. Boosted Tree: bstTree   - Binary & MultiClass Classification
# 04. SGB : gbm               - Binary & MultiClass Classification
# 05. Ada Boosting : adaboost - Binary & MultiClass Classification
# -----------------------------------------------------------------------------
# 01. XGBoost : xgbLinear  
fit.xgbL_1 <- caret::train(Class~., data=training_dataset[2:11], method="xgbLinear", 
                           metric="Accuracy", trControl=control)
print(fit.xgbL_1)
#plot(fit.xgbL_1)

# 02. XGBoost : xgbTree  
fit.xgbT_1 <- caret::train(Class~., data=training_dataset[2:11], method="xgbTree", 
                           metric="Accuracy", trControl=control)
print(fit.xgbT_1)
#plot(fit.xgbT_1)

# 03. Boosted Tree : bstTree 
fit.bstT_1 <- caret::train(Class~., data=training_dataset[2:11], method="bstTree", 
                           metric="Accuracy", trControl=control)
print(fit.bstT_1)
#plot(fit.bstT_1)

# 04. Stochastic Gradient Boosting : gbm 
fit.gbm_1 <- caret::train(Class~., data=training_dataset[2:11], method="gbm", 
                          metric="Accuracy", trControl=control)
print(fit.gbm_1)
#plot(fit.gbm_1)

# 05. Ada Boosting : adaboost 
fit.ada_1 <- caret::train(Class~., data=training_dataset[2:11], method="adaboost", 
                          metric="Accuracy", trControl=control)
print(fit.ada_1)
#plot(fit.ada_1)

# -----------------------------------------------------------------------------
## 2. Training - with explicit parameter tuning using preProcess method
# Training using Boosting and Neural Network Algorithms
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# 01. XGBoost     : xgbLinear - Binary & MultiClass Classification
# 02. XGBoost     : xgbTree   - Binary & MultiClass Classification
# 03. Boosted Tree: bstTree   - Binary & MultiClass Classification
# 04. SGB : gbm               - Binary & MultiClass Classification
# 05. Ada Boosting : adaboost - Binary & MultiClass Classification
# -----------------------------------------------------------------------------

# 01. XGBoost : xgbLinear - MultiClass Classification 
fit.xgbL_2 <- caret::train(Class~., data=training_dataset[2:11], method="xgbLinear", 
                           metric="Accuracy", trControl=control,
                           preProcess = c('center', 'scale','pca'))
print(fit.xgbL_2)
#plot(fit.xgbL_2)

# 02. XGBoost : xgbTree - MultiClass Classification 
fit.xgbT_2 <- caret::train(Class~., data=training_dataset[2:11], method="xgbTree", 
                           metric="Accuracy", trControl=control,
                           preProcess = c('center', 'scale','pca'))
print(fit.xgbT_2)
#plot(fit.xgbT_2)

# 03. Boosted Tree : bstTree - MultiClass Classification 
fit.bstT_2 <- caret::train(Class~., data=training_dataset[2:11], method="bstTree", 
                           metric="Accuracy", trControl=control,
                           preProcess = c('center', 'scale','pca'))
print(fit.bstT_2)
#plot(fit.bstT_2)

# 04. Stochastic Gradient Boosting : gbm 
fit.gbm_2 <- caret::train(Class~., data=training_dataset[2:11], method="gbm", 
                          metric="Accuracy", trControl=control,
                          preProcess = c('center', 'scale','pca'))
print(fit.gbm_2)
#plot(fit.gbm_2)

# 05. Ada Boosting : adaboost 
fit.ada_2 <- caret::train(Class~., data=training_dataset[2:11], method="adaboost", 
                          metric="Accuracy", trControl=control,
                          preProcess = c('center', 'scale','pca'))
print(fit.ada_2)
#plot(fit.ada_2)

# -----------------------------------------------------------------------------
## 3. Training - with explicit parameter tuning using preProcess method 
## & Automatic Grid i.e. tuneLength
## Training using Boosting and Neural Network Algorithms
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# 01. XGBoost     : xgbLinear - Binary & MultiClass Classification
# 02. XGBoost     : xgbTree   - Binary & MultiClass Classification
# 03. Boosted Tree: bstTree   - Binary & MultiClass Classification
# 04. SGB : gbm               - Binary & MultiClass Classification
# 05. Ada Boosting : adaboost - Binary & MultiClass Classification
# -----------------------------------------------------------------------------

# 01. XGBoost : xgbLinear - MultiClass Classification 
fit.xgbL_3 <- caret::train(Class~., data=training_dataset[2:11], method="xgbLinear", 
                           metric="Accuracy", trControl=control,
                           preProcess = c('center', 'scale','pca'), 
                           tuneLength = 3)
print(fit.xgbL_3)
#plot(fit.xgbL_3)

# 02. XGBoost : xgbTree - MultiClass Classification 
fit.xgbT_3 <- caret::train(Class~., data=training_dataset[2:11], method="xgbTree", 
                           metric="Accuracy", trControl=control,
                           preProcess = c('center', 'scale','pca'),
                           tuneLength = 3)
print(fit.xgbT_3)
#plot(fit.xgbT_3)

# 03. Boosted Tree : bstTree - MultiClass Classification 
fit.bstT_3 <- caret::train(Class~., data=training_dataset[2:11], method="bstTree", 
                           metric="Accuracy", trControl=control,
                           preProcess = c('center', 'scale','pca'),
                           tuneLength = 3)
print(fit.bstT_3)
#plot(fit.bstT_3)

# 04. Stochastic Gradient Boosting : gbm 
fit.gbm_3 <- caret::train(Class~., data=training_dataset[2:11], method="gbm", 
                          metric="Accuracy", trControl=control,
                          preProcess = c('center', 'scale','pca'),
                          tuneLength = 3)
print(fit.gbm_3)
#plot(fit.gbm_3)

# 05. Ada Boosting : adaboost 
fit.ada_3 <- caret::train(Class~., data=training_dataset[2:11], method="adaboost", 
                          metric="Accuracy", trControl=control,
                          preProcess = c('center', 'scale','pca'),
                          tuneLength = 3)
print(fit.ada_3)
#plot(fit.ada_3)

# collect resampling statistics of ALL trained models
results <- resamples(list(XGBL_1      = fit.xgbL_1, 
                          XGBT_1      = fit.xgbT_1,
                          BT_1        = fit.bstT_1,
                          GBM_1       = fit.gbm_1,
                          ADA_1       = fit.ada_1,
                          
                          XGBL_2      = fit.xgbL_2, 
                          XGBT_2      = fit.xgbT_2,
                          BT_2        = fit.bstT_2,
                          GBM_2       = fit.gbm_2,
                          ADA_2       = fit.ada_2,
                          
                          XGBL_3      = fit.xgbL_3, 
                          XGBT_3      = fit.xgbT_3,
                          BT_3        = fit.bstT_3,
                          GBM_3       = fit.gbm_3,
                          ADA_3       = fit.ada_3
))

# Summarize the fitted models
summary(results)
# Plot and rank the fitted models
dotplot(results)

# Test skill of the BEST trained model on validation/testing dataset
predictions_XGBT_2 <- predict(fit.xgbT_2, newdata=testing_dataset)

# Evaluate the BEST trained model and print results
res_XGBT_2  <- caret::confusionMatrix(predictions_XGBT_2, testing_dataset$Class)

print("Results from the BEST trained model ... ...\n"); 
print(res_XGBT_2) 
print(round(res_XGBT_1$overall, digits = 3))

## Save the BEST trained model to disk
final_model <- fit.xgbT_2;        saveRDS(final_model, "./final_model_XGBT_2.rds")

# Connecting a MySQL Database
m = dbDriver("MySQL")
myHost <- 'localhost' #'127.0.0.1'
myUsername = 'root'
myDbname = 'datasciencerecipes'
myPort = 3306
myPassword = 'root888'
con = dbConnect(m, user= myUsername, host= myHost, password= myPassword, dbname= myDbname, port= myPort)

if(dbIsValid(con)) {
  print('MySQL Connection is Successful')
} else {print('MySQL Connection is Unsuccessful')}

## Write a query here and execute it to retrive data from MySQL Database
sql = 'SELECT *
FROM breastcancerdata;'
result = dbSendQuery(conn = con, statement = sql)
dataset <- dbFetch(res = result)
dbClearResult(result)
dim(dataset)

## Load the trained model from disk
trained_model <- readRDS("./final_model_XGBT_2.rds")

# make a predictions on "new data" using the final model
cols <- c(2:10)
predicted_Class <- predict(trained_model, dataset[cols])
predicted_ClassProb <- predict(trained_model, dataset[cols], 
                               type = "prob")

# Save result in a CSV file and/ or MySQL Table
result <- data.frame(predicted_Class)
dim(result)
dim(dataset)

# merge prediction with dataset 
finalResult <- cbind(dataset, result)
dim(finalResult)

# in CSV file
cols <- c(1, 11, 12)
write.csv(finalResult[cols], file = "finalResult.csv", row.names = FALSE)

# in MySQL Table
dbWriteTable(conn = con, name = 'breastcancerresult', value = finalResult[cols], 
             row.names = FALSE, overwrite = TRUE)
dbDisconnect(conn = con)

# KAPPA Interpretation :
# it measures how much better the classier is comparing with guessing with the target distribution.
# Poor agreement        = 0.20 or less
# Fair agreement        = 0.20 to 0.40
# Moderate agreement    = 0.40 to 0.60
# Good agreement        = 0.60 to 0.80
# Very good agreement   = 0.80 to 1.00
