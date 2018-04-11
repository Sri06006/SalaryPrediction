rm(list=ls(all=TRUE))
setwd("C:\\Users\\R_files\\Mith")

data = read.csv("train.csv", header=TRUE, na.strings = c("0", "-1", "NA"))

str(data)

data = subset(data, select=-c(Candidate.ID, CollegeCode, CityCode,CityTier))
sum(is.na(data))
library(Amelia)
missmap(data,main="Missing vs Observed")


summary(data)

str(data)
head(data$Score.in.ComputerProgrammin)

data_num = subset(data, select= -c(Discipline,State,Graduation,Board.in.Twelth,Year.Of.Twelth.Completion, School.Board.in.Tenth, Date.Of.Birth,Gender))
data_cat = subset(data, select= c(Discipline,State,Graduation,Board.in.Twelth,Year.Of.Twelth.Completion, School.Board.in.Tenth, Date.Of.Birth,Gender))

#Outliers

outliersZ <- function(data, zCutOff = 1.96, replace = NA, values = FALSE, digits = 2) {
  #compute standard deviation (sample version n = n [not n-1])
  stdev <- sqrt(sum((data - mean(data, na.rm = T))^2, na.rm = T) / sum(!is.na(data)))
  #compute absolute z values for each value
  absZ <- abs(data - mean(data, na.rm = T)) / stdev
  #subset data that has absZ greater than the zCutOff and replace them with replace
  #can also replace with other values (such as max/mean of data)
  data[absZ > zCutOff] <- replace 
  
  if (values == TRUE) {
    return(round(absZ, digits)) #if values == TRUE, return z score for each value
  } else {
    return(round(data, digits)) #otherwise, return values with outliers replaced
  }
}
#Ran Outliers function on Numerical data

data2_or = outliersZ(data_num)
sum(is.na(data2_or))
summary(data2_or)


# data2_num - contains numerical data, data2_cat contains categorical variables
#data2_or contains data after removing outliers

str(data2_or)
str(data2_cat)
#Missing value Imputation
data_num = subset(data, select= -c(Discipline,State,Graduation,Board.in.Twelth,Year.Of.Twelth.Completion, School.Board.in.Tenth, Date.Of.Birth,Gender))
data_cat = subset(data, select= c(Discipline,State,Graduation,Board.in.Twelth,Year.Of.Twelth.Completion, School.Board.in.Tenth, Date.Of.Birth,Gender))





data3 = data2
library(DMwR)
data2 = centralImputation(data2)

data2 = cbind(data_num, data_cat)
#Check data after Imputing
sum(is.na(data2))
str(data2)



#will try PCA to see important components and check variances and correlation between features
rows=seq(1,nrow(data2),1)
set.seed(123)
trainRows=sample(rows,round(nrow(data2)*0.7))
train=data2[trainRows,]
test=data2[-(trainRows),]


pca = princomp(data2)
pca$loadings
summary(pca)
plot(pca)

#Feature Engineering
# Feature Selection
# Year.in Completion, - Graduation Years-  Year in twelth completion - can be removed
# Age - Remove Date of Birth and create a new variable Age by calculating Age from Date of Birth
# 

data_combined = cbind(data_num, data_cat)
data2 = data_combined

age_calc = function(dob,enddate=Sys.Date(),units='months'){
  if(!inherits(dob,"Date") | !inherits(enddate, "Date")){
    stop("Both dob and Date must be in Date classifiers")
  }
  start = as.POSIXlt(dob)
  end = as.POSIXlt(enddate)
  
  years = end$year - start$year
  if(units =='years'){
    result = ifelse((end$mon = start$mon) |
                      ((end$mon == start$mon) & (end$mday = start$mday)),
                      years - 1, years)
  }
  else if(units=='months'){
    months = (years-1) * 12
    result = months + start$mon
  }else if(units == 'days'){
    result = difftime(end, start,units='days')
  }else {
    stop("Unrecongnized units. Please choose years, month or days")
  }
  return(result)
}
data2$Age = age_calc(as.Date(data2$Date.Of.Birth, format="%Y-%m-%d"),enddate=Sys.Date(), units='years')
  
head(data2$Age)
data2 = subset(data2, select=-c(Date.Of.Birth, Board.in.Twelth, School.Board.in.Tenth, State))
#data2 = subset(data2, select=-c(Board.in.Twelth, School.Board.in.Tenth))
#data2 = subset(data2, select=-c(State))
str(data2)

data2 = subset(data2, select=-c(Discipline))

data2 = centralImputation(data2)

#######################################################################################
#Divide the data in to train and test data
library(caret)
set.seed(123)
samples = createDataPartition(data2$Pay_in_INR, times = 1, p=0.7, list=FALSE)
train = data2[samples,]
test = data2[-samples,]
########################################################################################

# Linear Regression

LM_model = lm(Pay_in_INR ~., data=train)
summary(LM_model)

LM_model$fitted.values

### Evaluation on test
regr.eval(test$Pay_in_INR, LM_model$fitted.values)

#Disciplinecontrol and instrumentation engineering
#Disciplineindustrial & management engineering
#GraduationMCA
#
#
##########################
library(car)
vif(LM_model)

alias(LM_model)


##Step AIC
library(MASS)
stepaic = stepAIC(LM_model, direction='both')


LM_model2 = lm(Pay_in_INR ~ Score.in.Tenth + Score.in.Twelth + CollegeTier + 
                 Score.in.English.language + Score.in.Logical.skill + Score.in.Quantitative.ability + 
                 Score.in.Domain + Score.in.ComputerProgramming + Score.in.MechanicalEngg + 
                 Score.in.ElectricalEngg + Score.in.CivilEngg + Score.in.conscientiousness + 
                 Score.in.agreeableness + Score.in.extraversion + Score.in.nueroticism + 
                 Score.in.openess_to_experience + Graduation + 
                 Year.Of.Twelth.Completion + Gender + Age,data=train)

summary(LM_model2)

regr.eval(test$Pay_in_INR, LM_model2$fitted.values)
plot(LM_model2)

###########################################################################################

library(randomForest)
#data_rf = randomForest(Pay_in_INR~., data=train,ntree=100)




#str(data2)
k = 10
data2$id <- sample(1:k, nrow(data2), replace = TRUE)
list = 1:k

library(dplyr)
prediction <- data.frame()
testsetCopy <- data.frame()
library(DMwR)
library(car)
library(MASS)
library(caret)
progress.cv = create_progress_bar("text")
progress.cv$init(k)

#function for k fold
for(i in 1:k){
  # remove rows with id i from dataframe to create training set
  # select rows with id i to create test set
  trainingset <- subset(data2, id %in% list[-i])
  testset <- subset(data2, id %in% c(i))
  
  data_rf = randomForest(Pay_in_INR~., data=trainingset,ntree=100)
  
  data_rf$predicted
  data_rf$importance
  varImpPlot(data_rf)
  predictions = predict(data_rf, newdata = trainingset, type = "response")
  temp = as.data.frame(predict(data_rf, newdata = testset, type="response"))
  prediction = rbind(prediction,temp)
  
  testsetCopy = rbind(testsetCopy, as.data.frame(testset$Pay_in_INR))
  
  
}

results = cbind(prediction, testsetCopy)
regr.eval(testsetCopy$`testset$Pay_in_INR`, prediction)


data_rf$importance

plot(data_rf)


######################################
#to run on test file

testdata = read.csv("test.csv", header=T)

testID = subset(testdata, select=c(Candidate.ID))
testdata = subset(testdata, select=-c(Candidate.ID, CollegeCode, CityCode,CityTier))

testdata_num = subset(testdata, select= -c(Discipline,State,Graduation,Board.in.Twelth,Year.Of.Twelth.Completion, School.Board.in.Tenth, Date.Of.Birth,Gender))
testdata_cat = subset(testdata, select= c(Discipline,State,Graduation,Board.in.Twelth,Year.Of.Twelth.Completion, School.Board.in.Tenth, Date.Of.Birth,Gender))


testdata2 = cbind(testdata_num, testdata_cat)




testdata2 = cbind(testdata_num, testdata_cat)
testdata2 = centralImputation(testdata2)


testdata2$Age = age_calc(as.Date(testdata2$Date.Of.Birth, format="%Y-%m-%d"),enddate=Sys.Date(), units='years')


testdata2 = subset(testdata2, select=-c(Date.Of.Birth, Board.in.Twelth, School.Board.in.Tenth, State))

testdata2$id <- sample(1:k, nrow(testdata2), replace = TRUE)

test_predictions = predict(data_rf, newdata=testdata2, type = "response")

Submissions = cbind(testID, test_predictions)
  
colnames(Submissions) = c("ID", "Salary")

write.csv(Submissions, "predictions.csv", row.names=F)

str(data2)
ggplot(data2, aes(x=Pay_in_INR, fill=Pay_in_INR)) + 
  geom_bar(stat="count", color="blue")+
  scale_fill_gradient(low="lightblue", high="green", guide=FALSE)+
  labs(title="Pay scale in training data", x="Digits")
hist(data2$Pay_in_INR, col="green", main="Training data")
hist(Submissions$Salary, col="blue", main="Predicted Salary scale")

plot(data2$Pay_in_INR)



