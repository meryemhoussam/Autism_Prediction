library(Amelia)
library(dplyr)
library(GGally)
library(ggplot2)
TRAIN=read.csv('Desktop/Rproject/train.txt',na.strings='?')
TEST=read.csv('Desktop/Rproject/test.txt',na.strings='?')
#visualisation des données imbalancées
View(TRAIN)
summary(TRAIN)
str(TRAIN)
dim(TRAIN)
colSums(is.na(TRAIN))
colnames(TRAIN)
Contries=TRAIN$contry_of_res
summary(Contries)
pie(x=table(Contries),main = "individus contries")
head(TRAIN)
glimpse(TRAIN)
table(TRAIN$gender,TRAIN$Class.ASD)
table(TRAIN$austim,TRAIN$Class.ASD)
prop.table(table(TRAIN$gender,TRAIN$Class.ASD),margin = 1)
prop.table(table(TRAIN$austim,TRAIN$Class.ASD),margin = 1)
VISUALIZED_TRAIN=select(TRAIN,age,gender,austim,result,Class.ASD)
TRAIN_SELECTED_NUMcol=select(TRAIN,age,result,Class.ASD)
ggcorr(TRAIN_SELECTED_NUMcol,nbreaks=6,label=TRUE,label_size=4,color='grey50')
VISUALIZED_TRAIN$Class.ASD=factor(VISUALIZED_TRAIN$Class.ASD)
str(VISUALIZED_TRAIN)
ggplot(VISUALIZED_TRAIN,aes(x=Class.ASD))+
  geom_bar(width=0.5,fill='coral')+
  geom_text(stat='count',
            aes(label=stat(count)),
            vjust=-0.5)+
  theme_classic()
#sick count by gender
ggplot(VISUALIZED_TRAIN,aes(x=Class.ASD,fill=gender))+
  geom_bar(position = position_dodge())+
  geom_text(stat='count',
            aes(label=stat(count)),
            position = position_dodge(width=1),
            vjust=-0.5)+
  theme_classic()
#sick count by austim
ggplot(VISUALIZED_TRAIN,aes(x=Class.ASD,fill=austim))+
  geom_bar(position = position_dodge())+
  geom_text(stat='count',
            aes(label=stat(count)),
            position = position_dodge(width=1),
            vjust=-0.5)+
  theme_classic()
#sick count by age
# Discretize age to plot autism
boxplot(VISUALIZED_TRAIN$age)
summary(VISUALIZED_TRAIN$age)
VISUALIZED_TRAIN$Discretized.age = cut(VISUALIZED_TRAIN$age, c(0,10,20,30,40,50,60,70,80))
View(VISUALIZED_TRAIN)
ggplot(VISUALIZED_TRAIN,aes(x=Class.ASD,fill=VISUALIZED_TRAIN$Discretized.age))+
  geom_bar(position = position_dodge())+
  geom_text(stat='count',
            aes(label=stat(count)),
            position = position_dodge(width=1),
            vjust=-0.5)+
  theme_classic()
VISUALIZED_TRAIN$Discretized.age=NULL
#Représentation de distribution des observations selon les 2 classes
table(TRAIN$Class.ASD) 
prop.table(table(TRAIN$Class.ASD))
barplot(prop.table(table(TRAIN$Class.ASD)), 
        col = rainbow(2),
        ylim = c(0,1),
        main = "Distribution des classes ")
# on constate que les deux colonnes : relation et ethnicity ont des valeurs NA
missmap(TRAIN, col=c("red" , "grey"))
#on va supprimer les colonnes A1-A10SCORE , ETHNICITY, CONTRY_OF_RES, RELATION, AGE_DESCR  
TRAIN_SELECTED=select(TRAIN,-1:-11,-14,-17,-20,-21)
View(TRAIN_SELECTED)
missmap(TRAIN_SELECTED, col=c("red" , "grey"))
#on va numeriser les données pour calculer leurs importances sur les prédictions
values= 2:5 
TRAIN_SELECTED[,values]<- lapply((TRAIN_SELECTED[,values]),factor)
levels(TRAIN_SELECTED$gender)=c(1,0)
levels(TRAIN_SELECTED$jaundice)=c(0,1)
levels(TRAIN_SELECTED$austim)=c(0,1)
levels(TRAIN_SELECTED$used_app_before)=c(0,1)
TRAIN_SELECTED
str(TRAIN_SELECTED)
#On va calculer l'importance des variables sur les prédictions en utilisant ROC_CURVE AREA as score
library(ggcorrplot)
library(caret)
library(randomForest)
library(gam)
roc_imp <- filterVarImp(x = TRAIN_SELECTED[,1:6], y = TRAIN_SELECTED$Class.ASD)
roc_imp
#ordroner les scores d'une façon décroissante
roc_imp <- data.frame(cbind(variable = rownames(roc_imp), score = roc_imp[,1]))
roc_imp
roc_imp$score <- as.double(roc_imp$score)
roc_imp
roc_imp[order(roc_imp$score,decreasing = TRUE),]
#afficher la figure qui illustre l'importance des variables sur les prédictions obtenue à partir de ROC_CURVE 
ggplot(roc_imp, aes(x=reorder(variable, score), y=score)) +
  geom_point() +
  geom_segment(aes(x=variable,xend=variable,y=0,yend=score)) +
  ylab("IncNodePurity") +
  xlab("NOM DES VARIABLES") +
  coord_flip()
#On va calculer l'importance des variables sur les prédictions en utilisant RANDOM FOREST
rf = randomForest(x= TRAIN_SELECTED[,1:6],y= TRAIN_SELECTED[,7])
var_imp <- varImp(rf, scale = FALSE)
#ordroner les scores d'une façon décroissante
var_imp_df <- data.frame(cbind(variable = rownames(var_imp), score = var_imp[,1]))
var_imp_df$score <- as.double(var_imp_df$score)
var_imp_df[order(var_imp_df$score,decreasing = TRUE),]
#afficher la figure qui illustre l'importance des variables sur les prédictions obtenue à partir de RF
ggplot(var_imp_df, aes(x=reorder(variable, score), y=score)) +
  geom_point() +
  geom_segment(aes(x=variable,xend=variable,y=0,yend=score)) +
  ylab("IncNodePurity") +
  xlab("NOM DES VARIABLES") +
  coord_flip()
#on compte utiliser l'importance de la première méthode ROC car elle nous semble plus logique
df <- select(TRAIN, result, austim, jaundice, age, gender,Class.ASD)
df
str(df)
table(df$Class.ASD)
prop.table(table(df$Class.ASD))
#on va factoriser les classes ASD
df$Class.ASD=factor(df$Class.ASD)
#on va créer des datasets d'entrainement et de validation
set.seed(100)
validation_index <- createDataPartition(df$Class.ASD, p=0.80, list=FALSE)
# selectionner 20% des données pour la validation
validation <- df[-validation_index,]
dim(validation)
# utiliser 80% des données restantes pour l'entrainement
dataset <- df[validation_index,]
dim(dataset)
#On va utiliser la Regression logistique sur les données non balancées
set.seed(100)

logit_model <- glm(data = dataset ,
                   formula = Class.ASD~. ,
                   family = "binomial" )
logit_pred <- predict(object = logit_model,
                      newdata = validation ,
                      type = "response" )
summary(logit_pred)
#la matrice de confusion
cm = confusionMatrix(data = as.factor(as.integer(logit_pred>0.3)) ,
                     reference =validation$Class.ASD,
                     positive = "1")
cm
cm_t = table(validation$Class.ASD, logit_pred>0.3)
cm_t
######## visualisation des metrics : GMEAN and F1_SCORE, car en cas d'imbalances, on les prend comme mesure
cm$byClass
cm$overall['Accuracy']
precision <- cm$byClass['Pos Pred Value']  
precision
recall <- cm$byClass['Sensitivity']
recall
f_measure <- 2 * ((precision * recall) / (precision + recall))
f_measure
gmean = sqrt((cm_t[2,2]/(cm_t[2,2]+cm_t[1,2])) * (cm_t[1,1]/(cm_t[1,1]+cm_t[2,1])))
gmean
############# on va balancer les données pour comparer les metrics
require(ROCR) # for the ROC curve 
require(ROSE) # for downsampling
table(dataset$Class.ASD)

data_bal <- ovun.sample(Class.ASD~ ., data =  dataset, method = "both", 
                                  N=475, seed = 1)$data
table(data_bal$Class.ASD)
logit_model2 <- glm(data = dataset ,
                   formula = Class.ASD~. ,
                   family = "binomial" )
logit_pred2 <- predict(object = logit_model2,
                      newdata = validation ,
                      type = "response" )
summary(logit_pred2)
cm_2 <- table(validation$Class.ASD, logit_pred2 >=0.3)
cm_2
cm2 = confusionMatrix(data = as.factor(as.integer(logit_pred2>0.3)) ,
                     reference =validation$Class.ASD,
                     positive = "1")
cm2
cm2$byClass
cm2$overall['Accuracy']
precision <- cm2$byClass['Pos Pred Value']  
precision
recall <- cm2$byClass['Sensitivity']
recall
f_measure <- 2 * ((precision * recall) / (precision + recall))
f_measure
gmean = sqrt((cm_2[2,2]/(cm_2[2,2]+cm_2[1,2])) * (cm_2[1,1]/(cm_2[1,1]+cm_2[2,1])))
gmean
################## Après avoir balancer les données on a constaté que les metrics restent 
#################les mêmes que pour les données imbalancées
########## maintenant on va essayer un nouveau model: SVM
control <- trainControl(method="repeatedcv", number=10, repeats=3)
svm <- train(Class.ASD ~., data = dataset, method = "svmLinear", 
              trControl = control,  preProcess = c("center","scale"), 
              tuneGrid = expand.grid(C = seq(0, 2, length = 20)))
svm
plot(svm)
svm$bestTune
svm_pred <- predict(svm, validation)
table(validation$Class.ASD, svm_pred)

cm_3 <- confusionMatrix(data = as.factor(svm_pred) ,
                        reference =validation$Class.ASD,
                        positive = "1")
######### pour visualiser les metrics du nouveau modèle
cm_3$overall['Accuracy']
cm_3$byClass
###### on a constaté que Accuracy a augmenté
#################### maintenant on va utiliser BAGGING
library(plyr)
library(readr)
library(caretEnsemble)

set.seed(100)

bagCART_model <- train(Class.ASD~., data= dataset, method="treebag", 
                       metric="Accuracy", trControl=control)

#Predictions
predictTest_Bagging = predict(bagCART_model, newdata = validation)

# Confusion matrix 
table(validation$Class.ASD, predictTest_Bagging)
cm_Bagging = confusionMatrix(data = as.factor(predictTest_Bagging) ,
                             reference =  validation$Class.ASD,
                             positive = "1")
cm_Bagging$overall["Accuracy"]

#################### maintenant on va utiliser BOOSTING
set.seed(100)

gbm_model <- train(Class.ASD ~., data=dataset, method="gbm", 
                   metric="Accuracy", trControl= control)

predictTest_Boosting = predict(gbm_model, newdata = validation)

table(validation$Class.ASD, predictTest_Boosting)

cm_Boosting = confusionMatrix(data = as.factor(predictTest_Boosting) ,
                              reference = validation$Class.ASD,
                              positive = "1")
cm_Boosting$overall["Accuracy"]

#################### maintenant on va utiliser STACKING
set.seed(100)

levels(dataset$Class.ASD) <- c("yes", "no")


control_stacking <- trainControl(method="repeatedcv", number=5, repeats=2, 
                                 classProbs=TRUE)

algorithms_to_use <- c('rpart', 'lda', 'knn', 'svmLinear')

stacked_models <- caretList(Class.ASD ~., data=dataset, trControl=control_stacking, 
                            methodList=algorithms_to_use)

stacking_results <- resamples(stacked_models)

summary(stacking_results)

stackControl <- trainControl(method="repeatedcv", number=5, repeats=3, savePredictions=TRUE, 
                             classProbs=TRUE)

glm_stack <- caretStack(stacked_models, method="glm", metric="Accuracy", trControl=stackControl)

print(glm_stack)
levels(validation$Class.ASD) <- c("yes", "no")
predictTest_Stacking = predict(glm_stack, newdata = validation)
table(validation$Class.ASD, predictTest_Stacking)

cm_Stacking = confusionMatrix(data = as.factor(predictTest_Stacking) ,
                              reference = validation$Class.ASD,
                              positive = "yes")
cm_Stacking
cm_Stacking$overall["Accuracy"]
############### on déduit que le modèle STACKING est le meilleur vu sa haute performance : 0.80
########## Tester le Stacking : on va commencer par nettoyer la dataset du test 
View(TEST)
missmap(TEST, col=c("red" , "grey"))
#on va supprimer les observations contenant des valeurs NA
TEST= na.omit(TEST)
#MISSINGNESS MAP change après la supression des observations NA
missmap(TEST, col=c("red" , "grey"))
#on va supprimer les colonnes A1-A10SCORE , ETHNICITY, CONTRY_OF_RES, RELATION, AGE_DESCR  
TEST_SELECTED=select(TEST,-1:-11,-14,-17,-20,-21)
View(TEST_SELECTED)
new_prediction <- predict(glm_stack, TEST)
submit <- data.frame(PatientID = TEST$ID, AUTISM = new_prediction)
write.csv(submit, file = "Desktop/Rproject/newTestResult.csv", row.names = FALSE)

View(submit)
