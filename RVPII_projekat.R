setwd("C:\\Users\\pekit\\Desktop\\Master - 1. semestar\\RVPII\\projekat\\podaci")

library(sparklyr)
library(dplyr)
library(ggplot2)


sc <- spark_connect(master = "local", version="2.4.3")
spark_get_java()


deaths.basic <- spark_read_csv(sc, 
                               name = "deaths", 
                               path = ".", 
                               header = T, 
                               memory = T)


deaths <- deaths.basic %>% 
  mutate(
    Population = as.double(Population)
  ) %>%
  filter(!is.na(Year) & 
    !is.na(Race__Ethnicity_Code) &
      !is.na(Race__Ethnicity_Name) &
      !is.na(Gender_Code) & 
      !is.na(Gender_Name) & 
      !is.na(Age_Group_Code) & 
      !is.na(Age_Group_Name) & 
      !is.na(Cause_of_Death_Code) & 
      !is.na(Cause_of_Death_Name) &
      !is.na(County_FIPS_Code) & 
      !is.na(County_Name) & 
      !is.na(Population) &
      !is.na(Number_of_Deaths) &
      Race__Ethnicity_Code != "T" &
      Age_Group_Code != 0)

population <- deaths %>% 
  select(Population) %>%
  collect()

num.deaths <- deaths %>%
  select(Number_of_Deaths) %>%
  collect()

population.quantiles <- quantile(population$Population)
population.low <- population.quantiles[[2]]
population.high <- population.quantiles[[4]]


num.deaths.quantiles <- quantile(
  num.deaths$Number_of_Deaths,
  probs = c(99.0, 99.1, 99.2, 99.3, 99.4, 99.5, 99.6, 99.7, 99.8, 99.9, 100)/100)
num.deaths.low <- num.deaths.quantiles[[1]]
num.deaths.high <- num.deaths.quantiles[[5]]


deaths <- deaths %>% 
  mutate(Age_Group_Descriptive = case_when(
           Age_Group_Code < 9 ~ "Young",
           TRUE ~ "Old"),
         Population_Descriptive = case_when(
           Population < population.low ~ "low",
           Population < population.high ~ "mid",
           TRUE ~ "high"),
         Number_of_Deaths_Descriptive = case_when(
           Number_of_Deaths < num.deaths.low ~ "low",
           Number_of_Deaths < num.deaths.high ~ "mid",
           TRUE ~ "high")
        )


library(DBI)
dbGetQuery(sc, "select count * from deaths")

deaths.split <- sdf_random_split(deaths, training = 0.75, test = 0.25, seed = 5)

deaths.training <- deaths.split$training
deaths.test <- deaths.split$test

iters <- c(3, 5, 10, 15, 20, 30, 50, 70)
model.accuracies <- c(length(iters))
counter <- 0

for (iter in iters) {
  counter <- counter + 1
  log.reg <- ml_logistic_regression(
    x = deaths.training, 
    formula = Age_Group_Descriptive ~ 
      Gender_Code + 
      Race__Ethnicity_Code +
      Population_Descriptive +
      Cause_of_Death_Code +
      County_Name, 
    family = "binomial", max_iter = iter, threshold = 0.5)
  
  log.reg.perfs <- ml_evaluate(log.reg, deaths.test)
  model.accuracies[counter] <- log.reg.perfs$accuracy()
}


##### Prikaz preciznosti na grafiku

df <- data.frame(max_iterations = iters, accuracy = model.accuracies)
df

g <- ggplot(df, aes(max_iterations, accuracy)) +
  geom_line() +
  geom_point()

g


##### K-tostruka validacija
k <- 4
model.formula <- Age_Group_Descriptive ~ 
  Gender_Code + 
  Race__Ethnicity_Code +
  Population_Descriptive +
  Cause_of_Death_Code +
  County_Name

accuracies.dec.tree <- c()
accuracies.log.reg <- c()
accuracies.svm <- c()

partition_sizes <- rep(1/k, times = k)
names(partition_sizes) <- paste0("fold", as.character(1:k))

deaths.partitions <- deaths %>%
  sdf_random_split(weights = partition_sizes, seed = 86)

for (i in 1:k) {
  
  training <- sdf_bind_rows(deaths.partitions[-i])
  
  dec.tree <- ml_decision_tree_classifier(
    x = training, 
    formula = model.formula,
    max_depth = 5, 
    min_instances_per_node = 1000, 
    impurity = "gini")
  
  log.reg <- ml_logistic_regression(
    x = training, 
    formula = model.formula, 
    family = "binomial", 
    max_iter = 20, 
    threshold = 0.5)
  
  svm <- ml_linear_svc(
    x = training,
    formula = model.formula,
    max_iter = 20, 
    standardization = T)
  
  evaluate.dec.tree <- ml_evaluate(dec.tree, deaths.partitions[[i]])
  accuracies.dec.tree[i] <- evaluate.dec.tree$Accuracy
  
  evaluate.log.reg <- ml_evaluate(log.reg, deaths.partitions[[i]])  
  accuracies.log.reg[i] <- evaluate.log.reg$accuracy()
  
  evaluate.svm <- ml_evaluate(svm, deaths.partitions[[i]])
  accuracies.svm[i] <- evaluate.svm$Accuracy
  
}

print(mean(accuracies.dec.tree))
print(mean(accuracies.log.reg))
print(mean(accuracies.svm))



#### Klasterizacija 1

deaths.for.clusters <- deaths %>% 
  filter(Number_of_Deaths > 0 & 
           Number_of_Deaths < 5000)

deaths.per.age <- ml_kmeans(
  deaths.for.clusters,
  ~ Age_Group_Code + Number_of_Deaths, 
  k = 3, 
  max_iter = 10, 
  init_mode = "k-means||")

deaths.per.age
deaths.per.age$model$summary$cluster()

ml_evaluate(deaths.per.age, deaths %>% select(Age_Group_Code, Number_of_Deaths))

cluster.values <- deaths.per.age$model$summary$cluster() %>% collect()
deaths.for.clusters <- deaths.for.clusters %>% collect()
deaths.for.clusters$clust <- as.factor(cluster.values$prediction)

cluster.centers.df <- data.frame(
  age = deaths.per.age$centers$Age_Group_Code,
  num_deaths = deaths.per.age$centers$Number_of_Deaths
)

ggplot(data = deaths.for.clusters,
       aes(x = Age_Group_Code, y = Number_of_Deaths, colour = clust)) +
  geom_jitter(size = 2) +
  geom_point(data = cluster.centers.df,
           aes(x = age, y = num_deaths),
           color = "brown",
           size = 4,
           shape = 15)


#### Klasterizacija 2
deaths.for.clusters2 <- deaths %>% 
  filter(Number_of_Deaths > 0 & 
           Number_of_Deaths < 5000) %>%
  group_by(Race__Ethnicity_Code,
           Gender_Code,
           Cause_of_Death_Code,
           County_Name,
           Age_Group_Code) %>%
  summarise(Deaths_Sum=sum(Number_of_Deaths), .groups = "keep")

deaths.per.age2 <- deaths.for.clusters2 %>% 
  ml_kmeans(~ Age_Group_Code + Deaths_Sum, 
            k = 3, 
            max_iter = 10, 
            init_mode = "k-means||")

cluster.values2 <- deaths.per.age2$model$summary$cluster() %>% collect()
deaths.for.clusters2 <- deaths.for.clusters2 %>% collect()
deaths.for.clusters2$clust <- as.factor(cluster.values2$prediction)

cluster.centers.df2 <- data.frame(
  age = deaths.per.age2$centers$Age_Group_Code,
  num_deaths = deaths.per.age2$centers$Deaths_Sum
)
  

ggplot(data = deaths.for.clusters2,
       aes(x = Age_Group_Code, y = Deaths_Sum, colour = clust)) +
  geom_jitter(size = 2) +
  geom_point(data = cluster.centers.df2,
             aes(x = age, y = num_deaths),
             color = "brown",
             size = 4,
             shape = 15)
