---
title: "Project2"
output: pdf_document
date: "2024-06-19"
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)

library(dbt.DataIO)
library(DataVisualizations)
library(ggplot2)
library(reshape2)

iris_data <- ReadLRN("abc.lrn")

crabs_data <- ReadLRN("1974Crabs.lrn")
crabs_sex <- ReadCLS("1974Crabs_sex.cls")
crabs_sp <- ReadCLS("1974Crabs_sp.cls")

# Robust Normalization
iris_pp <- RobustNormalization(iris_data$Data)
crabs_pp <- RobustNormalization(crabs_data$Data[,2:6]) 
index <- 1:length(crabs_data$Data[,1])
crabs_pp <- cbind(index, crabs_pp)

#Signed Log
iris_sl <- SignedLog(iris_data$Data)
crabs_sl <- SignedLog(crabs_data$Data)

# z-transformation
iris_z <- scale(iris_data$Data)
crabs_z <- scale(crabs_data$Data)
```

# Clustering

Each group is assigned a different dataset to work with. The dataset will be a supervised learning dataset, implying a specific classification. The goal is, to find a clustering (unsupervised learning) which resembles the given classification. Finding a clustering could imply meaningful structures have a relationship with the classification. Follow the steps to obtain the clustering and evaluate in bullet points, why you find or might not find a match between clustering and classification.

1.  Preprocess the data if necessary

to preprocess we applied the RobustNormalization method from the package DataVisualisations on the data of both data sets below a summary of the data before and after preprocessing.

### Iris summary before

```{r, echo = F, eval = T}
summary(iris_data$Data)
```

### Iris summary after

```{r, echo = F, eval = T}
summary(iris_pp)
```

```{r, echo = F, eval = T}
library(dplyr)

df <- as.data.frame(crabs_pp)

# Schmelze das Data Frame für ggplot2
library(reshape2)
df_melted <- melt(df, variable.name = "Column")

# Erstelle eine Funktion zur Berechnung der Dichte
calc_density <- function(x) {
  d <- density(x)
  data.frame(x = d$x, y = d$y)
}

# Berechne die Dichte für jede Spalte
densities <- df_melted %>%
  group_by(Column) %>%
  do(calc_density(.$value))

# Erstelle den Plot
ggplot(densities, aes(x = x, y = y, color = Column)) +
  geom_line() +
  scale_x_continuous(limits = c(-2,2))
```

### Crabs summary before

```{r, echo = F, eval = T}
summary(crabs_data$Data)
```

### Crabs summary after

```{r, echo = F, eval = T}
summary(crabs_pp)
```

2.  Clustering with ESOM

## Clustering of the Iris Data set

```{r, echo = F, eval = T}
library(Umatrix)

iris_esom <- esomTrain(iris_sl, Key = 1:nrow(iris_sl))

plotMatrix(iris_esom$Umatrix, iris_esom$BestMatches, iris_solution$Cls)


```

```{r, echo = F, eval = T}

iris_pmatrix <- pmatrixForEsom(iris_sl, iris_esom$Weights, iris_esom$Lines, iris_esom$Columns, iris_esom$Toroid)

plotMatrix(iris_pmatrix, ColorStyle = "Pmatrix")

```

```{r, echo = F, eval = T}

iris_ustar <- CalcUstarmatrix(iris_esom$Umatrix, iris_pmatrix)

plotMatrix(iris_ustar, iris_esom$BestMatches, iris_solution$Cls)

```

## Clustering of the Crabs Data set with Sex highlighted

```{r, echo = F, eval = T}

crabs_esom <- esomTrain(crabs_pp, Key = 1:nrow(crabs_sl))

plotMatrix(crabs_esom$Umatrix, crabs_esom$BestMatches, crabs_sex$Cls)

```

## Clustering of the Crabs Data set with SP highlighted

```{r, echo = F, eval = T}
plotMatrix(crabs_esom$Umatrix, crabs_esom$BestMatches, crabs_sp$Cls)

```

```{r, echo = F, eval = T}

crabs_pmatrix <- pmatrixForEsom(crabs_pp, crabs_esom$Weights, crabs_esom$Lines, crabs_esom$Columns, crabs_esom$Toroid)

plotMatrix(crabs_pmatrix, ColorStyle = "Pmatrix")

```

```{r, echo = F, eval = T}

crabs_ustar <- CalcUstarmatrix(crabs_esom$Umatrix, crabs_pmatrix)

plotMatrix(crabs_ustar, crabs_esom$BestMatches, crabs_sp$Cls)

```

3.  Clustering with DBS

```{r, echo = F, eval = T}
library(DatabionicSwarm)

```

4.  Evaluate match between classification and clustering
