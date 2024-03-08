###############################################################################
# Script: Movie_Cluster.R
# Copyright (c) 2018 by Alan Montgomery. Distributed using license CC BY-NC 4.0
# To view this license see https://creativecommons.org/licenses/by-nc/4.0/
#
# R script for for analyzing movie similarities
#
# Requires the following files:
#   Movie_Data.RData         This is an R dataset, must use load
#   Movie_Data_LDA10.RData   precomputed LDA cluster solution
#
# The data included for this exercise is for internal use only and
# may not be posted or distributed further.
# Specifically the files opus_movies.txt and opus_keywords.txt
# is data that is provided by The Numbers (http://www.the-numbers.com),
# powered by OpusData (http://www.opusdata.com).
# The opus_movielens_tags.txt is available from Movielens
# which is located at http://grouplens.org/datasets/movielens/latest
###############################################################################



###############################################################################
### @setup the environment by loading packages and data
###############################################################################

# load in necessary packages
if (!require(topicmodels)) {install.packages("topicmodels"); library(topicmodels)}
if (!require(tm)) {install.packages("tm"); library(tm)}
if (!require(wordcloud)) {install.packages("wordcloud"); library(wordcloud)}
if (!require(proxy)) {install.packages("proxy"); library(proxy)}
if (!require(lattice)) {install.packages("lattice"); library(lattice)}
if (!require(LDAvis)) {install.packages("LDAvis"); library(LDAvis)}
if (!require(ggplot2)) {install.packages("ggplot2"); library(ggplot2)}
if (!require(slam)) {install.packages("slam"); library(slam)}           # sparse matrices
if (!require(plyr)) {install.packages("plyr"); library(plyr)}           # manipulating data
if (!require(NLP)) {install.packages("NLP"); library(NLP)}              # language processing

# function we will use for visualizing LDA cluster result
# see https://gist.github.com/trinker/477d7ae65ff6ca73cace
topicmodels2LDAvis <- function(x, ...){
  post <- topicmodels::posterior(x)
  if (ncol(post[["topics"]]) < 3) stop("The model must contain > 2 topics")
  mat <- x@wordassignments
  LDAvis::createJSON(
    phi = post[["terms"]], 
    theta = post[["topics"]],
    vocab = colnames(post[["terms"]]),
    doc.length = slam::row_sums(mat, na.rm = TRUE),
    term.frequency = slam::col_sums(mat, na.rm = TRUE)
  )
}

# set to working directory of script (assumes data in same directory as script)
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))  # only works in Rstudio scripts
# alternatively set the working directory manually
#setwd("~/Documents/class/marketing analytics/class/movielda") #!! set to your directory



###############################################################################
### @input data
###############################################################################

# load in the data from a special RData set format (this is not a text data)
load("Movie_Data.RData")

# read in movie datasets
movies=read.delim("opus_movies.txt",header=T)  # the Opus movie data
tags=read.delim("opus_movielens_tags.txt",header=T)  # just the tags from movielens

## make modifications to the dataset

# create a short version of the title
movies$short_name=strtrim(enc2native(as.character(movies$display_name)),20)
# change data formats
movies$release_date=as.Date(as.character(movies$release_date),format="%Y-%m-%d")
movies$release_month=format(movies$release_date,"%m")
movies$release_monthyear=format(movies$release_date,"%m-%Y")
tags$odid=as.factor(tags$odid)
# map the months to seasons
movies$release_season=rep('1Winter',length(movies$release_month))
movies$release_season[movies$release_month %in% c('03','04')]='2Spring'
movies$release_season[movies$release_month %in% c('05','06','07')]='3Summer'
movies$release_season[movies$release_month %in% c('08','09','10')]='4Fall'
movies$release_season[movies$release_month %in% c('11','12')]='5Holiday'
# remove punctuation from genre and rating
movies$rating=revalue(movies$rating,c("PG-13"="PG13"))
movies$genre=revalue(movies$genre,c("Black Comedy"="BlackComedy","Concert/Performance"="Performance","Romantic Comedy"="RomanticComedy","Thriller/Suspense"="Thriller"))
# create a matrix with genre and rating as 
dummygenre=model.matrix(~genre,movies)[,-1]  # omit the intercept in the first column
dummyrating=model.matrix(~rating,movies)[,-1]  # omit the intercept in the first column
# since these are matrix, we coerce them to lists, merge them, and then overwrite movies
movies=cbind(movies,as.data.frame(cbind(dummygenre,dummyrating)))
valgenre=colnames(dummygenre)
valrating=colnames(dummyrating)

# create a standardized version of the data
nvariables=sapply(movies,is.numeric)
nvariables=names(nvariables[nvariables])
smovies=scale(movies[,nvariables])


## transform the terms into a structure that can be used for topic modeling

# use this definition of mterms for movielens tags
# put data in sparse matrix form using simple_triplet_matrix as needed by LDA
mterms=simple_triplet_matrix(i=as.integer(tags$odid),j=as.integer(tags$tag),v=tags$count,
                             dimnames=list(levels(tags$odid),levels(tags$tag)))
# let's only keep words that are used frequently (by at least 20 movies)
mterms=mterms[,apply(mterms,2,sum)>=20]
# also delete any movies that do not have any terms
mterms=mterms[apply(mterms,1,sum)>0,]

# determine dimensions of mterms
umovies=movies[movies$odid %in% as.integer(rownames(mterms)),]   # create a subset of the movies that have terms
lmterms=apply(mterms,1,sum)   # compute the sum of each of the rows (# of terms per movie)
lwterms=apply(mterms,2,sum)   # compute the sum of each of the columns (# of times word used)

# also create another version as DocumentTermMatrix
tmterms = as.DocumentTermMatrix(mterms,weight=weightTfIdf)

# create a vector with the names of the most frequent terms
topterms = findFreqTerms(tmterms,20)
idxtopterms = (1:ncol(mterms))[colnames(mterms) %in% topterms]  # get the indices of the topterms

# create a matrix with just the top keywords (cast this as a dense matrix)
movieterms = as.matrix(mterms[,topterms])



###############################################################################
### @describe the data and visualize it
###############################################################################

# let's look at our data
ls()             # to speed up the analysis results have been precomputed
str(moviedata)   # dataset with 1153 movies
str(movietags)   # dataset with 962 user generated tags that describe the movie

# movietags is a special text mining object (it is a sparse matrix)
movietags     # this is an 1153 X 962 sparse matrix or DocumentTermMatrix; 1,094,139 of the elements are 0, only 15,047 are not zero
inspect(movietags[1,])  # this is the first row (elements that are zero do not print)
inspect(movietags[,1])  # this is the first column
inspect(movietags["Titanic",])  # alternatively we can look at a specific movie
inspect(movietags[c("Titanic","Jurassic Park","Finding Nemo"),])
inspect(movietags[,"action"])

# find frequent terms
findFreqTerms(movietags,lowfreq=500)

# find tags associated with "action"
findAssocs(movietags, "action", 0.3)

# turn the TermDocumentMatrix into a regular matrix that is expected by most procedures like kmeans (warning: not a good idea for large datasets)
mat_tdm=t(as.matrix(movietags))  # rows are terms, columns are movies
mat_dtm=as.matrix(movietags)  # rows are movies, columns are terms

# get word counts in decreasing order
tag_freqs=sort(rowSums(mat_tdm),decreasing=TRUE)
toptags=names(tag_freqs[1:10])

# create a data frame with words and their frequencies
dm=data.frame(word=names(tag_freqs),freq=tag_freqs)
# lets plot the wordcloud with just the top 50 words
wordcloud(dm[1:50,]$word,dm$freq,random.order=FALSE,colors=brewer.pal(8,"Dark2"))

# let plot the wordcloud for all movies in specific genre
select_movies=moviedata$short_name[moviedata$genre=="Action"]
movie_tag_freqs=sort(rowSums(mat_tdm[,select_movies]),decreasing=TRUE)
# create a data frame with words and their frequencies
dm2=data.frame(word=names(movie_tag_freqs),freq=movie_tag_freqs)
# lets plot the wordcloud with just the top 50 words
wordcloud(dm2[1:50,]$word,dm2$freq,random.order=FALSE,colors=brewer.pal(8,"Dark2"))



###############################################################################
### @kmeans-characteristics :: clustering of movies based upon characteristics
###############################################################################

# make a list of variables to include in a kmeans solution
qlist=c("production_budget","sequel",valgenre,valrating)

# compute a k=9 solution
set.seed(569123)   # initialize the random number generator so we all get the same results
grpA=kmeans(smovies[,qlist],centers=9)
valclusters=1:9

# plot the solutions against the production value and genreAction
# since the data is categorical most of the plots will overlay one another,
# so instead we jitter the points -- which adds a small random number to each
par(mfrow=c(1,1),mar=c(5,4,4,1)+.1)
plot(movies$production_budget, jitter(movies$genreAction),col=grpA$cluster)
points(grpA$centers[,c("production_budget","genreAction")],col=valclusters,pch=8,cex=2)
legend("topright",pch=8,bty="n",col=valclusters,as.character(valclusters))

# compare the cluster solutions with the Release Season
CrossTable(movies$release_season,grpA$cluster)   # slightly nicer cross tabulation

# summarize the centroids
grpAcenter=t(grpA$centers)
rownames(grpAcenter)=strtrim(colnames(movies[,qlist]),40)
print(grpAcenter[qlist,])
parallelplot(t(grpAcenter[qlist,]),
             main="Movie Clusters based upon Budget, Genre and Rating",
             auto.key=list(text=as.character(valclusters),space="bottom",columns=3,lines=T))

# print a table with the movies assigned to each cluster
for (i in valclusters) {
  print(paste("* * * Movies in Cluster #",i," * * *"))
  print(movies$display_name[grpA$cluster==i])
}



###############################################################################
### @hierarchical clustering of movies based upon keywords
###############################################################################

# create a matrix from the terms for clustering
mtxterms=as.matrix(mterms)  # this is a dense matrix
# normalize terms as % of times used in movie to make movies more comparable
for (i in 1:nrow(mtxterms)) {
  mtxterms[i,]=mtxterms[i,]/sum(mtxterms[i,])
}
rownames(mtxterms)=umovies$short_name

## let's determine how many clusters to use by computing many cluster solutions
grpC=hclust(dist(mtxterms,method='cosine'),method='ward')
plot(grpC,hang=-1,cex=.5)

# it is too hard to read so let's plot small clusters
grpCd=as.dendrogram(grpC)
grpCcut=cut(grpCd,h=5)
ntree=length(grpCcut$lower)  # how many sub-trees do we have
plot(grpCcut$upper,main="Upper tree")
for (i in 1:ntree) {
  plot(grpCcut$lower[[i]],cex=.4,main=paste("Subtree",i))
}
grpCcluster = cutree(grpC,ntree)   # give the assignments based upon 'ntree' cuts

# plot the solutions against the production value and genreAction
# since the data is categorical most of the plots will overlay one another,
# so instead we jitter the points -- which adds a small random number to each
par(mfrow=c(1,1),mar=c(5,4,4,1)+.1)
plot(umovies$production_budget, jitter(umovies$genreAction),col=grpCcluster)
legend("topright",legend=as.character(1:ntree),col=1:ntree,pch=8)

# compare the cluster solutions with the Release Season
CrossTable(umovies$release_season,grpCcluster)

# print a table with the movies assigned to each cluster
for (i in 1:ntree) {
  print(paste("* * * Movies in Cluster #",i," * * *"))
  print(umovies$display_name[grpCcluster==i])
}

# let's focus on scheduling the Maze Runner in 2014
imovie=(1:nrow(umovies))[umovies$display_name=="The Maze Runner"]
umovies[imovie,]  # look at our target movie: The Maze Runner
grpCcluster[imovie]  # check target movie cluster
i2014=umovies$release_year==2014  # create index of the movies released in 2014
xtabs(~umovies$release_month[i2014])
xtabs(~umovies$release_month[i2014]+grpCcluster[i2014])



###############################################################################
### @kmeans-document-terms
###############################################################################

# create a new matrix that standardizes the rows (e.g., every row sums to one)
smat_dtm=t(apply(mat_dtm,1,function(x)(x/sum(x))))  # use this to cluster movies
smat_tdm=t(apply(mat_tdm,1,function(x)(x/sum(x))))  # use this to cluster tags
#smat_dtm=scale(mat_dtm)  # use this to standardize mean to zero and sd to 1
#smat_tdm=scale(mat_tdm)  # use this to standardize mean to zero and sd to 1

# to illustrate the data print out data for a few movies and terms
smat_dtm[c("Titanic","Iron Man","Total Recall","The Maze Runner"),
         c("action","sci-fi","predictable","funny")]

# compute multiple cluster solutions
kclust=c(2:20,50,100)              # create a vector of k values to try
nclust=length(kclust)              # number of kmeans solutions to compute
bss=wss=rep(0,nclust)              # initialize vectors bss and wss to zeroes
set.seed(34612)                    # set the seed so we can repeat the results
grpQ=as.list(rep(NULL,nclust))     # create empty list to save results
# compute SS for each cluster
for (i in 1:nclust) {
  grpQ[[i]]=kmeans(smat_dtm,kclust[i],nstart=5)  # compute kmeans solution
  wss[i]=grpQ[[i]]$tot.withinss        # save the within SS
  bss[i]=grpQ[[i]]$betweenss           # save the between SS
}

# plot the results and look for the "Hockey-Stick" effect
par(mfrow=c(1,1))
plot(kclust,wss,type="l",main="Within SS for k-means")  # Within SS is variation of errors
points(kclust,wss)
plot(kclust,bss/(wss+bss),type="l",main="R-Squared for k-means")  # R-Squared is ratio of explained variation
points(kclust,bss/(wss+bss))

# choose a specific value of k and focus on understanding this solution
k=10
(grpA=grpQ[[which(kclust==k)]])  # copy the previously computed results, same as grpA=kmeans(smat_dtm,k,nstart=5)
knames=as.character(1:k)

# check number of movies in each cluster
( result=table(grpA$cluster) )
barplot(result,xlab="Cluster",ylab="Count",main="Frequency of Clusters for k=10",horiz=TRUE)

# plot the solutions against two columns
par(mfrow=c(1,1),mar=c(5,4,4,1))
plot(jitter(smat_dtm[,"sci-fi"]), jitter(smat_dtm[,"action"]),xlab="sci-fi",ylab="action",col=grpA$cluster)
points(grpA$centers[,c("sci-fi","action")],col=1:k,pch=8,cex=2)
legend("topright",pch=8,bty="n",col=1:k,knames)

# summarize the centroids
round(grpA$centers[,toptags],2)   # print the centroid values for each cluster
# try parallel lines plot using top tags
parallelplot(grpA$centers[,toptags],auto.key=list(text=knames,space="top",columns=4,lines=T))
# create a parallel plot to visualize the centroid values (too hard to see anything, so just randomly sample 10 users)
parallelplot(grpA$centers[,sample(1:ncol(grpA$centers),20)],auto.key=list(text=knames,space="top",columns=4,lines=T))

# check the relationship between the genres and the clusters
table(moviedata$genre,grpA$cluster)

# print the list of movies in each group
morder=order(grpA$cluster)
grpA$cluster[morder]  # print the movie names by their cluster



###############################################################################
### @topicmodel
###############################################################################

# setup the parameters for LDA control vector
burnin=1000     # number of initial iterations to discard for Gibbs sampler (for slow processors use 500)
iter=5000       # number of iterations to use for estimation  (for slow processors use 1000)
thin=50         # only save every 50th iteration to save on storage
seed=list(203,5,63,101,765)  # random number generator seeds
nstart=5        # number of repeated random starts
best=TRUE       # only return the model with maximum posterior likelihood
set.seed(142123) # initialize the random number generator so we all get the same results

# estimate a series of LDA models (each run can take a few minutes depending upon your processor)
ktopics=10
ClusterOUT = LDA(mat_dtm,ktopics,method="Gibbs",control=list(nstart=nstart,seed=seed,best=best,burnin=burnin,iter=iter,thin=thin))
#load("Movie_Data_LDA10.RData")  # retrieve previously estimated version (to speed up analysis)
knames=as.character(1:ktopics)

# probability of topic assignments (each movie has its own unique profile)
ClustAssign = ClusterOUT@gamma   # this is a matrix with the row as the movie and column as the topic
rownames(ClustAssign)=moviedata$short_name
dim(ClustAssign)  # check the dimension (#Movies x K)
write.table(ClustAssign,file="topics_allmovies.txt",sep="\t")
ClustBest = apply(ClustAssign,1,which.max)  # determine the best guess of a cluster, a vector with best guess
head(cbind(round(ClustAssign,2),ClustBest),n=10)   # show the actual topic probabilities and best guess associated with the first 10 movies
#table(ClustBest,moviedata$genre)

# matrix with probabilities of each term per topic
ClustTopics = exp(ClusterOUT@beta)
colnames(ClustTopics)=colnames(mat_dtm)
dim(ClustTopics)  # check the dimension (K x #Tags)
write.table(t(ClustTopics),file="topics_allterms.txt",sep="\t")  # save to a file to import into Excel
# show the topics and associated terms
parallelplot(ClustTopics[,toptags],main="Topic associated with top Tags",auto.key=list(text=knames,space="top",columns=3,lines=T))
round(head(ClustTopics[,toptags]),3)   # print topics in columns and probabilities in rows

# visualize our LDA result
serVis(topicmodels2LDAvis(ClusterOUT))

# show the topics associated with a selected movie
iname="Titanic"
iname2="Transformers: Revenge"
iname3="The Hunger Games: Catc"
iname4="The Twilight Saga: New"
iname5="The Maze Runner"
barplot(ClustAssign[iname,],names.arg=knames,main=paste("Topics Associated for",iname5))  # !! change iname to iname5 to view others

# compute euclidean distance between topics of the movies with ClustAssign
mdist=dist(ClustAssign)
mdist[[c(iname,iname2,iname3,iname4,iname5)]]  # distance between select movies

# last we can try to predict the vector of original tags using our topic model
totaltags=apply(mat_dtm,1,sum)   # compute the sum of each of the rows (# of terms per movie)
ClustGuess=(ClustAssign%*%ClustTopics)*totaltags  # determine the best guess for each movie/term combination
# we can compare the predictions for a selected movie
mcompare=cbind(actual=as.vector(mat_dtm[iname5,]),pred=ClustGuess[iname5,])
cor(mcompare)  # overall comparison using correlation
print(mcompare[mcompare[,"actual"]>2,])  # compare actual words used more than 2 times
print(mcompare[mcompare[,"pred"]>2,])  # compare words predicted to be used more than 2 times
print(mcompare[toptags,])  # compare using top words overall
write.table(mcompare,file="topics_prediction.txt",sep="\t")

# plot selected topics for topmovies
#plot(ClustAssign[1:20,c(5,6)],xlab="Topic5",ylab="Topic6")
#text(ClustAssign[1:20,c(5,6)],labels=rownames(ClustAssign[1:10,]),cex=.5)