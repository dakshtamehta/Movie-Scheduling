#
# Script: Movie_Analysis.R
#
# R script for for analyzing movie similarities
#
# Requires the following files:
#  opus_movies.txt              Movie characteristics of wide releases from 2006-2014
#  opus_movielens_tags.txt      Keywords that describe the movie from MovieLens
#
# The data included for this exercise is for internal use only and
# may not be posted or distributed further.
# Specifically the file opus_movies.txt 
# is provided by The Numbers (http://www.the-numbers.com),
# powered by OpusData (http://www.opusdata.com).
# The opus_movielens_tags.txt is available from Movielens
# which is located at http://grouplens.org/datasets/movielens/latest
#




##################### setup environment  ######################

# setup environment
if (!require(plyr)) {install.packages("plyr"); library(plyr)}           # manipulating data
if (!require(slam)) {install.packages("slam"); library(slam)}           # sparse matrices
if (!require(NLP)) {install.packages("NLP"); library(NLP)}              # language processing
if (!require(tm)) {install.packages("tm"); library(tm)}                 # text models
if (!require(topicmodels)) {install.packages("topicmodels"); library(topicmodels)}   # topic models
if (!require(lattice)) {install.packages("lattice"); library(lattice)}  # extra graphics




##################### input the data  ######################

## read in the data

# in RStudio select Menu Bar --> Session --> Set Working Directory --> To Source File Directory
# or automatically set working directory to be that of the script
#setwd(dirname(rstudioapi::getActiveDocumentContext()$path))  # only works in Rstudio scripts
# set to your correct working directory
setwd("~/Documents/class/marketing analytics/cases/movies/data")

# read in movie datasets
movies=read.delim("opus_movies.txt",header=T,stringsAsFactors=FALSE)  # the Opus movie data
tags=read.delim("opus_movielens_tags.txt",header=T,stringsAsFactors=FALSE)  # just the tags from movielens

# strip non-ASCII characters from tags
tags$tag=iconv(tags$tag,from="UTF-8",to="ASCII",sub="")
tags=aggregate(count~odid+tag,data=tags,FUN=sum)  # aggregate count by movie and tag
tags=tags[tags$tag!="",]  # remove blanks
tags$tag=as.factor(tags$tag) # convert from string to factor
tags$odid=as.factor(tags$odid)  # convert from string to factor

## make modifications to the dataset

# strip non-ASCII characters from movies
movies$display_name=iconv(movies$display_name,from="UTF-8",to="ASCII",sub='')
# create a short version of the title
movies$short_name=strtrim(enc2native(as.character(movies$display_name)),20)
# change data formats
movies$release_date=as.Date(as.character(movies$release_date),format="%Y-%m-%d")
movies$release_month=format(movies$release_date,"%m")
movies$release_monthyear=format(movies$release_date,"%m-%Y")
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




##################### k-means clustering of movies based upon characteristics ######################

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




##################### hierarchical clustering of movies based upon keywords ######################

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




##################### estimate an LDA topic model using keywords  ######################

# setup the parameters for LDA control vector
burnin=1000      # number of initial iterations to discard for Gibbs sampler (for slow processors use 500)
iter=5000        # number of iterations to use for estimation  (for slow processors use 1000)
thin=50          # only save every 50th iteration to save on storage
seed=list(203,5,63,101,765)  # random number generator seeds
nstart=5         # number of repeated random starts
best=TRUE        # only return the model with maximum posterior likelihood
set.seed(142123) # initialize the random number generator so we all get the same results

# estimate a series of LDA models (each run can take a few minutes depending upon your processor)
ClusterOUT = LDA(mterms,6,method="Gibbs",control=list(nstart=nstart,seed=seed,best=best,burnin=burnin,iter=iter,thin=thin))




### analyze a particular model with k topics

#
# let's understand the topics
#

# matrix with probabilities of each term per topic
# this is a matrix whose rows are topics, columns are terms,
# and each element is the probability of a term for a specific topic
ClustTopics = exp(ClusterOUT@beta)
colnames(ClustTopics)=colnames(mterms)
dim(ClustTopics)   # nrows is the # of topics, and ncols is # of terms
write.table(t(ClustTopics),file="topics_terms.txt")   # save this data to a file that you can import to Excel if you want

# show the topics and associated terms
parallelplot(ClustTopics[,idxtopterms],main="Topic associated with selected Terms")
print(format(t(ClustTopics),digits=1,scientific=FALSE))   # print topics in columns and probabilities in rows

# to better understand the topics lets print out the 20 most likely terms used for each
results=matrix('a',nrow=20,ncol=6)
for (i in 1:6) {
  idxtopterms=order(ClustTopics[i,],decreasing=TRUE)  # find the indices associated with the topic
  topterms=ClustTopics[i,idxtopterms[1:20]]  # identify the terms with the highest probabilities
  results[,i]=names(topterms)                # save the names
}
print(results)
write.table(results,file="topics_top20terms.txt")    # save this data to a file that you can import to Excel if you want

#
# let's understand the movies
#

# probability of topic assignments (each movie has its own unique profile)
# rows are movies and columns are topics
ClustAssign = ClusterOUT@gamma   # this is a matrix with the row as the movie and column as the topic
rownames(ClustAssign)=umovies$display_name
dim(ClustAssign)
head(ClustAssign,n=10)   # show the actual topic probabilities associated with the first 10 movies

# to better understand the topics lets print out the 20 most likely movies for each
results=matrix('a',nrow=20,ncol=6)
for (i in 1:6) {
  idxtopmovies=order(ClustAssign[,i],decreasing=TRUE)   # find the indices associated with the topic
  topmovies=ClustAssign[idxtopmovies[1:20],i]   # identify the terms with the highest probaiblities
  results[,i]=names(topmovies)                # save the names
}
print(results)
write.table(results,file="topics_top20movies.txt")   # save this data to a file that you can import to Excel if you want


#
# let's compare our target movie with the others
#

# find the index associated with our target movie
imovie=(1:nrow(umovies))[umovies$display_name=="The Maze Runner"]
print(umovies[imovie,])

# show the topics associated with a selected movie
barplot(ClustAssign[imovie,],names.arg=1:ncol(ClustAssign),main=paste("Topics Associated with selected movie",umovies$display_name[imovie]))

# visualize the distribution of topics across the movies
boxplot(ClustAssign,xlab="Topic",ylab="Probability of Topic across Movies")

# we can compute the distance between a target movie and all other movies in the "topics" space
topicdist=ClustAssign-matrix(ClustAssign[imovie,],nrow=nrow(ClustAssign),ncol=ncol(ClustAssign),byrow=T)
topicdistss=sqrt(apply(topicdist^2,1,sum))  # let's take the root of the sum of square of the distance between movies
augmovies=cbind(umovies,topicdistss)  # add the distance to the original movie data frame
augmovies=augmovies[-imovie,]  # let's remove "The Maze Runner" from our set
idxorder=order(augmovies$topicdistss)  # get the index of movies by similarity
head(augmovies[idxorder,c("odid","topicdistss","display_name","release_date","genre","rating")])  # most similar movies
tail(augmovies[idxorder,c("odid","topicdistss","display_name","release_date","genre","rating")])  # most dissimilar movies
write.table(augmovies,file="topics_similar_themazerunner.txt")   # save this data to a file that you can import to Excel if you want

