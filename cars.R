tt=read.csv('~/Desktop/imports-85.data.txt', na.strings = "?"
)

names(tt)=c("symboling","normalized-losses","make","fuel-type","aspiration","num-of-doors","body-style","drive-wheels","engine-location","wheel-base","length","width","height","curb-weight","engine-type","num-of-cylinders","engine-size","fuel-system","bore","stroke","compression-ratio","horsepower","peak-rpm","city-mpg","highway-mpg","price")


tt2=tt[! is.na(tt[['price']]),]

relevant= tt2[,c("make","fuel-type","aspiration","num-of-doors","body-style","drive-wheels","engine-location","wheel-base","length","width","height","curb-weight","engine-type","num-of-cylinders","engine-size","fuel-system","bore","stroke","compression-ratio","horsepower","peak-rpm","city-mpg","highway-mpg","price")]



relevant2= lapply(relevant, function(x)  {if (is.numeric(x) && (x != relevant[['price']])) {as.numeric(scale(x)); print (x)} else {as.factor(x)}})

relevant2[['price']]= as.numeric(as.character(relevant2$price))
summary(relevant2$price)
relevant[1,]

lapply(relevant2, class)
class(relevant2)
relevant2=data.frame(relevant2)
modi=lm(price ~ ., data=relevant2, na.action=na.omit)
ani=anova(modi)
sort(modi$coefficients, decreasing=TRUE)

