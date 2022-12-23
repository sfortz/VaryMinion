loadPackages<-function(requiredPackages){
    not_installed <- requiredPackages[!(requiredPackages %in% installed.packages()[ , "Package"])]      # Extract not installed packages
    if(length(not_installed)) install.packages(not_installed, repos = "https://cran.r-project.org/")    # Installs packages if not yet installed
    for(rp in requiredPackages){
        if (!require(rp, character.only = TRUE)) {
            library(rp)                                                                                 # Load packages
        }
    }
}

multiCompare <- function(df, measure, pvalue, outputPath, title){

    packages <- c("tsutils")
    loadPackages(packages)     # Install and/or load packages

    for(i in 1:ncol(df)) {       # for-loop over columns
      df[,i] <- as.numeric(1-df[,i])
    }

    title <- paste(outputPath,title, sep ="/")
    title <- paste(title,".png", sep ="")

    resfactor = 4
    png(filename=title, res = 72*resfactor, height=480*resfactor, width=480*resfactor)
    nem = nemenyi(df, plottype = "vmcb", conf.level = 1.0 - pvalue) # Nemenyi post-hoc analysis
    dev.off()
}

boxplots <- function(df, outputPath, title){
    packages <- c("ggplot2")
    loadPackages(packages)     # Install and/or load packages

    df$learners <- factor(df$learners)
    df$tasks <- factor(df$tasks)
    df$metric <- as.numeric(df$metric)

    bp= ggplot(data=df, aes(x=learners, y=metric)) + geom_boxplot() #+ theme_light()

    ggsave(title, bp, path=outputPath)
}

commented <- function(){

    df$learners <- factor(df$learners)
    df$tasks <- factor(df$tasks)
    df[,3] <- as.numeric(df[,3])

    Name <- c("Jon", "Bill", "Maria", "Ben", "Tina")
    Age <- c(23, 41, 32, 58, 26)
    Age2 <- c(2, 57, 22, 18, 90)
    Age3 <- c(63, 12, 34, 86, 99)

    df <- data.frame(name=Name, age=c(Age, Age2, Age3))

    # create a boxplot by using geom_boxplot() function of ggplot2 package
    bp=ggplot(data=df, aes(x=Name)) #, aes(x="name", y="ages")) + geom_boxplot()

    ggplot(data=df, aes(x=name, y=age)) + geom_boxplot()

    ggsave(paste(title, "_boxplot.png", sep =""), bp, path=outputPath)

    return(scores)
}