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
    print(nem)
}



boxplots <- function(df, outputPath, metric_name){
    packages <- c("ggplot2")
    loadPackages(packages)     # Install and/or load packages

    df$learners <- factor(df$learner)
    df$dataset <- factor(df$dataset)
    df$metric <- as.numeric(df$metric)

    order = c('BPIC15', 'BPIC20', 'claroline-dis_10', 'claroline-rand_10', 'claroline-dis_50', 'claroline-rand_50')

    bp = ggplot(data=df, aes(x=learners, y=metric))
    bp = bp + geom_boxplot(linewidth=0.2, outlier.size = 1)
    bp =  bp + facet_wrap(~factor(dataset, levels=order), ncol = 2)
    bp =  bp + theme(axis.text.x = element_text(angle = 90))
    bp = bp + labs(x="RNN Configurations", y=metric_name)
    bp = bp + theme(axis.text.x=element_text(size=9))
    #bp = bp + stat_summary(fun.y="mean",color="blue", shape=18, size=0.3)

    title <- paste("figure_", metric_name, sep ="")
    title <- paste(title,".png", sep ="")
    ggsave(title, bp, path=outputPath)
}
