loadPackages<-function(requiredPackages){
    not_installed <- requiredPackages[!(requiredPackages %in% installed.packages()[ , "Package"])]      # Extract not installed packages
    if(length(not_installed)) install.packages(not_installed, repos = "https://cran.r-project.org/")    # Installs packages if not yet installed
    for(rp in requiredPackages){
        if (!require(rp, character.only = TRUE)) {
            library(rp)                                                                                 # Load packages
        }
    }
}


# The 3d column of df should contain values corresponding to measure
nemenyi <- function(df, measure, pvalue, outputPath, title){
    print('This is R code!')
    packages <- c("ggplot2","xgboost","mlr3","mlr3benchmark","mlr3learners")
    loadPackages(packages)     # Install and/or load packages

    df$learners <- factor(df$learners)
    df$tasks <- factor(df$tasks)
    df[,3] <- as.numeric(df[,3])

    bm = as.BenchmarkAggr(df, task_id = "tasks", learner_id = "learners", independent = TRUE, strip_prefix = FALSE)

    bm$friedman_test()
    ph = bm$friedman_posthoc(meas = measure, p.value = pvalue) # Nemenyi post-hoc analysis

    ranks = bm$rank_data()
    scores = rowMeans(ranks[,c(-1)])

    CD_style1 = autoplot(bm, type = "cd",  meas = measure, test = "nemenyi", p.value = pvalue, style = 1, minimize = FALSE) + theme_bw()
    CD_style2 = autoplot(bm, type = "cd",  meas = measure, test = "nemenyi", p.value = pvalue, style = 2, minimize = FALSE) + theme_bw()
    ggsave(paste(title, "_CD_style1.png", sep =""), CD_style1, path=outputPath)
    ggsave(paste(title, "_CD_style2.png", sep =""), CD_style2, path=outputPath)

    fn = autoplot(bm, type = "fn", meas = measure, p.value = pvalue, minimize = FALSE)
    mean = autoplot(bm, type = "mean", meas = measure, minimize = FALSE)
    box = autoplot(bm, type = "box", meas = measure, minimize = FALSE)
    ggsave(paste(title, "_fn.png", sep =""), fn, path=outputPath)
    ggsave(paste(title, "_mean.png", sep =""), mean, path=outputPath)
    ggsave(paste(title, "_box.png", sep =""), box, path=outputPath)

    return(scores)
}