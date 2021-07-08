


par(mfrow = c(1,3))

large_scale_data <- rnorm( n=20, mean = 30, sd = 20)
small_scale_data <- c( 15, rnorm( n=18, mean = 3, sd = 3 ), -10)

plot(large_scale_data, pch=16, col=2, cex = 1, ylab="", main="lines show group means", 
     ylim = c( min(small_scale_data), max(large_scale_data)  ))
points(small_scale_data, pch=16, col=4, cex = 1)
abline( h = mean(large_scale_data), col=2, lty = 3)
abline( h = mean(small_scale_data), col=4, lty = 3)


subtract_mean_large_scale_data <- large_scale_data - mean(large_scale_data)
subtract_mean_small_scale_data <- small_scale_data - mean(small_scale_data)

plot( subtract_mean_large_scale_data, pch=16, col=2, cex = 1, ylab="", main="subtract mean")
points(subtract_mean_small_scale_data, pch=16, col=4, cex = 1)
abline( h = mean(subtract_mean_large_scale_data))
text( x = 10, y = min( subtract_mean_large_scale_data), 
      labels = "subtracting the mean gives each group a mean of 0")


scaled_large_data <- ( large_scale_data - mean(large_scale_data) ) / sd(large_scale_data) 
scaled_small_data <- ( small_scale_data - mean(small_scale_data) ) / sd(small_scale_data)

scaled_large_data - scale(large_scale_data)   # this is what the R function scale() with default 
                                              # parameters does


plot( scaled_small_data, pch=16, col=4, cex=1, ylab="", main="scaled/standardised")
points( scaled_large_data, pch = 16, col=2, cex=1)
abline( h = 0)
