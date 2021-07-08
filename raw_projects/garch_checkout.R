

# ARCH 1 -------------------------------------------------------------------
arch1 <- 
    tibble(       t = 0:99,
            epsilon = rnorm(n=100),
                 a0 = runif(1),
                 a1 = runif(1),
              sigma = NA,
      sigma_squared = NA,
                r_t = 0
          )

for( i in 2:100 ){
  
    arch1$sigma_squared[i] <- arch1$a0[1] + arch1$a1[1] * arch1$r_t[i-1]^2
    arch1$sigma[i] <- sqrt( arch1$sigma_squared[i] ) 
    arch1$r_t[i] <- arch1$sigma[i] * arch1$epsilon[i]
}

plot( x = arch1$t, y = arch1$r_t, 
      main = paste0( "a0 = ", arch1$a0[1], "   a1 = ", arch1$a1[1] ),
      lwd = 2, 
      type = "l"
    )
lines( x = arch1$t, y = arch1$epsilon, col=2 )
lines( x = arch1$t, y = arch1$sigma, col = 3 )
abline( h=0, col="grey" )
    
 
#  GARCH(1,1) ------------------------------------------------

a0 <- runif(1)
a1 <- runif(1)
b1 <- runif(1)
if( a1 + b1 >= 1 ){ b1 <- 1 - a1 - 0.05}
garch11 <- 
  tibble(       t = 0:99,
                epsilon = rnorm(n=100),
                a0 = a0,
                a1 = a1,
                b1 = b1,
                sigma = NA,
                sigma_squared = 1,
                r_t = 0
  )

for( i in 2:100 ){
  
  garch11$sigma_squared[i] <- garch11$a0[1] + 
                              garch11$a1[1] * garch11$r_t[i-1]^2 +
                              garch11$b1[1] * garch11$sigma_squared[i-1]
  garch11$sigma[i] <- sqrt( garch11$sigma_squared[i] ) 
  garch11$r_t[i] <- garch11$sigma[i] * garch11$epsilon[i]
}

plot( x = garch11$t, y = garch11$r_t, 
      main = bquote( sigma[t]^2*" = "*.(garch11$a0[1])*" + "*.(garch11$a1[1])*r[t-1]^2*" + "*.(garch11$b1[1])*sigma[t-1]^2),
      lwd = 2, 
      type = "l"
)
lines( x = garch11$t, y = garch11$epsilon, col=2 )
lines( x = garch11$t, y = garch11$sigma, col = 3 )
abline( h=0, col="grey" )
