require(dplyr)
require(ca) # correspondence analysis

# CA

# creating the data
ID <- sample( 1:340, 340, replace=FALSE )

investment <- c(
                  rep("fixed_deposit", 79),
                  rep("fixed_deposit", 58),
                  rep("fixed_deposit", 49),
                  rep("bond", 10),
                  rep("bond", 8),
                  rep("bond", 9),
                  rep("unit_trust", 12),
                  rep("unit_trust", 10),
                  rep("unit_trust", 19),
                  rep("options/futures", 10),
                  rep("options/futures", 34),
                  rep("options/futures", 42)
                 
               )

risk_profile <- c(
                   rep( "risk_averse", 79),
                   rep( "risk_neutral", 58),
                   rep( "risk_seeker", 49),
                   rep( "risk_averse", 10),
                   rep( "risk_neutral", 8),
                   rep( "risk_seeker", 9),
                   rep( "risk_averse", 12),
                   rep( "risk_neutral", 10),
                   rep( "risk_seeker", 19),
                   rep( "risk_averse", 10),
                   rep( "risk_neutral", 34),
                   rep( "risk_seeker", 42)
                )

data <- data_frame( ID, investment, risk_profile) %>% arrange(ID)
  
data %>% select( investment, risk_profile) %>% table() %>% addmargins()

data %>% select( investment, risk_profile) %>% table() %>% chisq.test()

ca <- data %>% select( investment, risk_profile) %>% table() %>% ca()
plot(ca)

# MCA

age <- c( rep( "young", 11), rep("middle-age", 30), rep("retired", 70),
           rep( "young", 20), rep("middle-age", 80), rep("retired", 10),
           rep( "young", 90), rep("middle-age", 19), rep("retired", 10)
          )

mca_data <- data %>% arrange( risk_profile) 
mca_data <- cbind(mca_data, age) %>% 
  arrange(ID) %>% 
  as.data.frame() %>%
  select(-ID) %>%
  mutate( investment = as.factor(investment),
          risk_profile = as.factor(risk_profile),
          age = as.factor(age))


mjca( mca_data ,lambda="Burt")$Burt %>% View
mjca( mca_data ,lambda="Burt") %>% plot

mca4  <-  mjca( mca_data, lambda = "Burt", nd = 5)
cats = apply( mca_data, 2, function(x) nlevels(as.factor(x)))

require(ggplot2)
# data frame with variable coordinates

# data frame for ggplot
mca4_vars_df = data.frame(mca4$colcoord, Variable = rep(names(cats), cats))
rownames(mca4_vars_df) = mca4$levelnames

# plot
ggplot(data = mca4_vars_df, 
       aes(x = X1, y = X2, label = rownames(mca4_vars_df))) +
  geom_hline(yintercept = 0, colour = "gray70") +
  geom_vline(xintercept = 0, colour = "gray70") +
  geom_text(aes(colour = Variable)) +
  ggtitle("MCA plot of variables using R package ca") 
