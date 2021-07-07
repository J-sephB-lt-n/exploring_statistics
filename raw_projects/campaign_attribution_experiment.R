

# the_truth <- 
#   tibble( customer_base_type = c( "young_optimistic",
#                                   "young_grumpy",
#                                   "old_optimistic",
#                                   "old_grumpy"
#                                 ),
#           base_prob = round( runif( 4, 0.1, 0.5 ), digits=2 ),
#           campaign1_effect = round( runif(4, 0, 0.2), digits=2 ),
#           campaign2_effect = round( runif(4, 0, 0.2), digits=2 ),
#           campaign3_effect = round( runif(4, 0, 0.2), digits=2 ),
#           campaign4_effect = round( runif(4, 0, 0.2), digits=2 ),
#           
#         )
# 
# n <- 1000
# run_experiment <- 
#   tibble(        order_id = 1:n,
#           customer_base_type = sample(       x = the_truth$customer_base_type,
#                                           size = n,
#                                        replace = TRUE,
#                                           prob = c( 0.25,
#                                                     0.25,
#                                                     0.25,
#                                                     0.25
#                                                   )
#                                      )   
#         ) %>% 
#     # assign campaigns to each person: 
#     mutate( campaign1 = sample( 0:1, size=n(), replace=TRUE, prob=c(0.5,0.5) ),
#             campaign2 = sample( 0:1, size=n(), replace=TRUE, prob=c(0.5,0.5) ),
#             campaign3 = sample( 0:1, size=n(), replace=TRUE, prob=c(0.5,0.5) ),
#             campaign4 = sample( 0:1, size=n(), replace=TRUE, prob=c(0.5,0.5) )
#           ) %>% 
#     # calculate true prob:
#     left_join( the_truth ) %>% 
#     mutate( true_prob = base_prob * 
#                         (1 + campaign1*campaign1_effect) * 
#                         (1 + campaign2*campaign2_effect) *
#                         (1 + campaign3*campaign3_effect) *
#                         (1 + campaign4*campaign4_effect) 
#           ) %>% 
#     
#     # decide at random which orders happen based on their true_prob:
#     mutate( rand01 = runif(n()) ) %>% 
#     mutate( order_happened = if_else( true_prob > rand01,
#                                       1,
#                                       0
#                                     )
#           )
# 
# # size of unique groups:
# run_experiment %>% 
#   count( customer_base_type, campaign1, campaign2, campaign3, campaign4, true_prob ) %>% 
#   arrange( desc(n) ) %>% 
#   View()
# 
# run_model <- glm( formula = order_happened ~ campaign1 + campaign2 + campaign3 + campaign4,
#                    family = binomial(link="logit"),
#                      data = run_experiment  
#                 )
# 
# exp( coef(run_model) )
# the_truth
# 
# campaign_attribution <- 
#   tibble( campaign = c("(none)",paste("campaign",1:4,sep="")) ) %>% 
#   mutate( prob = c( predict(  object = run_model,
#                              newdata = tibble(campaign1=0,campaign2=0,campaign3=0,campaign4=0),
#                                 type = "response" 
#                            )
#                     ,
#                     predict(  object = run_model,
#                               newdata = tibble(campaign1=1,campaign2=0,campaign3=0,campaign4=0),
#                               type = "response" 
#                            )
#                     ,
#                     predict(  object = run_model,
#                               newdata = tibble(campaign1=0,campaign2=2,campaign3=0,campaign4=0),
#                               type = "response" 
#                            )
#                     ,
#                     predict(  object = run_model,
#                               newdata = tibble(campaign1=0,campaign2=0,campaign3=3,campaign4=0),
#                               type = "response" 
#                            )
#                     ,
#                     predict(  object = run_model,
#                               newdata = tibble(campaign1=0,campaign2=0,campaign3=3,campaign4=1),
#                               type = "response" 
#                     )                    
#                   ) 
#         ) %>% 
#   mutate( attribute_prop = prob / sum(prob) )
# 
# campaign_attribution

n <- 1000
run_experiment <- 
  tibble(  order_id = 1:n,
          base_prob = runif(   n = n, 
                             min = 0,
                             max = 0.2
                           ) %>% round(2),
          c1_effect = runif(n,-0.03,0.115) %>% round(2),
          c2_effect = runif(n,-0.03,0.2) %>% round(2),
          c3_effect = runif(n,-0.03,0.25) %>% round(2),
          had_c1 = sample(0:1, size=n, replace=TRUE),
          had_c2 = sample(0:1, size=n, replace=TRUE),
          had_c3 = sample(0:1, size=n, replace=TRUE)
        ) %>% 
  mutate( rand01 = runif(n())) %>%
  mutate( final_prob = base_prob * 
                       ( 1 +  c1_effect*had_c1 ) *
                       ( 1 +  c2_effect*had_c2 ) *
                       ( 1 +  c3_effect*had_c3 )
        ) %>% 
  mutate( ordered = if_else(final_prob > rand01, 1, 0) ) %>% 
  mutate( prob_without_camp1 = base_prob * 
                                   #( 1 +  c1_effect*had_c1 ) *
                                   ( 1 +  c2_effect*had_c2 ) *
                                   ( 1 +  c3_effect*had_c3 )
          ,
          prob_without_camp2 = base_prob * 
            ( 1 +  c1_effect*had_c1 ) *
            #( 1 +  c2_effect*had_c2 ) *
            ( 1 +  c3_effect*had_c3 )
          ,
          prob_without_camp3 = base_prob * 
            ( 1 +  c1_effect*had_c1 ) *
            ( 1 +  c2_effect*had_c2 ) #*
            # ( 1 +  c3_effect*had_c3 )
  ) %>% 
  mutate( attrib_organic = base_prob,
          attrib_camp1 = final_prob - prob_without_camp1,
          attrib_camp2 = final_prob - prob_without_camp2,
          attrib_camp3 = final_prob - prob_without_camp3
        ) %>% 
  mutate( final_attrib_denominator = attrib_organic + attrib_camp1 + attrib_camp2 + attrib_camp3,
          final_attrib_organic = ordered * ( attrib_organic / final_attrib_denominator ),
        final_attrib_campaign1 = ordered * ( attrib_camp1 / final_attrib_denominator ),
        final_attrib_campaign2 = ordered * ( attrib_camp2 / final_attrib_denominator ),
        final_attrib_campaign3 = ordered * ( attrib_camp3 / final_attrib_denominator )
        ) %>% 
  mutate_all( function(x){ ifelse( is.nan(x), 0, x ) } )
  
run_experiment

# see if GLM can get a similar attribution result:
fit_glm <- glm( ordered ~ had_c1 + had_c2 + had_c3,
                family = binomial(link="logit"),
                data = run_experiment
              )

est_base_prob <- predict(  object = fit_glm,
                          newdata = tibble(had_c1=0, had_c2=0, had_c3=0),
                             type = "response"
                       )

glm_calc <- 
  run_experiment %>% 
    # only keep successful orders:
    filter( ordered == 1 ) %>% 
    select( order_id,
            had_c1,
            had_c2,
            had_c3
          ) %>% 
    mutate( est_base_prob = est_base_prob, 
            est_c1_effect = exp(coef(fit_glm))[["had_c1"]],
            est_c2_effect = exp(coef(fit_glm))[["had_c2"]], 
            est_c3_effect = exp(coef(fit_glm))[["had_c3"]] 
          ) %>% 
    mutate( final_prob = est_base_prob * 
                         if_else( had_c1==1, est_c1_effect, 1) *
                         if_else( had_c2==1, est_c2_effect, 1) *
                         if_else( had_c3==1, est_c3_effect, 1)
          ) %>% 
    mutate( final_prob_without_c1 = 
              est_base_prob * 
              #if_else( had_c1==1, est_c1_effect, 1) *
              if_else( had_c2==1, est_c2_effect, 1) *
              if_else( had_c3==1, est_c3_effect, 1),
            final_prob_without_c2 = 
              est_base_prob * 
              if_else( had_c1==1, est_c1_effect, 1) *
              #if_else( had_c2==1, est_c2_effect, 1) *
              if_else( had_c3==1, est_c3_effect, 1),
            final_prob_without_c3 = 
            est_base_prob * 
              if_else( had_c1==1, est_c1_effect, 1) *
              if_else( had_c2==1, est_c2_effect, 1)
              #if_else( had_c3==1, est_c3_effect, 1)
          ) %>% 
  mutate( attrib_organic = est_base_prob,
          attrib_camp1 = final_prob - final_prob_without_c1,
          attrib_camp2 = final_prob - final_prob_without_c2,
          attrib_camp3 = final_prob - final_prob_without_c3
  ) %>% 
  mutate( final_attrib_denominator = attrib_organic + attrib_camp1 + attrib_camp2 + attrib_camp3,
          final_attrib_organic = attrib_organic / final_attrib_denominator,
          final_attrib_campaign1 = attrib_camp1 / final_attrib_denominator,
          final_attrib_campaign2 = attrib_camp2 / final_attrib_denominator,
          final_attrib_campaign3 = attrib_camp3 / final_attrib_denominator
  ) #%>% 
  #mutate_all( function(x){ ifelse( is.nan(x), 0, x ) } )
  

# summarise results:
paste0("\n Total Orders: ", sum(run_experiment$ordered), "\n") %>% cat()
paste0( "\n Median organic probability of ordering = ", median(run_experiment$base_prob), "\n" ) %>% cat()
paste0( "\n Model estimated base probability of ordering = ", round(est_base_prob,3), "\n" ) %>% cat()
paste0( "\n Total Orders attributable to Organic sales: ", 
        round(sum(run_experiment$final_attrib_organic)),
        "   (", 
        round( 100 * round(sum(run_experiment$final_attrib_organic)) / sum(run_experiment$ordered), digits=1 ), 
        "%) \n"
) %>% cat()
paste0( "\n MODEL ESTIMATE: Total Orders attributable to Organic sales: ", 
        round(sum(glm_calc$final_attrib_organic)),
        "   (", 
        round( 100 * round(sum(glm_calc$final_attrib_organic)) / sum(run_experiment$ordered), digits=1 ), 
        "%) \n"
) %>% cat()
paste0( "\n Total Orders attributable to campaign 1: ", 
        round(sum(run_experiment$final_attrib_campaign1)),
        "   (", 
        round( 100 * round(sum(run_experiment$final_attrib_campaign1)) / sum(run_experiment$ordered), digits=1 ), 
        "%) \n"
      ) %>% cat()
paste0( "\n MODEL ESTIMATE: Total Orders attributable to campaign 1: ", 
        round(sum(glm_calc$final_attrib_campaign1)),
        "   (", 
        round( 100 * round(sum(glm_calc$final_attrib_campaign1)) / sum(run_experiment$ordered), digits=1 ), 
        "%) \n"
) %>% cat()
paste0( "\n Total Orders attributable to campaign 2: ", 
        round(sum(run_experiment$final_attrib_campaign2)),
        "   (", 
        round( 100 * round(sum(run_experiment$final_attrib_campaign2)) / sum(run_experiment$ordered), digits=1 ), 
        "%) \n"
) %>% cat()
paste0( "\n MODEL ESTIMATE: Total Orders attributable to campaign 2: ", 
        round(sum(glm_calc$final_attrib_campaign2)),
        "   (", 
        round( 100 * round(sum(glm_calc$final_attrib_campaign2)) / sum(run_experiment$ordered), digits=1 ), 
        "%) \n"
) %>% cat()
paste0( "\n Total Orders attributable to campaign 3: ", 
        round(sum(run_experiment$final_attrib_campaign3)),
        "   (", 
        round( 100 * round(sum(run_experiment$final_attrib_campaign3)) / sum(run_experiment$ordered), digits=1 ), 
        "%) \n"
) %>% cat()
paste0( "\n MODEL ESTIMATE: Total Orders attributable to campaign 3: ", 
        round(sum(glm_calc$final_attrib_campaign3)),
        "   (", 
        round( 100 * round(sum(glm_calc$final_attrib_campaign3)) / sum(run_experiment$ordered), digits=1 ), 
        "%) \n"
) %>% cat()


