
library(dplyr)

customer_data <- 
  data_frame( id = 1:8,
              control_target = c( rep("control",4), rep("treatment",4) )
  )


library(shiny)

# Define UI for application that draws a histogram
ui <- fluidPage(

    # Sidebar with a slider input for number of bins
    sidebarLayout(
       sidebarPanel(
          sliderInput("id_1_preperiod", 
                      "ID 1: preperiod (control)", 
                      value = 0,
                      min = 0,
                      max = 100),
          sliderInput("id_1_campaign", 
                       "ID 1: campaign (control)", 
                       value = 0,
                       min = 0,
                       max = 100),
          sliderInput("id_2_preperiod", 
                      "ID 2: preperiod (control)", 
                      value = 0,
                      min = 0,
                      max = 100),
          sliderInput("id_2_campaign", 
                      "ID 2: campaign (control)", 
                      value = 0,
                      min = 0,
                      max = 100),
          sliderInput("id_3_preperiod", 
                      "ID 3: preperiod (control)", 
                      value = 0,
                      min = 0,
                      max = 100),
          sliderInput("id_3_campaign", 
                      "ID 3: campaign (control)", 
                      value = 0,
                      min = 0,
                      max = 100),
          sliderInput("id_4_preperiod", 
                      "ID 4: preperiod (control)", 
                      value = 0,
                      min = 0,
                      max = 100),
          sliderInput("id_4_campaign", 
                      "ID 4: campaign (control)", 
                      value = 0,
                      min = 0,
                      max = 100),
          sliderInput("id_5_preperiod", 
                      "ID 5: preperiod (TREATMENT)", 
                      value = 0,
                      min = 0,
                      max = 100),
          sliderInput("id_5_campaign", 
                      "ID 5: campaign (TREATMENT)", 
                      value = 0,
                      min = 0,
                      max = 100),
          sliderInput("id_6_preperiod", 
                      "ID 6: preperiod (TREATMENT)", 
                      value = 0,
                      min = 0,
                      max = 100),
          sliderInput("id_6_campaign", 
                      "ID 6: campaign (TREATMENT)", 
                      value = 0,
                      min = 0,
                      max = 100),
          sliderInput("id_7_preperiod", 
                      "ID 7: preperiod (TREATMENT)", 
                      value = 0,
                      min = 0,
                      max = 100),
          sliderInput("id_7_campaign", 
                      "ID 7: campaign (TREATMENT)", 
                      value = 0,
                      min = 0,
                      max = 100),
          sliderInput("id_8_preperiod", 
                      "ID 8: preperiod (TREATMENT)", 
                      value = 0,
                      min = 0,
                      max = 100),
          sliderInput("id_8_campaign", 
                      "ID 8: campaign (TREATMENT)", 
                      value = 0,
                      min = 0,
                      max = 100),

          fluid = TRUE,
          position = "left"
    
       ),
    
       ##### MAIN PANEL #####
       mainPanel(
          plotOutput("distPlot"),
          textOutput("uplift_text")
             )
   )
)

# Define server logic required to draw a histogram
server <- function(input, output) {
   
   output$distPlot <- renderPlot({
  
     control_preperiod_mean <- 
        mean( c( input$id_1_preperiod, 
                 input$id_2_preperiod, 
                 input$id_3_preperiod, 
                 input$id_4_preperiod
               )
            )
     
     control_campaign_mean <- 
       mean( c( input$id_1_campaign, 
                input$id_2_campaign, 
                input$id_3_campaign, 
                input$id_4_campaign
              )
           )
     
     adjustment_factor <- control_campaign_mean / control_preperiod_mean
     
      par(mfrow=c(2,4), mar =c(2,2,1,1) )
      
      for( customer_id in 1:8 ){
        plot( x = c(1,2),
              y = c( input[[paste0("id_",customer_id,"_preperiod")]],
                     input[[paste0("id_",customer_id,"_campaign")]]
                   ),
              main = paste0( customer_id, 
                             "  ", 
                             toupper( unlist(customer_data[customer_id,"control_target"]) )
                            ),
              xlim = c(0.5,2.5),
              ylim = c(-100,200),
              type = "b",
              xaxt = "n",
              xlab = "",
              ylab = "",
              las = 2,
              pch = 16,
              cex = 2,
              lwd = 1.5
            )
        
        abline( h = c(0, 100), col="grey")
        
        #if( customer_id > 4 ){
          
        uplift <- input[[paste0("id_",customer_id,"_campaign")]] -
                  (adjustment_factor*input[[paste0("id_",customer_id,"_preperiod")]])
        
        text( x = 1.5, y = 180, labels = round(uplift,2), col = "green", cex = 2)
        
        # draw in the expected spend:
        points( x = 2,
                y = (adjustment_factor*input[[paste0("id_",customer_id,"_preperiod")]]),
                pch = "X",
                col = 3,
                cex = 2
              )
        #}
        # draw in CONTROL preperiod mean and campaign mean:
        lines( x = c(0.5,1),
               y = c(control_preperiod_mean, control_preperiod_mean),
               col = 2,
               lwd = 2
             )
        lines( x = c(2,2.5),
               y = c(control_campaign_mean, control_campaign_mean),
               col = 2,
               lwd = 2
        )
        lines( x = c(1,2),
               y = c(control_preperiod_mean, control_campaign_mean),
               col = 2,
               lty = 2
             )
      }
      
   })
   
   output$uplift_text <- renderText({
     
     control_preperiod_mean <-
       mean( c( input$id_1_preperiod,
                input$id_2_preperiod,
                input$id_3_preperiod,
                input$id_4_preperiod
       )
       )

     control_campaign_mean <-
       mean( c( input$id_1_campaign,
                input$id_2_campaign,
                input$id_3_campaign,
                input$id_4_campaign
       )
       )
     
     target_preperiod_mean <- 
       mean( c( input$id_5_preperiod,
                input$id_6_preperiod,
                input$id_7_preperiod,
                input$id_8_preperiod
              )
       )
     target_campaign_mean <- 
       mean( c( input$id_5_campaign,
                input$id_6_campaign,
                input$id_7_campaign,
                input$id_8_campaign
              )
       )

     adjustment_factor <- control_campaign_mean / control_preperiod_mean
     
      paste0( "total uplift = ",
              "(",  
              input$id_5_campaign,
              "+", input$id_6_campaign,
              "+", input$id_7_campaign,
              "+", input$id_8_campaign,
              ")",
              " - ",
              "(",
              input$id_5_preperiod,
              "+", input$id_6_preperiod,
              "+", input$id_7_preperiod,
              "+", input$id_8_preperiod,
              ")*",
              "(",
              input$id_1_campaign,
              "+", input$id_2_campaign,
              "+", input$id_3_campaign,
              "+", input$id_4_campaign,
              ")/",
              "(",
              input$id_1_preperiod,
              "+", input$id_2_preperiod,
              "+", input$id_3_preperiod,
              "+", input$id_4_preperiod,
              ")",
              " = [ ",
              target_campaign_mean,
              "-",
              "(",
              target_preperiod_mean, "*",
              control_campaign_mean, "/",
              control_preperiod_mean,
              ") ] * 4",
              " = ",
              (target_campaign_mean - (target_preperiod_mean*adjustment_factor))*4
            )
      
     
   })
   
}

# Run the application 
shinyApp(ui = ui, server = server)

