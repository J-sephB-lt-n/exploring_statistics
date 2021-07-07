#
# This is a Shiny web application. You can run the application by clicking
# the 'Run App' button above.
#
# Find out more about building applications with Shiny here:
#
#    http://shiny.rstudio.com/
#

library(shiny)


ui <- fluidPage(
  titlePanel("censusVis"),
  
  sidebarLayout(
    sidebarPanel(
      helpText(" joe is cool \n"),
      
      selectInput("var", 
                  label = "Choose a variable to display",
                  choices = c("option 1", 
                              "option 2",
                              "option 3", 
                              "option 4"),
                  selected = "option 1"),
      
      sliderInput("range", 
                  label = "enter number range here",
                  min = 0, max = 69, value = c(17, 52)),
      
      textInput("user.text", h3("enter text here"), 
                value = "Enter text..."),
      
      dateRangeInput("dates", h3("enter D4te r4nge"))
      ),
    
    mainPanel(
      textOutput("selected_var"),
      textOutput("min_max"),
      textOutput("user_text"),
      textOutput("date_range"),
      
      "\n
      other possible renderings: \n
      \n renderDataTable	DataTable \n
      \n renderImage	images (saved as a link to a source file) \n
      \n renderPlot	plots \n
      \n renderPrint	any printed output \n
      \n renderTable	data frame, matrix, other table like structures \n
      \n renderText	character strings \n
      \n renderUI	a Shiny tag object or HTML"
    )
  )
)

server <- function(input, output) {
  
  output$selected_var <- renderText({ 
    paste(input$var, "selected")
  })
  
  output$min_max <- renderText({ 
    paste("range = ",
          input$range[1], ":", input$range[2])
  })
  
  output$user_text <- renderText({ 
    paste( input$user.text )
  })
  
  output$date_range <- renderText({ 
    paste( "daterange = ", input$dates[1], " : ", input$dates[2] )
  })
  
  
}

shinyApp(ui, server)
