#install.packages("shiny")
library(shiny)

ui = fluidPage(
  titlePanel("My first shiny app"),
  
  sidebarLayout(
    sidebarPanel(),
    mainPanel(
      strong("Analysis of numeric variables:", align = "center"),
      strong("We can understand the distribution from these plots", align = "center"),
      img(src = "histograms.png", align = "centre")
    )
  )
)

server = function(input, output) {
  
}

shinyApp(ui = ui, server = server)

