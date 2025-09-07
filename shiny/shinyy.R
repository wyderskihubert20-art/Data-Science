# app.R
library(shiny)
library(ggplot2)
library(dplyr)
library(DT)
library(randomForest)
library(shinythemes)
library(shinyWidgets)

# =============================
# Load Data & Train Model
# =============================
data <- read.csv("CC_CUSTOMERS_CLEANED.csv")
set.seed(42)
rf_model <- randomForest(
  Risk_Score ~ BALANCE + PURCHASES + PAYMENTS +
               Purchase_Installment_Ratio + Credit_Utilization +
               Payment_Ratio + TOTAL_SPENDING,
  data = data, ntree = 200
)

# =============================
# UI
# =============================
ui <- fluidPage(
  theme = shinytheme("flatly"),
  
  titlePanel("Credit Card Customer Insights & Risk Prediction"),
  
  sidebarLayout(
    sidebarPanel(
      h4("Predict Risk Score"),
      wellPanel(
        fluidRow(
          column(6, numericInput("balance", "Balance", value = 1000)),
          column(6, numericInput("purchases", "Purchases", value = 500))
        ),
        fluidRow(
          column(6, numericInput("payments", "Payments", value = 200)),
          column(6, numericInput("pir", "Purchase Installment Ratio", value = 0.2))
        ),
        fluidRow(
          column(6, numericInput("cu", "Credit Utilization", value = 0.3)),
          column(6, numericInput("pr", "Payment Ratio", value = 1))
        ),
        numericInput("total_spending", "Total Spending", value = 700),
        actionBttn("predict_btn", "Predict Risk Score", style = "fill", color = "primary", size = "md")
      )
    ),
    
    mainPanel(
      tabsetPanel(
        tabPanel("Overview", DTOutput("data_table")),
        tabPanel("Clusters", plotOutput("cluster_plot", height = "500px")),
        tabPanel("Prediction", 
                 fluidRow(
                   column(12, 
                          br(),
                          uiOutput("risk_box")
                   )
                 )
        )
      )
    )
  )
)

# =============================
# Server
# =============================
server <- function(input, output) {
  
  # Show data table
  output$data_table <- renderDT({
    datatable(data, options = list(pageLength = 10, scrollX = TRUE))
  })
  
  # Cluster plot
  output$cluster_plot <- renderPlot({
    ggplot(data, aes(x = BALANCE, y = PURCHASES, color = factor(Cluster))) +
      geom_point(alpha = 0.6, size = 3) +
      labs(color = "Cluster", x = "Balance", y = "Purchases") +
      theme_minimal(base_size = 14) +
      scale_color_brewer(palette = "Set2")
  })
  
  # Predict Risk Score
  observeEvent(input$predict_btn, {
    new_customer <- data.frame(
      BALANCE = input$balance,
      PURCHASES = input$purchases,
      PAYMENTS = input$payments,
      Purchase_Installment_Ratio = input$pir,
      Credit_Utilization = input$cu,
      Payment_Ratio = input$pr,
      TOTAL_SPENDING = input$total_spending
    )
    
    pred <- predict(rf_model, new_customer)
    
    # Assign risk category
    risk_category <- case_when(
      pred < 0.2 ~ "Low Risk",
      pred < 0.5 ~ "Medium Risk",
      TRUE ~ "High Risk"
    )
    
    output$risk_box <- renderUI({
      tagList(
        h4("Predicted Risk Score: ", round(pred, 4)),
        tags$div(
          style = paste0("padding: 20px; font-weight:bold; font-size:20px; color:white; background-color:",
                         ifelse(risk_category == "Low Risk","#28a745",
                         ifelse(risk_category == "Medium Risk","#ffc107","#dc3545")),"; border-radius:10px; text-align:center;"),
          risk_category
        )
      )
    })
  })
}

# =============================
# Run App
# =============================
shinyApp(ui, server)
