############################################################################################################################
##         Zur Alterspyramide zu rechnen
##############################################################################################################################
library(tidyverse)
library(eeptools) # um Alter zu berechnen
library(ggplot2)# f�r muster age pyramid

# empty global enviroment
rm(list = ls())

options(warn=-1)# warnung ausblenden
#############################################################################################################
# F�gen Sie die Eingabedaten zu Ihrem aktuellen Arbeitsverzeichnis hinzu und geben Sie den Pfad an
###########################################################################################################################################
#Input von andere Team _Condition_code=E84.0,E84.1,E84.80,E84.87,E84.88,E84.9,O80_2021-03-03_15-25-58
#data <- read.csv("r/projectathon/filename.csv")
###############################################################################################################
data_folder <- "./opt/train_data/"
result_folder <- "./opt/pht_results/"
# result_folder <- "./opt/train_data/" if local execution
print(paste(data_folder, "cord_input.csv", sep = ""))

data <- read.csv(paste(data_folder ,"cord_input.csv", sep = ""))# aus projektbereich ordner


# Eleminiere doppelte Patienten
data <- data %>% distinct(PatientIdentifikator, AngabeDiag1, .keep_all = TRUE)
data$PatientIdentifikator <- NULL

# Berechne Alter auf der grund von Geburtsdatum
data$AngabeAlter <- floor(age_calc(as.Date(data$AngabeGeburtsdatum), unit="years"))
data$AngabeGeburtsdatum <- NULL


#WRITE Mean Data for PHT and add up if available ---------------------------
data_pht_man = data %>% subset(AngabeGeschlecht=="m")
data_pht_woman = data %>% subset(AngabeGeschlecht=="f")

output_pht_df <- data.frame(
  sex = c("male", "female"),
  number = c(length(data_pht_man$AngabeAlter), length(data_pht_woman$AngabeAlter)),
  age_mean = c(mean(data_pht_man$AngabeAlter), mean(data_pht_woman$AngabeAlter))
)

#Check if there are previous results -> if yes add up
if (file.exists(paste(result_folder,"result_mean.csv", sep = ""))) {

  previous_mean_df <- read.csv2(paste0(result_folder,"result_mean.csv"))

  output_both <- previous_mean_df %>%
    # combine both datasets - add second dataframe in rows
    bind_rows(
      output_pht_df
    ) %>%
    # rename column "number" to preserve for calculation
    dplyr::rename(
      number_old = number
    ) %>%
    # apply following operaitions on grouped varaible sex
    group_by(sex) %>%
    # summarize table: compute mean and number
    dplyr::summarize(
      # sum up all numbers of patients
      number = sum(number_old),
      # compute new mean
      age_mean = sum(age_mean * number_old)/number
    )

  # overwrite vairable for storing
  output_pht_df <- output_both

  message("previous PHT result found -> Add up")
}

# print results for fun
print(output_pht_df)

# write results (combined or not) to csv
write.csv2(output_pht_df, "./opt/pht_results/result_mean.csv", row.names = FALSE)
#----------------------------------------------------------------------------


# Teile in Altersgruppen ein
data$AngabeAlter <- cut(data$AngabeAlter, breaks = c(0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120))

# Gruppiere
result  <- as.data.frame(data %>%
                           group_by(Einrichtungsidentifikator, AngabeDiag1, AngabeGeschlecht, AngabeAlter) %>%
                           summarise(Anzahl = n()))

# Entferne nicht benoetigte Spalten
result$TextDiagnose1 <- NULL
result$TextDiagnose2 <- NULL
result$AngabeDiag2 <- NULL


################## Um der Alterspyramid zu rechnen######################################################################
# Nehmen wir Geschlechht, Alter, Anzahl
############################################################################################################################
stratified <- result[,c('AngabeGeschlecht','AngabeAlter','Anzahl')]
stratified_female <- (data = stratified %>% subset(AngabeGeschlecht=="f"))
stratified_male <- (data = stratified %>% subset(AngabeGeschlecht=="m")) %>% transform(Anzahl = (data = stratified %>% subset(AngabeGeschlecht=="m"))$Anzahl * -1 )
stratified_wide <- rbind(stratified_female,stratified_male)

#Abkuerzung ändern statt "f", "female" und statt "m" "male" verwenden
stratified_wide$AngabeGeschlecht [stratified_wide$AngabeGeschlecht == "f"] <- "female"
stratified_wide$AngabeGeschlecht [stratified_wide$AngabeGeschlecht == "m"] <- "male"


#wenn im PHT zuvor daten erstellt wurde -> auslesen
if (file.exists(paste(result_folder,"result_table.csv", sep = ""))) {

  data_pht <- read.csv(paste(result_folder,"result_table.csv", sep = ""))#("pht/results.csv")

  data <- rbind(stratified_wide, data_pht)
  message("previous PHT result found -> Add up")
} else {

  message("No previous PHT result found -> Assume first")
}

#FUER PHT DAS GANZE RAUSSCHREIBEN - inklusive neuen Daten
write.csv(stratified_wide, "./opt/pht_results/result_table.csv")

#Labellen name als angabe
names(stratified_wide)[names(stratified_wide)== "AngabeAlter"] <- "ageG"
names(stratified_wide)[names(stratified_wide)== "Anzahl"] <- "Count"
names(stratified_wide)[names(stratified_wide)== "AngabeGeschlecht"] <- "gender"

#Alterspyramid kozipieren
g <- ggplot(stratified_wide,aes(x=Count,y=ageG,fill=gender))
g + geom_bar(stat="identity")