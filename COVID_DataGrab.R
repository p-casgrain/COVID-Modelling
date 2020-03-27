library(magrittr)
library(lubridate)
library(stringr)
library(foreach)

# Set download directory
download.dir <- "~/Desktop/COVID-Modelling/CSV"
url.template <- "https://www.ecdc.europa.eu/sites/default/files/documents/COVID-19-geographic-disbtribution-worldwide-{YMD.date}.csv"

# Download CSVs
covid.url <- str_glue(url.template,YMD.date= {today()-1} %>% format("%Y-%m-%d"))
download.file(covid.url,covid.url %>% basename %>% file.path(DOWNLOAD_DIR,.))
