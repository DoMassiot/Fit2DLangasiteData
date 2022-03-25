# Fit2DLangasiteData

2D fit of element maps from TEM microscopy  
2022 03 25  
by Dominique Massiot - CEMHTI-CNRS UPR3079, Orléans France - https://www.cemhti.cnrs-orleans.fr/?nom=massiot   

2 scripts for simulation of 2D element maps from TEM microscopy
  + 'make zone dataset.py' to generate the zone datasets
  + 'Fit2D data.py' to fit the data with 2D gaussian lineshapes

## reference work

These scritp have been used to model the data described in the following publication :  
"Stabilisation of the trigonal langasite structure in Ca3Ga2-2xZnxGe4+xO14 (0 ≤ x ≤ 1) with partial ordering of three isoelectronic cations characterised by a multi-technique approach"  
by Haytem Bazzaoui, Cécile Genevois, Dominique Massiot, Vincent Sarou-Kanian, Emmanuel Veron, Sébastien Chenu, Přemysl Beran, Michael J. Pitcher and Mathieu Allix  
submitted to ??  
preprint available at ??

## 'make zone dataset.py'

to generate the zone datasets
takes the raw data and builds the zone_data datasets used by the fitting routine
this script has to be executed piror to [Fit2D data.py]

## 'Fit2D data.py' 
fits the elmental datasets generated by the script above and generates 2D controur plots and a summary of results.  
The fitting routine was derived from # https://scipython.com/blog/non-linear-least-squares-fitting-of-a-two-dimensional-data
