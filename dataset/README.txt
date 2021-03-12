Csv files of the bottle data used for training and validation of the CANYON-MED neural networks.

List of acronyms:
cruise_name: name of the cruise 
pres: pressure (dbar)
temp: temperature (°C)
psal: salnity
psal_q: salinity flag
doxy: dissolved O2 (µmol/kg)
doxy_q: dissolved O2 flag

alkali: total alkalinity (µmol/kg)
alkali_q: total alkalinity flag 
tcarbn: total carbon (Dissolved Inorganic Carbon) (µmol/kg)
tcarbn_q: total carbon flag
phts_insi: pH on the total scale at in situ pressure and temperature
phts_ini_q: pH on the total scale at in situ pressure and temperature flag
nitrat: nitrates (NO3-) (µmol/kg)
nitrat_q: nitrates flag
phosphat: phosphates (PO43-) (µmol/kg)
phosphat_q: phosphates flag
silcat: silicates (Si(OH)4) (µmol/kg)
silcat_q: silicates flag

categ: traning/validation, use of the data in the neural network development

flag correspondance:
0: no flag
1: good data
2: probably good data
4: bad data
9: missing data