# La regla de 1 error estándar elige Random Forest

Entre los modelos cuya pérdida pareada frente al mejor cabe en un error estándar, gana el más simple, en un orden fijado antes de comparar: regresión logística, árbol, Random Forest, XGBoost y MLP. Con cuatro ventanas, las diferencias pequeñas son ruido, y elegir siempre la media más alta sería sobreajustar la selección.

Quedan dentro Random Forest, con 0.8172, y XGBoost, con 0.8162, a 0.0011 con un error estándar de 0.0035. Gana Random Forest.

**Considerado:** elegir por coste en lugar de por ROC-AUC. Se habría llegado a otro modelo, porque con costes 1:3 el campeón es el más caro en el test. La métrica principal se fijó antes y no se cambia después de verlo.
