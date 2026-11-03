# Calibración isotónica y umbral derivado de costes supuestos

El modelo base se entrena con lo anterior a 2016-10-01 y el calibrador se ajusta en 2016T4. El umbral es `t* = C_FP / (C_FP + C_FN)` sobre la probabilidad calibrada, de modo que la decisión se explica con una fórmula y se puede repetir con otros costes.

La isotónica gana en 2016T4, con un Brier de 0.1592 frente a 0.1602 de la sigmoide y 0.1622 sin calibrar, y supera la mejora mínima de 0.001 que fijó el protocolo. Sin `scale_pos_weight` ni `class_weight`, que distorsionarían la probabilidad.

**Los costes 1:3 son un supuesto, no datos del hotel.** Dan un umbral de 0.25, que en el test avisa del 77.3 % de las reservas y cae justo antes de un escalón de la isotónica: con un umbral algo mayor, el coste bajaría. No se corrige, porque moverlo mirando el test invalidaría la cifra. La interfaz permite probar otros costes.
