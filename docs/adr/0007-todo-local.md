# Todo se ejecuta en local, sin hosting remoto

La API, la interfaz y MLflow se levantan con Docker en esta máquina, publicados solo en `127.0.0.1`, y los runs de MLflow quedan en un SQLite local. El enunciado pide que el código se ejecute en la defensa, no un despliegue, y sin servicios remotos no hay credenciales, costes ni una demo que falle por algo ajeno.

**Consecuencia:** el MLflow de Docker empieza vacío, porque los runs reales están en `mlflow.db`, ignorado por git. El README explica cómo verlos. Quien quiera publicarlo tendrá que añadir autenticación, porque la API no la tiene.
