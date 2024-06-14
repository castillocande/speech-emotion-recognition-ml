# Aclaración
Recuerden que este repositorio es solo un ejemplo para demostrar una posible lógica para organizar archivos a la hora de realizar experimentos de machine learning. La idea no es que trabajen con esta misma estructura, sino que entiendan la lógica y puedan adaptar esta estructura a sus necesidades. Sientanse libres de crear mas carpetas en cualquier nivel, de sumar scripts, de dividir o reubicar funciones. Piensen en las pruebas que quieran realizar, en los valores que para ustedes van a ser variables, y en las cosas que van a dejar fijas. Como repaso, la estructura que definan debería permitirles: 

- Reproducir cualquier experimento, es decir, obtener exactamente los mismos resultados.
- Definir un nuevo experimento de manera sencilla a partir del uso de archivos de configuración.
- Respetar una estructura en la cual un script no genera resultados que se almacenan por encima de su nivel.
- Tener registro solamente de aquellas partes del repositorio que no sean bases de datos, resultados que puedo recrear, arhivos de testeo de debuggeo o archivos con información sensible como keys o contraseñas.
- Autodocumentar los experimentos para comprender de donde vienen los resultados.
- Separar el código de las configuraciones o los aspectos variables en sus experimentos.
