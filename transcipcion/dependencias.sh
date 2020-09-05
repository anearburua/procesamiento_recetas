#Exportar las credenciales a mano en la terminal
export GOOGLE_APPLICATION_CREDENTIALS="/home/ane/Escritorio/procesamiento_recetas/transcripcion/transcripciontexto.json"

#Instalar cloud speech y cloud storage
pip install --upgrade google-cloud-speech
pip install google-cloud-storage

printf "\n Se han instalado las dependencias para la transcripción \n"
