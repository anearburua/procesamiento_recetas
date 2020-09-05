#!/usr/bin/env python

exec(open("init_transcripcion.py").read())

from transcibir_audio import reconocer_audio

# Consigue los archivos de audio de la base de datos, los descarga y transcribe a su correspondiente archivo de texto
def transcripcion_principal():

	# Crear carpetas para los archivos de texto y audio si no existen
	if not os.path.exists("transcripciones"):
		os.makedirs("transcripciones")
	if not os.path.exists("base_datos"):
		os.makedirs("base_datos")

	# Acceder a los archivos de Google Storage
	storage_client = storage.Client()
	blobs = storage_client.list_blobs("audio-recetas")
	# Crear un archivo de texto y descargar el audio por cada elemento en Google Storage
	for blob in blobs:
		name = blob.name
		storage_uri ='gs://audio-recetas/' + name
		file_path = os.path.join("base_datos",name)

		name_p = name.partition(".")[0]
		transcription_path = os.path.join("transcripciones",name_p)
		f = open(transcription_path,"w+")
		blob.download_to_filename(file_path)

		# Transcibir audio según su duración
		f=wave.open(file_path,'r')
		frames = f.getnframes()
		rate = f.getframerate()
		duration = frames / float(rate)

		reconocer_audio(storage_uri,duration,transcription_path)
	print("Se ha completado la transcripción de todos los archivos")

# Llamada al programa principal
if __name__ == "__main__":
    	transcripcion_principal()





